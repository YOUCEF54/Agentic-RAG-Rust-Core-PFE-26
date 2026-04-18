use pyo3::prelude::*;
use pdf_oxide::PdfDocument;
use pdf_oxide::pipeline::{TextPipeline, TextPipelineConfig};
use pdf_oxide::pipeline::reading_order::ReadingOrderContext;
use pdfium_render::prelude::*;
use pdf_oxide::ReadingOrder;
use pdf_oxide::pipeline::converters::OutputConverter;
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use once_cell::sync::OnceCell;
use arrow_array::{
    FixedSizeListArray, Float32Array, Float64Array, Int32Array, RecordBatch,
    RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::connection::CreateTableMode;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::connect;
use tokio::runtime::Runtime;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use pyo3::exceptions::PyRuntimeError;

const MIN_CHUNK_CHARS: usize = 80;
static EMBEDDER: OnceCell<Mutex<TextEmbedding>> = OnceCell::new();
static RUNTIME: OnceCell<Runtime> = OnceCell::new();


fn get_runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to build Tokio runtime")
    })
}

fn get_embedder() -> Result<&'static Mutex<TextEmbedding>, String> {
    EMBEDDER.get_or_try_init(|| {
        std::env::set_var("TOKENIZERS_PARALLELISM", "false");
        let options = InitOptions::new(EmbeddingModel::BGESmallENV15)
            .with_show_download_progress(false);
        TextEmbedding::try_new(options)
            .map(Mutex::new)
            .map_err(|e| format!("Fastembed init failed: {e:?}"))
    })
}

#[pyfunction]
fn load_embed_model() -> PyResult<()> {
    get_embedder()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    Ok(())
}

#[pyfunction]
fn embed_texts_rust(py: Python<'_>, texts: Vec<String>, embed_batch_size: usize) -> PyResult<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    // 1. Prevent panics/degradation from invalid batch sizes
    let batch_size = embed_batch_size.clamp(2,256);

    // 2. Release the GIL for the entire CPU-bound embedding process
    let embeddings = py.allow_threads(|| {
        let embedder = get_embedder()?;
        let guard = embedder.lock().map_err(|_| "Embedder mutex poisoned")?;

        guard.embed(texts, Some(batch_size))
             .map_err(|e| format!("Fastembed failed: {e}"))
    });

    embeddings.map_err(|e| PyRuntimeError::new_err(e))
}

#[pyfunction]
fn smart_chunker(text: String, max_chars: usize, overlap: usize) -> PyResult<Vec<String>> {
    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();
    let mut current_pos = 0;

    while current_pos < total {
        let mut end_pos = std::cmp::min(current_pos + max_chars, total);
        if end_pos < total {
            let lookback_range = if end_pos > current_pos + (max_chars / 2) {
                max_chars / 2
            } else {
                end_pos - current_pos
            };

            if let Some(pos) = chars[end_pos - lookback_range..end_pos]
                .iter()
                .rposition(|&c| c == '.' || c == '\n')
            {
                end_pos = (end_pos - lookback_range) + pos + 1;
            }
        }
        let chunk: String = chars[current_pos..end_pos].iter().collect();
        let trimmed = chunk.trim().to_string();
        if !trimmed.is_empty() {
            chunks.push(trimmed);
        }
        current_pos = end_pos;
        if current_pos < total && current_pos > overlap {
            // current_pos -= overlap;
            let next_pos = if current_pos > overlap { current_pos - overlap } else { end_pos };
            current_pos = std::cmp::max(next_pos, current_pos + 1); // Force forward progress
        }
    }
    Ok(chunks)
}

#[pyfunction]
fn load_pdf_pages_many(paths: Vec<String>) -> PyResult<Vec<String>> {
    let mut results: Vec<(usize, Vec<String>)> = paths
        .par_iter()
        .enumerate()
        .map(|(idx, path)| extract_pages_from_path(path).map(|pages| (idx, pages)))
        .collect::<Result<Vec<_>, String>>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;

    results.sort_by_key(|(idx, _)| *idx);

    let mut all_pages = Vec::new();
    for (_, pages) in results {
        all_pages.extend(pages);
    }

    Ok(all_pages)
}

fn extract_pages_from_path(path: &str) -> Result<Vec<String>, String> {
    let mut doc = PdfDocument::open(path)
        .map_err(|e| format!("Could not open PDF: {:?}", e))?;

    let config = TextPipelineConfig::default();
    let pipeline = TextPipeline::with_config(config.clone());
    let converter = pdf_oxide::pipeline::converters::MarkdownOutputConverter::new();

    let num_pages = doc.page_count()
        .map_err(|e| format!("Could not get page count: {:?}", e))?;

    let mut pages_text = Vec::new();

    for i in 0..num_pages {
        // 1. Pass the ColumnAware parameter directly to the extraction method
        if let Ok(spans) = doc.extract_spans_with_reading_order(i, ReadingOrder::ColumnAware) {
            
            // 2. The context now only needs the page number
            let context = ReadingOrderContext::default()
                .with_page(i as u32);

            if let Ok(ordered_spans) = pipeline.process(spans, context) {
                if let Ok(markdown) = converter.convert(&ordered_spans, &config) {
                    let trimmed = markdown.trim().to_string();
                    if !trimmed.is_empty() {
                        pages_text.push(trimmed);
                    }
                }
            }
        }
    }

    Ok(pages_text)
}

#[pyfunction]
fn load_pdf_pages_pdfium_many(paths: Vec<String>) -> PyResult<Vec<String>> {
    let results: Vec<(usize, Vec<String>)> = paths
        .par_iter()
        .enumerate()
        .map(|(idx, path)| {
            // Bindings MUST be created per-thread/per-file for C-FFI safety
            let bind = Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./"))
                .or_else(|_| Pdfium::bind_to_system_library())
                .map_err(|e| format!("Failed to bind to PDFium: {:?}", e))?;
                
            let pdfium = Pdfium::new(bind);
            let doc = pdfium
                .load_pdf_from_file(path, None)
                .map_err(|e| format!("Could not open PDF {}: {:?}", path, e))?;

            let mut pages_content = Vec::new();

            for page in doc.pages().iter() {
                if let Ok(text_ptr) = page.text() {
                    let raw_text = text_ptr.all();
                    let cleaned = clean_research_text(&raw_text);
                    if !cleaned.is_empty() && !is_reference_page(&cleaned) {
                        pages_content.push(cleaned);
                    }
                }
            }
            Ok((idx, pages_content))
        })
        .collect::<Result<Vec<_>, String>>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    let mut sorted_results = results;
    sorted_results.sort_by_key(|(idx, _)| *idx);
    
    Ok(sorted_results.into_iter().flat_map(|(_, p)| p).collect())
}

fn is_reference_page(text: &str) -> bool {
    let ref_lines = text.lines()
        .filter(|l| {
            let t = l.trim();
            t.starts_with('[') && t.len() > 3 && t.chars().nth(1).map(|c| c.is_ascii_digit()).unwrap_or(false)
        })
        .count();
    // If more than 40% of non-empty lines look like references, skip this page
    let total = text.lines().filter(|l| !l.trim().is_empty()).count();
    total > 0 && ref_lines * 100 / total > 40
}

fn clean_research_text(text: &str) -> String {
    // Split into paragraphs first — preserve them as structural units
    let paragraphs: Vec<&str> = text.split("\n\n").collect();

    let cleaned_paragraphs: Vec<String> = paragraphs
        .iter()
        .map(|para| {
            // Within each paragraph: remove control chars, fix hyphenation, join lines
            let no_ctrl: String = para
                .chars()
                .filter(|c| !c.is_control() || c.is_whitespace())
                .collect();

            let lines: Vec<&str> = no_ctrl.lines().map(|l| l.trim()).collect();
            let mut buf = String::new();

            for (i, &line) in lines.iter().enumerate() {
                if line.is_empty() { continue; }
                if line.ends_with('-') {
                    // dehyphenate: "infor-\nmation" → "information"
                    buf.push_str(&line[..line.len() - 1]);
                } else {
                    buf.push_str(line);
                    if i < lines.len() - 1 {
                        let last = line.chars().last().unwrap_or(' ');
                        if ".!?\":".contains(last) {
                            buf.push(' '); // sentence ended — single space is enough
                        } else {
                            buf.push(' '); // mid-sentence line break — just a space
                        }
                    }
                }
            }
            // Normalize spacing WITHIN the paragraph only
            buf.split_whitespace().collect::<Vec<_>>().join(" ")
        })
        .filter(|p| !p.is_empty())
        .collect();

    // Re-join paragraphs with double newline — preserved for split_sentences
    cleaned_paragraphs.join("\n\n")
}

fn split_sentences(text: &str) -> Vec<String> {
    let abbrevs = ["dr", "mr", "mrs", "prof", "fig", "et al", "e.g", "i.e", "vs", "no", "vol"];

    // First split on paragraph boundaries — these are hard boundaries
    let paragraphs: Vec<&str> = text.split("\n\n").map(str::trim).filter(|s| !s.is_empty()).collect();

    let mut sentences: Vec<String> = Vec::new();

    for paragraph in paragraphs {
        let bytes = paragraph.as_bytes();
        let mut start = 0;
        let mut i = 0;

        while i < bytes.len() {
            if bytes[i] == b'.' || bytes[i] == b'!' || bytes[i] == b'?' {
                let after = i + 1;
                if after < bytes.len() && (bytes[after] == b' ' || bytes[after] == b'\n') {
                    let word_start = paragraph[..i]
                        .rfind(|c: char| c.is_whitespace())
                        .map(|p| p + 1)
                        .unwrap_or(0);
                    let word_before = paragraph[word_start..i].to_lowercase();
                    let is_abbrev = abbrevs.iter().any(|&a| word_before == a)
                        || word_before.len() == 1;

                    if !is_abbrev {
                        let s = paragraph[start..=i].trim().to_string();
                        if !s.is_empty() { sentences.push(s); }
                        start = after;
                    }
                }
            }
            i += 1;
        }
        // Remainder of paragraph (or the whole paragraph if no sentence boundary found)
        let tail = paragraph[start..].trim().to_string();
        if !tail.is_empty() { sentences.push(tail); }
    }

    sentences
}

#[pyfunction]
fn sliding_window_chunker(
    text: String,
    max_chars: usize,
    overlap: usize,
    ) -> PyResult<Vec<String>> {
    let sentences = split_sentences(&text);
    let mut chunks: Vec<String> = Vec::new();
    let mut window: Vec<usize> = Vec::new();
    let mut window_len: usize = 0;

    for (idx, sent) in sentences.iter().enumerate() {
        // Guard: single sentence exceeds max_chars — emit alone, reset window
        if sent.len() > max_chars {
            if !window.is_empty() {
                let chunk = window.iter()
                    .map(|&i| sentences[i].as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                if chunk.len() >= MIN_CHUNK_CHARS { chunks.push(chunk); }
                window.clear();
                window_len = 0;
            }
            if sent.len() >= MIN_CHUNK_CHARS { chunks.push(sent.clone()); }
            continue;
        }

        if window_len + sent.len() + 1 > max_chars && !window.is_empty() {
            // Emit current window
            let chunk = window.iter()
                .map(|&i| sentences[i].as_str())
                .collect::<Vec<_>>()
                .join(" ");
            if chunk.len() >= MIN_CHUNK_CHARS { chunks.push(chunk); }

            // Slide: drop from front ONLY while:
            //   1. more than one sentence remains (always keep last as overlap seed)
            //   2. dropping it still leaves enough to fill the overlap budget
            while window.len() > 1 {
                let front_len = sentences[window[0]].len() + 1;
                // Would we still have >= overlap chars after dropping?
                if window_len.saturating_sub(front_len) >= overlap {
                    window_len = window_len.saturating_sub(front_len);
                    window.remove(0);
                } else {
                    break;
                }
            }
        }

        window_len += sent.len() + 1;
        window.push(idx);
    }

    // Flush final window
    if !window.is_empty() {
        let chunk = window.iter()
            .map(|&i| sentences[i].as_str())
            .collect::<Vec<_>>()
            .join(" ");
        if chunk.len() >= MIN_CHUNK_CHARS { chunks.push(chunk); }
    }

    Ok(chunks)
}
#[pyfunction]
fn load_pdf_pages_markdown(paths: Vec<String>) -> PyResult<Vec<String>> {
    let mut results: Vec<(usize, Vec<String>)> = paths
        .par_iter()
        .enumerate()
        .map(|(idx, path)| {
            extract_markdown_from_path(path).map(|pages| (idx, pages))
        })
        .collect::<Result<Vec<_>, String>>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;

    results.sort_by_key(|(idx, _)| *idx);

    let mut all_pages = Vec::new();
    for (_, pages) in results {
        all_pages.extend(pages);
    }
    Ok(all_pages)
}

fn extract_markdown_from_path(path: &str) -> Result<Vec<String>, String> {
    let mut doc = PdfDocument::open(path)
        .map_err(|e| format!("Could not open PDF: {:?}", e))?;

    let num_pages = doc.page_count()
        .map_err(|e| format!("Could not get page count: {:?}", e))?;

    let mut pages_md = Vec::new();

    for i in 0..num_pages {
        match doc.to_markdown(i, &Default::default()) {
            Ok(md) => {
                let trimmed = md.trim().to_string();
                if !trimmed.is_empty() {
                    pages_md.push(trimmed);
                }
            }
            Err(_) => continue, // skip pages that fail, don't crash
        }
    }

    Ok(pages_md)
}

fn extract_page_texts_with_numbers(path: &str) -> Result<Vec<(usize, String)>, String> {
    let mut doc = PdfDocument::open(path)
        .map_err(|e| format!("Could not open PDF: {:?}", e))?;

    let mut pages_text = Vec::new();
    let config = TextPipelineConfig::default();
    let pipeline = TextPipeline::with_config(config);

    let num_pages = doc.page_count()
        .map_err(|e| format!("Could not get page count: {:?}", e))?;

    for i in 0..num_pages {
        if let Ok(spans) = doc.extract_spans(i) {
            let context = ReadingOrderContext::default().with_page(i as u32);
            if let Ok(ordered_spans) = pipeline.process(spans, context) {
                let mut page_full_text = String::new();
                for span in ordered_spans {
                    page_full_text.push_str(&span.span.text);
                    page_full_text.push(' ');
                }
                let trimmed = page_full_text.trim().to_string();
                if !trimmed.is_empty() {
                    pages_text.push((i as usize + 1, trimmed));
                }
            }
        }
    }

    Ok(pages_text)
}

fn find_ascii_case_insensitive_positions(text: &str, needle: &str) -> Vec<usize> {
    let hay = text.as_bytes();
    let needle_bytes = needle.as_bytes();
    if needle_bytes.is_empty() || needle_bytes.len() > hay.len() {
        return Vec::new();
    }

    let mut positions = Vec::new();
    for (start, _) in text.char_indices() {
        if start + needle_bytes.len() > hay.len() {
            break;
        }
        let mut matched = true;
        for (i, n) in needle_bytes.iter().enumerate() {
            let b = hay[start + i];
            if !b.is_ascii() || b.to_ascii_lowercase() != n.to_ascii_lowercase() {
                matched = false;
                break;
            }
        }
        if matched {
            positions.push(start);
        }
    }
    positions
}

fn extract_snippet(text: &str, center_byte: usize, radius_chars: usize) -> String {
    let mut char_positions: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    char_positions.push(text.len());

    let center_char = match char_positions.binary_search(&center_byte) {
        Ok(idx) => idx,
        Err(idx) => idx.saturating_sub(1),
    };

    let start_char = center_char.saturating_sub(radius_chars);
    let end_char = std::cmp::min(center_char + radius_chars, char_positions.len().saturating_sub(1));

    let start_byte = char_positions[start_char];
    let end_byte = char_positions[end_char];
    text[start_byte..end_byte].trim().to_string()
}

#[pyfunction]
fn extract_visual_elements(path: String, max_pages: Option<usize>) -> PyResult<Vec<(String, usize, String)>> {
    let pages = extract_page_texts_with_numbers(&path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;

    let mut results: Vec<(String, usize, String)> = Vec::new();
    let mut seen: HashSet<(String, usize, String)> = HashSet::new();

    let keywords = vec![
        ("table", "table"),
        ("figure", "figure"),
        ("fig.", "figure"),
        ("fig ", "figure"),
        ("graph", "graph"),
        ("chart", "graph"),
        ("diagram", "figure"),
    ];

    for (page_num, text) in pages {
        if let Some(limit) = max_pages {
            if page_num > limit {
                break;
            }
        }

        for (needle, kind) in &keywords {
            for pos in find_ascii_case_insensitive_positions(&text, needle) {
                let center = pos + (needle.len() / 2);
                let snippet = extract_snippet(&text, center, 120);
                let item = (kind.to_string(), page_num, snippet);
                if seen.insert(item.clone()) {
                    results.push(item);
                }
            }
        }
    }

    Ok(results)
}

#[pyfunction]
fn lancedb_create_or_open(
    db_dir: String,
    table_name: String,
    texts: Vec<String>,
    vectors: Vec<Vec<f32>>,
    overwrite: bool,
    ) -> PyResult<()> {
    get_runtime()
        .block_on(async {
            let db = connect(&db_dir)
                .execute()
                .await
                .map_err(|e| format!("DB connect failed: {:?}", e))?;

            let batches = build_batches(&texts, &vectors)?;
            let mode = if overwrite {
                CreateTableMode::Overwrite
            } else {
                CreateTableMode::exist_ok(|b| b)
            };

            db.create_table(&table_name, Box::new(batches))
                .mode(mode)
                .execute()
                .await
                .map_err(|e| format!("Table create failed: {:?}", e))?;

            Ok::<(), String>(())
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

#[pyfunction]
fn lancedb_search(
    db_dir: String,
    table_name: String,
    query_vector: Vec<f32>,
    top_k: usize,
    ) -> PyResult<Vec<(String, f32)>> {
    get_runtime()
        .block_on(async {
            let db = connect(&db_dir)
                .execute()
                .await
                .map_err(|e| format!("DB connect failed: {:?}", e))?;

            let table = db
                .open_table(&table_name)
                .execute()
                .await
                .map_err(|e| format!("Open table failed: {:?}", e))?;

            let batches = table
                .query()
                .limit(top_k)
                .nearest_to(query_vector)
                .map_err(|e| format!("nearest_to failed: {:?}", e))?
                .execute()
                .await
                .map_err(|e| format!("Query execute failed: {:?}", e))?
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| format!("Collect failed: {:?}", e))?;

            parse_search_results(batches)
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

fn build_batches(
    texts: &[String],
    vectors: &[Vec<f32>],
    ) -> Result<RecordBatchIterator<std::vec::IntoIter<Result<RecordBatch, arrow_schema::ArrowError>>>, String> {
    if texts.is_empty() {
        return Err("No texts provided to build LanceDB table.".to_string());
    }
    if texts.len() != vectors.len() {
        return Err("Texts and vectors length mismatch.".to_string());
    }

    let dim = vectors[0].len();
    if dim == 0 {
        return Err("Vectors have zero dimension.".to_string());
    }
    for v in vectors.iter() {
        if v.len() != dim {
            return Err("Inconsistent vector dimensions.".to_string());
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            true,
        ),
    ]));

    let ids = Int32Array::from_iter_values(0..texts.len() as i32);
    let text_array = StringArray::from_iter_values(texts.iter().map(|s| s.as_str()));
    let mut flat = Vec::with_capacity(vectors.len() * dim);
    for v in vectors.iter() {
        flat.extend_from_slice(v);
    }
    let values = Float32Array::from(flat);
    let value_field = Arc::new(Field::new("item", DataType::Float32, true));
    let vector_array = FixedSizeListArray::try_new(
        value_field,
        dim as i32,
        Arc::new(values),
        None,
    )
    .map_err(|e| format!("{:?}", e))?;

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids), Arc::new(text_array), Arc::new(vector_array)],
    )
    .map_err(|e| format!("{:?}", e))?;

    let iter = vec![Ok(batch)].into_iter();
    Ok(RecordBatchIterator::new(iter, schema))
}

fn parse_search_results(batches: Vec<RecordBatch>) -> Result<Vec<(String, f32)>, String> {
    let mut results = Vec::new();

    for batch in batches {
        let schema = batch.schema();

        let text_idx = schema
            .index_of("text")
            .map_err(|_| "Missing 'text' column in results".to_string())?;
        let text_col = batch
            .column(text_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| "Invalid 'text' column type".to_string())?;

        let dist_idx = schema.index_of("_distance").ok();
        enum DistCol<'a> {
            F32(&'a Float32Array),
            F64(&'a Float64Array),
            None,
        }

        let dist_col = match dist_idx {
            Some(i) => {
                let col = batch.column(i);
                if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
                    DistCol::F32(arr)
                } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                    DistCol::F64(arr)
                } else {
                    DistCol::None
                }
            }
            None => DistCol::None,
        };

        for row in 0..batch.num_rows() {
            let text = text_col.value(row).to_string();
            let distance = match &dist_col {
                DistCol::F32(arr) => arr.value(row),
                DistCol::F64(arr) => arr.value(row) as f32,
                DistCol::None => 0.0,
            };
            results.push((text, distance));
        }
    }

    Ok(results)
}

#[pymodule]
fn rag_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_embed_model, m)?)?;
    m.add_function(wrap_pyfunction!(embed_texts_rust, m)?)?;
    m.add_function(wrap_pyfunction!(smart_chunker, m)?)?;
    m.add_function(wrap_pyfunction!(sliding_window_chunker, m)?)?;
    m.add_function(wrap_pyfunction!(load_pdf_pages_many, m)?)?;
    m.add_function(wrap_pyfunction!(load_pdf_pages_markdown, m)?)?;
    m.add_function(wrap_pyfunction!(lancedb_create_or_open, m)?)?;
    m.add_function(wrap_pyfunction!(lancedb_search, m)?)?;
    m.add_function(wrap_pyfunction!(load_pdf_pages_pdfium_many, m)?)?;
    m.add_function(wrap_pyfunction!(extract_visual_elements, m)?)?;
    Ok(())
}