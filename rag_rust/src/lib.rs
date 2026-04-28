use pyo3::prelude::*;
use pdf_oxide::PdfDocument;
use pdf_oxide::pipeline::{TextPipeline, TextPipelineConfig};
use pdf_oxide::pipeline::reading_order::ReadingOrderContext;
use pdfium_render::prelude::*;
use pdf_oxide::ReadingOrder;
use pdf_oxide::pipeline::converters::OutputConverter;
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::{Arc, Mutex}; // added Mutex
use once_cell::sync::OnceCell;
use arrow_array::{
    FixedSizeListArray, Float32Array, Float64Array, Int32Array, RecordBatch,
    RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::connection::CreateTableMode;
use lancedb::query::{ExecutableQuery, QueryBase}; // not-original
use lancedb::connect;
use lancedb::DistanceType; // added
use tokio::runtime::Runtime;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding}; // added
use pyo3::exceptions::PyRuntimeError;
use reqwest::Client;
// use reqwest::blocking::Client; // added
use serde_json::json;
use std::env; // added
use dotenvy::dotenv; // added


const MIN_CHUNK_CHARS: usize = 80;
static EMBEDDER: OnceCell<Mutex<TextEmbedding>> = OnceCell::new(); // added
static RUNTIME: OnceCell<Runtime> = OnceCell::new();
// zembed config
const ZE_EMBED_MODEL: &str = "zembed-1";
const ZE_API_BASE: &str = "https://api.zeroentropy.dev";


fn get_runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to build Tokio runtime")
    })
}

// added (local-embed)
fn get_embedder() -> Result<&'static Mutex<TextEmbedding>, String> {
    EMBEDDER.get_or_try_init(|| {
        std::env::set_var("TOKENIZERS_PARALLELISM", "false");
        let options = InitOptions::new(EmbeddingModel::BGELargeENV15)
            .with_show_download_progress(false);
        TextEmbedding::try_new(options)
            .map(Mutex::new)
            .map_err(|e| format!("Fastembed init failed: {e:?}"))
    })
}

// added (local-embed)
#[pyfunction]
fn load_embed_model_local() -> PyResult<()> {
    get_embedder()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    Ok(())
}

// changed func name from "load_embed_model_zembed" to "load_embed_model_zembed"
#[pyfunction]
fn load_embed_model_zembed() -> PyResult<()> {
    let _ = dotenvy::dotenv();
    std::env::var("ZEROENTROPY_API_KEY").map_err(|_| {
        PyRuntimeError::new_err(
            "ZEROENTROPY_API_KEY environment variable is not set. Set it before calling load_embed_model_zembed().",
        )
    })?;
    Ok(())
}

async fn ze_embed(
    texts: Vec<String>,
    input_type: &str,
    batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("ZEROENTROPY_API_KEY")
        .map_err(|_| "ZEROENTROPY_API_KEY env var not set".to_string())?;

    let client = Client::new();
    let url = format!("{ZE_API_BASE}/v1/models/embed");
    let mut all_embeddings = Vec::with_capacity(texts.len());

    for chunk in texts.chunks(batch_size.clamp(1, 256)) {
        let body = json!({
            "input": chunk,
            "input_type": input_type,
            "model": ZE_EMBED_MODEL,
            "encoding_format": "float"
        });

        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("ZeroEntropy API error {status}: {text}"));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("JSON parse error: {e}"))?;

        let results = data["results"]
            .as_array()
            .ok_or_else(|| format!("Missing 'results' in response: {data}"))?;

        for result in results {
            let embedding = result["embedding"]
                .as_array()
                .ok_or_else(|| "Missing 'embedding' field in result".to_string())?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| f as f32)
                        .ok_or_else(|| "Non-numeric value in embedding".to_string())
                })
                .collect::<Result<Vec<f32>, _>>()?;

            all_embeddings.push(embedding);
        }
    }

    Ok(all_embeddings)
}

// changed func name from ""
#[pyfunction]
fn embed_texts_rust_zembed(py: Python<'_>, texts: Vec<String>, embed_batch_size: usize) -> PyResult<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let embeddings = py.allow_threads(|| {
        get_runtime().block_on(ze_embed(texts, "document", embed_batch_size))
    });

    embeddings.map_err(|e| PyRuntimeError::new_err(e))
}

// added (local-embed)
#[pyfunction]
fn embed_texts_rust_local(py: Python<'_>, texts: Vec<String>, embed_batch_size: usize) -> PyResult<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    // 1. Prevent panics/degradation from invalid batch sizes
    let batch_size = embed_batch_size.clamp(2,256);

    // 2. Release the GIL for the entire CPU-bound embedding process
    let embeddings = py.allow_threads(|| {
        let embedder = get_embedder()?;
        let mut guard = embedder.lock().map_err(|_| "Embedder mutex poisoned")?;

        guard.embed(texts, Some(batch_size))
             .map_err(|e| format!("Fastembed failed: {e}"))
    });

    embeddings.map_err(|e| PyRuntimeError::new_err(e))
}


#[pyfunction]
fn embed_query_rust_zembed(py: Python<'_>, query: String) -> PyResult<Vec<f32>> {
    py.allow_threads(|| {
        get_runtime().block_on(async {
            let mut vecs = ze_embed(vec![query], "query", 1).await?;
            vecs.pop().ok_or_else(|| "Empty embedding response".to_string())
        })
    })
    .map_err(PyRuntimeError::new_err)
}

// deleted old chunking method "smart_chunker(...)"

// this func relyes on pdf_oxide, it is outdated, try using the pdfium based new function
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

// added
fn is_reference_page(text: &str) -> bool {
    let non_empty_lines: Vec<&str> = text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    
    if non_empty_lines.is_empty() { return false; }
    
    let ref_lines = non_empty_lines.iter().filter(|l| {
        l.starts_with('[') 
        && l.chars().nth(1).map(|c| c.is_ascii_digit()).unwrap_or(false)
    }).count();

    // Also catch pages starting with "REFERENCES" header
    let has_ref_header = non_empty_lines.iter()
        .take(3)
        .any(|l| l.to_uppercase().starts_with("REFERENCES"));

    has_ref_header || (ref_lines * 100 / non_empty_lines.len() > 35)
}

// added
fn is_reference_paragraph(para: &str) -> bool {
    let lines: Vec<&str> = para.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();

    if lines.is_empty() { return false; }

    // Header line: "REFERENCES" or "References" alone
    if lines.len() == 1 {
        let u = lines[0].to_uppercase();
        return u == "REFERENCES" || u == "BIBLIOGRAPHY";
    }

    let ref_lines = lines.iter().filter(|l| {
        l.starts_with('[')
            && l.len() > 3
            && l.chars().nth(1).map(|c| c.is_ascii_digit()).unwrap_or(false)
    }).count();

    // >60% of lines look like "[N] Author..." → reference paragraph
    ref_lines * 100 / lines.len() > 60
}


// added
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

// added
fn clean_research_text(text: &str) -> String {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();

    let cleaned_paragraphs: Vec<String> = paragraphs
        .iter()
        .filter(|para| !is_reference_paragraph(para.trim()))  // ← NEW filter
        .map(|para| {
            let no_ctrl: String = para
                .chars()
                .filter(|c| !c.is_control() || c.is_whitespace())
                .collect();

            let lines: Vec<&str> = no_ctrl.lines().map(|l| l.trim()).collect();
            let mut buf = String::new();

            for (i, &line) in lines.iter().enumerate() {
                if line.is_empty() { continue; }
                if line.ends_with('-') {
                    buf.push_str(&line[..line.len() - 1]);
                } else {
                    buf.push_str(line);
                    if i < lines.len() - 1 {
                        let last = line.chars().last().unwrap_or(' ');
                        if ".!?\":".contains(last) {
                            buf.push('\n');  // ← P6 fix: preserve sentence boundary
                        } else {
                            buf.push(' ');
                        }
                    }
                }
            }
            buf.split_whitespace().collect::<Vec<_>>().join(" ")
        })
        .filter(|p| !p.is_empty())
        .collect();

    cleaned_paragraphs.join("\n\n")
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

// added
fn split_sentences(text: &str) -> Vec<String> {
    let abbrevs = ["dr", "al.", "mr", "mrs", "prof", "fig", "et al", "e.g", "i.e", "vs", "no", "vol"];

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

// added
fn strip_academic_header(text: &str) -> &str {
    if let Some(first_break) = text.find("\n\n") {
        let first_para = &text[..first_break];
        if first_para.contains('@') && first_para.len() < 600 {
            return text[first_break..].trim_start();
        }
    }
    if text.contains('@') && text.len() < 600 {
        return "";
    }
    text
}

// mod
#[pyfunction]
fn sliding_window_chunker(
    text:      String,
    max_chars: usize,
    overlap:   usize,
    ) -> PyResult<Vec<String>> {
    let stripped = strip_academic_header(&text);
    if stripped.is_empty() {
        return Ok(Vec::new());
    }

    let sentences = split_sentences(stripped);
    if sentences.is_empty() {
        return Ok(Vec::new());
    }

    // Compute overlap in sentences, not chars — avoids the over-drop bug
    let avg_sent_len = sentences.iter().map(|s| s.len()).sum::<usize>() / sentences.len();
    let overlap_sents = ((overlap + avg_sent_len - 1) / avg_sent_len).max(1);

    let n = sentences.len();
    let mut chunks: Vec<String> = Vec::new();
    let mut start_idx: usize = 0;

    loop {
        if start_idx >= n { break; }

        // Greedily fill window up to max_chars
        let mut end_idx   = start_idx;
        let mut window_len = 0usize;

        while end_idx < n {
            let add = sentences[end_idx].len() + if end_idx > start_idx { 1 } else { 0 };
            if window_len + add > max_chars && end_idx > start_idx {
                break;
            }
            window_len += add;
            end_idx += 1;
        }

        // Emit
        let chunk = sentences[start_idx..end_idx].join(" ");
        if chunk.len() >= MIN_CHUNK_CHARS {
            chunks.push(chunk);
        }

        // Force progress if a single sentence exceeds max_chars
        if end_idx == start_idx {
            if sentences[start_idx].len() >= MIN_CHUNK_CHARS {
                chunks.push(sentences[start_idx].clone());
            }
            start_idx += 1;
            continue;
        }

        // Slide: next window starts overlap_sents before end, but always moves forward
        let next = end_idx.saturating_sub(overlap_sents);
        start_idx = next.max(start_idx + 1);
    }

    Ok(chunks)
}

// logic for cosine similarity
fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product: f32 = v1.iter().zip(v2).map(|(a, b)| a * b).sum();
    let norm_v1: f32 = v1.iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
    let norm_v2: f32 = v2.iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
    dot_product / (norm_v1 * norm_v2)
}

// Helper for dynamic thresholding (Finds the value at the p-th percentile)
fn calculate_dynamic_threshold(distances: &[f32], percentile: f32) -> f32 {
    if distances.is_empty() { return 0.0; }
    let mut sorted = distances.to_vec();
    // Sort safely (floats can be tricky, but cosine distances are usually well-behaved)
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let index = ((sorted.len() - 1) as f32 * percentile) as usize;
    sorted[index]
}

// Fallback: A simple version of your previous logic for oversized semantic chunks
fn simple_sliding_window(text: &str, max_chars: usize) -> Vec<String> {
    let sentences = split_sentences(text);
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_len = 0;

    for s in sentences {
        if current_len + s.len() > max_chars && !current_chunk.is_empty() {
            chunks.push(current_chunk.join(" "));
            current_chunk.clear();
            current_len = 0;
        }
        current_len += s.len();
        current_chunk.push(s);
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.join(" "));
    }
    chunks
}
use anyhow::Error;
#[pyfunction]
fn semantic_window_chunker_advanced(
    text: String,
    max_chars: usize,
    window_size: usize,
    threshold_percentile: f32,
    ) -> PyResult<Vec<String>> {
    let stripped = strip_academic_header(&text);
    let sentences = split_sentences(stripped);
    if sentences.len() < window_size {
        return Ok(vec![sentences.join(" ")]);
    }

    // FIX E0639: Initialize non-exhaustive struct
    let model_mutex = EMBEDDER.get_or_init(|| {
        let mut options = InitOptions::default();
        // options.model_name = EmbeddingModel::AllMiniLML6V2;
        // options.model_name = EmbeddingModel::SnowflakeArcticEmbedXS; 
        options.model_name = EmbeddingModel::MultilingualE5Small;
        let m = TextEmbedding::try_new(options)
            .expect("Failed to initialize FastEmbed model");
        Mutex::new(m)
    });
    // 2. Verrouillage du Mutex et extraction du garde
    let mut model_guard = model_mutex.lock().map_err(|e| {
        PyRuntimeError::new_err(format!("Mutex lock failed: {}", e))
    })?;

    // 1. Create Overlapping Windows
    let windows: Vec<String> = sentences
        .windows(window_size)
        .map(|w| w.join(" "))
        .collect();

    // 2. Embed Windows
    let window_embeddings = model_guard.embed(windows, None)
            .map_err(|e: anyhow::Error| PyRuntimeError::new_err(e.to_string()))?;
    // 3. Calculate "Distance" between consecutive windows
    let mut distances = Vec::new();
    for i in 0..window_embeddings.len() - 1 {
        let dist = 1.0 - cosine_similarity(&window_embeddings[i], &window_embeddings[i+1]);
        distances.push(dist);
    }

    // 4. Identify Break Points
    let threshold = calculate_dynamic_threshold(&distances, threshold_percentile);
    let mut break_points = Vec::new();
    for (i, &dist) in distances.iter().enumerate() {
        if dist > threshold {
            break_points.push(i + (window_size / 2)); 
        }
    }

    // 5. Final Assembly
    let mut chunks = Vec::new();
    let mut current_start = 0;
    
    for bp in break_points {
        if bp >= sentences.len() { continue; }
        let chunk_text = sentences[current_start..=bp].join(" ");
        
        if chunk_text.len() > max_chars {
            chunks.extend(simple_sliding_window(&chunk_text, max_chars));
        } else if !chunk_text.is_empty() {
            chunks.push(chunk_text);
        }
        current_start = bp + 1;
    }
    
    if current_start < sentences.len() {
        let tail = sentences[current_start..].join(" ");
        if tail.len() > max_chars {
            chunks.extend(simple_sliding_window(&tail, max_chars));
        } else {
            chunks.push(tail);
        }
    }

    Ok(chunks)
}
// outdated
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

// outdated
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

// outdated
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

// added


// ── schema now carries source (filename) and page number ──────────────────
fn build_batches(
    texts:   &[String],
    sources: &[String],   // NEW: PDF filename per chunk
    pages:   &[i32],      // NEW: 1-based page number per chunk
    vectors: &[Vec<f32>],
) -> Result<
    RecordBatchIterator<std::vec::IntoIter<Result<RecordBatch, arrow_schema::ArrowError>>>,
    String,
> {
    if texts.is_empty() {
        return Err("No texts provided.".to_string());
    }
    if texts.len() != vectors.len() || texts.len() != sources.len() || texts.len() != pages.len() {
        return Err(format!(
            "Length mismatch: texts={}, sources={}, pages={}, vectors={}",
            texts.len(), sources.len(), pages.len(), vectors.len()
        ));
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
        Field::new("id",     DataType::Int32, false),
        Field::new("source", DataType::Utf8,  false),  // NEW
        Field::new("page",   DataType::Int32, false),  // NEW
        Field::new("text",   DataType::Utf8,  false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            true,
        ),
    ]));

    let ids          = Int32Array::from_iter_values(0..texts.len() as i32);
    let source_array = StringArray::from_iter_values(sources.iter().map(|s| s.as_str()));
    let page_array   = Int32Array::from_iter_values(pages.iter().copied());
    let text_array   = StringArray::from_iter_values(texts.iter().map(|s| s.as_str()));

    let mut flat = Vec::with_capacity(vectors.len() * dim);
    for v in vectors.iter() {
        flat.extend_from_slice(v);
    }
    let values       = Float32Array::from(flat);
    let value_field  = Arc::new(Field::new("item", DataType::Float32, true));
    let vector_array = FixedSizeListArray::try_new(value_field, dim as i32, Arc::new(values), None)
        .map_err(|e| format!("{:?}", e))?;

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(ids),
            Arc::new(source_array),
            Arc::new(page_array),
            Arc::new(text_array),
            Arc::new(vector_array),
        ],
    )
    .map_err(|e| format!("{:?}", e))?;

    Ok(RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema))
}
// ── updated create/open — now accepts sources and pages ───────────────────
// #[pyfunction]
// fn lancedb_create_or_open(
//     db_dir:     String,
//     table_name: String,
//     texts:      Vec<String>,
//     sources:    Vec<String>,  // NEW
//     pages:      Vec<i32>,     // NEW
//     vectors:    Vec<Vec<f32>>,
//     overwrite:  bool,
//     ) -> PyResult<()> {
//     get_runtime()
//         .block_on(async {
//             let db = connect(&db_dir)
//                 .execute()
//                 .await
//                 .map_err(|e| format!("DB connect failed: {:?}", e))?;

//             let batches = build_batches(&texts, &sources, &pages, &vectors)?;

//             let mode = if overwrite {
//                 CreateTableMode::Overwrite
//             } else {
//                 CreateTableMode::exist_ok(|b| b)
//             };

//             db.create_table(&table_name, Box::new(batches))
//                 .mode(mode)
//                 .execute()
//                 .await
//                 .map_err(|e| format!("Table create failed: {:?}", e))?;

//             Ok::<(), String>(())
//         })
//         .map_err(|e| PyRuntimeError::new_err(e))
// }
// ── updated create/open — now handles APPENDING ───────────────────────────
#[pyfunction]
fn lancedb_create_or_open(
    db_dir:     String,
    table_name: String,
    texts:      Vec<String>,
    sources:    Vec<String>,
    pages:      Vec<i32>,
    vectors:    Vec<Vec<f32>>,
    overwrite:  bool,
) -> PyResult<()> {
    get_runtime()
        .block_on(async {
            let db = connect(&db_dir)
                .execute()
                .await
                .map_err(|e| format!("DB connect failed: {:?}", e))?;

            // Safety check: Don't do anything if there's no data to insert
            if vectors.is_empty() {
                return Ok::<(), String>(());
            }

            let batches = build_batches(&texts, &sources, &pages, &vectors)?;

            // Check if the table already exists
            let table_names = db.table_names().execute().await.unwrap_or_default();
            let table_exists = table_names.contains(&table_name);

            if overwrite || !table_exists {
                // OVERWRITE / CREATE MODE
                db.create_table(&table_name, Box::new(batches))
                    .mode(CreateTableMode::Overwrite)
                    .execute()
                    .await
                    .map_err(|e| format!("Table create failed: {:?}", e))?;
            } else {
                // APPEND MODE
                let table = db
                    .open_table(&table_name)
                    .execute()
                    .await
                    .map_err(|e| format!("Open table failed: {:?}", e))?;
                    
                table.add(Box::new(batches))
                    .execute()
                    .await
                    .map_err(|e| format!("Table add failed: {:?}", e))?;
            }

            Ok::<(), String>(())
        })
        .map_err(|e| PyRuntimeError::new_err(e))
}

// ── search now returns (text, source, page, distance) ─────────────────────
#[pyfunction]
fn lancedb_search(
    db_dir:       String,
    table_name:   String,
    query_vector: Vec<f32>,
    top_k:        usize,
) -> PyResult<Vec<(String, String, i32, f32)>> {   // ← tuple now has 4 fields
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
                .distance_type(DistanceType::Cosine)
                .execute()
                .await
                .map_err(|e| format!("Query execute failed: {:?}", e))?
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| format!("Collect failed: {:?}", e))?;

            parse_search_results(batches)
        })
        .map_err(|e| PyRuntimeError::new_err(e))
}

// mod
fn parse_search_results(
    batches: Vec<RecordBatch>,
) -> Result<Vec<(String, String, i32, f32)>, String> {
    let mut results = Vec::new();

    for batch in batches {
        let schema = batch.schema();

        // ── text ──────────────────────────────────────────────────────────
        let text_col = batch
            .column(schema.index_of("text").map_err(|_| "Missing 'text' column")?)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("Invalid 'text' column type")?;

        // ── source ────────────────────────────────────────────────────────
        let source_col = batch
            .column(schema.index_of("source").map_err(|_| "Missing 'source' column")?)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("Invalid 'source' column type")?;

        // ── page ──────────────────────────────────────────────────────────
        let page_col = batch
            .column(schema.index_of("page").map_err(|_| "Missing 'page' column")?)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or("Invalid 'page' column type")?;

        // ── distance (optional — LanceDB adds _distance automatically) ────
        enum DistCol<'a> {
            F32(&'a Float32Array),
            F64(&'a Float64Array),
            None,
        }
        let dist_col = match schema.index_of("_distance").ok() {
            Some(i) => {
                let col = batch.column(i);
                if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
                    DistCol::F32(a)
                } else if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
                    DistCol::F64(a)
                } else {
                    DistCol::None
                }
            }
            None => DistCol::None,
        };

        for row in 0..batch.num_rows() {
            let text     = text_col.value(row).to_string();
            let source   = source_col.value(row).to_string();
            let page     = page_col.value(row);
            let distance = match &dist_col {
                DistCol::F32(a) => a.value(row),
                DistCol::F64(a) => a.value(row) as f32,
                DistCol::None   => 0.0,
            };
            results.push((text, source, page, distance));
        }
    }

    Ok(results)
}

// added

// ── filtered search: restrict results to one source document ──────────────
#[pyfunction]
fn lancedb_search_filtered(
    db_dir:        String,
    table_name:    String,
    query_vector:  Vec<f32>,
    top_k:         usize,
    source_filter: Option<String>,   // e.g. Some("2603.07379v1.pdf")
) -> PyResult<Vec<(String, String, i32, f32)>> {
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

            let mut q = table
                .query()
                .limit(top_k)
                .nearest_to(query_vector)
                .map_err(|e| format!("nearest_to failed: {:?}", e))?;

            // Apply source filter when provided
            if let Some(ref src) = source_filter {
                q = q.only_if(format!("source = '{}'", src));
            }

            let batches = q
                .execute()
                .await
                .map_err(|e| format!("Query execute failed: {:?}", e))?
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| format!("Collect failed: {:?}", e))?;

            parse_search_results(batches)
        })
        .map_err(|e| PyRuntimeError::new_err(e))
}


#[pymodule]
fn rag_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_embed_model_zembed, m)?)?;
    m.add_function(wrap_pyfunction!(load_embed_model_local, m)?)?;
    m.add_function(wrap_pyfunction!(embed_texts_rust_local, m)?)?;
    m.add_function(wrap_pyfunction!(embed_texts_rust_zembed, m)?)?;
    m.add_function(wrap_pyfunction!(embed_query_rust_zembed, m)?)?;
    // m.add_function(wrap_pyfunction!(sliding_window_chunker, m)?)?;
    m.add_function(wrap_pyfunction!(semantic_window_chunker_advanced, m)?)?;
    m.add_function(wrap_pyfunction!(load_pdf_pages_many, m)?)?;
    m.add_function(wrap_pyfunction!(load_pdf_pages_markdown, m)?)?;
    m.add_function(wrap_pyfunction!(lancedb_create_or_open, m)?)?;
    m.add_function(wrap_pyfunction!(lancedb_search, m)?)?;
    m.add_function(wrap_pyfunction!(lancedb_search_filtered, m)?)?;
    m.add_function(wrap_pyfunction!(load_pdf_pages_pdfium_many, m)?)?;
    m.add_function(wrap_pyfunction!(extract_visual_elements, m)?)?;
    Ok(())
}
