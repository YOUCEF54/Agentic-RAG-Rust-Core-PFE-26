use pyo3::prelude::*;
use pdf_oxide::PdfDocument;
use pdf_oxide::pipeline::{TextPipeline, TextPipelineConfig};
use pdf_oxide::pipeline::reading_order::ReadingOrderContext;
use rayon::prelude::*;
use std::sync::Arc;
use std::cell::RefCell;
use once_cell::sync::OnceCell;
use std::sync::Mutex;
use arrow_array::types::Float32Type;
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

thread_local! {
    static LOCAL_MODEL: RefCell<Option<TextEmbedding>> = RefCell::new(None);
}

// fn get_local_model() -> &'static std::thread::LocalKey<RefCell<Option<TextEmbedding>>> {
//     &LOCAL_MODEL
// }

// ── Global Tokio runtime — created once, reused across all calls ───────────────
static RUNTIME: OnceCell<Runtime> = OnceCell::new();

fn get_runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to build Tokio runtime")
    })
}

// ── Cached DB connection — reconnects only when db_dir changes ─────────────────
static DB_CACHE: OnceCell<std::sync::Mutex<(String, lancedb::Connection)>> = OnceCell::new();

async fn get_or_connect(db_dir: &str) -> Result<lancedb::Connection, String> {
    if let Some(cache) = DB_CACHE.get() {
        if let Ok(guard) = cache.lock() {
            if guard.0 == db_dir {
                return Ok(guard.1.clone());
            }
        }
    }

    let conn = connect(db_dir)
        .execute()
        .await
        .map_err(|e| format!("DB connect failed: {:?}", e))?;

    let _ = DB_CACHE.get_or_init(|| {
        std::sync::Mutex::new((db_dir.to_string(), conn.clone()))
    });

    if let Some(cache) = DB_CACHE.get() {
        if let Ok(mut guard) = cache.lock() {
            if guard.0 != db_dir {
                *guard = (db_dir.to_string(), conn.clone());
            }
        }
    }

    Ok(conn)
}

// ── Fix: Cached Table handle — open_table is called only once per table name ───
//
// lancedb_search was calling open_table on every invocation — that's a
// filesystem stat + metadata read on every search across 100 profiling runs.
// Caching the Table handle eliminates that overhead entirely.
//
// Key: "<db_dir>/<table_name>" — invalidated if either changes.
static TABLE_CACHE: OnceCell<std::sync::Mutex<(String, lancedb::Table)>> = OnceCell::new();

async fn get_or_open_table(db: &lancedb::Connection, db_dir: &str, table_name: &str) -> Result<lancedb::Table, String> {
    let cache_key = format!("{}/{}", db_dir, table_name);

    if let Some(cache) = TABLE_CACHE.get() {
        if let Ok(guard) = cache.lock() {
            if guard.0 == cache_key {
                return Ok(guard.1.clone());
            }
        }
    }

    let table = db
        .open_table(table_name)
        .execute()
        .await
        .map_err(|e| format!("Open table failed: {:?}", e))?;

    let _ = TABLE_CACHE.get_or_init(|| {
        std::sync::Mutex::new((cache_key.clone(), table.clone()))
    });

    if let Some(cache) = TABLE_CACHE.get() {
        if let Ok(mut guard) = cache.lock() {
            if guard.0 != cache_key {
                *guard = (cache_key, table.clone());
            }
        }
    }

    Ok(table)
}

// ── Fix: fastembed global model — same ONNX runtime as rust_only ───────────────
//
// SentenceTransformer (Python/PyTorch) was the bottleneck for embedding.
// fastembed uses the ONNX Runtime which is significantly faster on CPU.
// The model is loaded once into a OnceCell and reused across all 100 runs,
// mirroring exactly how rust_only loads its model outside the loop.
//
// Model: BGE Small EN v1.5 — 384-dim, matches the LanceDB schema dim below.
// ── Dans lib.rs ──────────────────────────────────────────────────────────────

static EMBED_MODEL: OnceCell<Mutex<TextEmbedding>> = OnceCell::new();

fn get_embed_model() -> &'static Mutex<TextEmbedding> {
    EMBED_MODEL.get_or_init(|| {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15)
        ).expect("Failed to load fastembed model");
        Mutex::new(model)
    })
}

#[pyfunction]
fn embed_texts_rust(texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
    let model_mutex = get_embed_model();
    let mut model = model_mutex.lock().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lock failed")
    })?;

    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    // On passe TOUT le vecteur d'un coup. 
    // ONNX va paralléliser le calcul matriciel en interne.
    model
        .embed(refs, Some(256)) 
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))
}
// static EMBED_MODEL: OnceCell<Mutex<TextEmbedding>> = OnceCell::new();
// static EMBED_MODEL: OnceCell<Arc<TextEmbedding>> = OnceCell::new();
// static EMBED_MODEL: OnceCell<Arc<RwLock<TextEmbedding>>> = OnceCell::new();


/// Load the fastembed model eagerly. Call once at startup from Python
/// (mirrors embedder::load_model() in rust_only).
#[pyfunction]
fn load_embed_model() -> PyResult<()> {
    get_embed_model(); // Pré-charge le modèle en mémoire
    Ok(())
}


// ── Cosine similarity ──────────────────────────────────────────────────────────
fn cosine_similarity_inner(a: &[f32], b: &[f32]) -> Option<f32> {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        None
    } else {
        Some(dot / denom)
    }
}

#[pyfunction]
fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Vectors must have the same length",
        ));
    }
    cosine_similarity_inner(&a, &b).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Zero-norm vector")
    })
}

#[pyfunction]
fn top_k_cosine(query: Vec<f32>, vectors: Vec<Vec<f32>>, k: usize) -> PyResult<Vec<(usize, f32)>> {
    if k == 0 {
        return Ok(Vec::new());
    }

    let mut scores = Vec::with_capacity(vectors.len());
    for (idx, v) in vectors.iter().enumerate() {
        if v.len() != query.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All vectors must have the same length as the query",
            ));
        }
        if let Some(sim) = cosine_similarity_inner(&query, v) {
            scores.push((idx, sim));
        }
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if scores.len() > k {
        scores.truncate(k);
    }
    Ok(scores)
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
            current_pos -= overlap;
        }
    }
    Ok(chunks)
}

#[pyfunction]
fn load_pdf_pages(path: String) -> PyResult<Vec<String>> {
    extract_pages_from_path(&path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(e)
    })
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
            let db = get_or_connect(&db_dir).await?;

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

            // Invalidate the table cache on every insert so the next search
            // picks up the freshly written table (overwrite changes the snapshot).
            if let Some(cache) = TABLE_CACHE.get() {
                if let Ok(mut guard) = cache.lock() {
                    guard.0 = String::new(); // poison the key
                }
            }

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
            let db = get_or_connect(&db_dir).await?;

            // Use cached table handle — no open_table on repeat calls
            let table = get_or_open_table(&db, &db_dir, &table_name).await?;

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

#[pyfunction]
fn load_pdf_pages_many(paths: Vec<String>) -> PyResult<Vec<String>> {
    let mut results: Vec<(usize, Vec<String>)> = paths
        .par_iter()
        .enumerate()
        .map(|(idx, path)| {
            extract_pages_from_path(path)
                .map(|pages| (idx, pages))
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

fn extract_pages_from_path(path: &str) -> Result<Vec<String>, String> {
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
                    pages_text.push(trimmed);
                }
            }
        }
    }

    Ok(pages_text)
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
    let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vectors.iter().map(|v| {
            Some(v.iter().map(|x| Some(*x)).collect::<Vec<_>>())
        }),
        dim as i32,
    );

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
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(top_k_cosine, m)?)?;
    m.add_function(wrap_pyfunction!(smart_chunker, m)?)?;
    m.add_function(wrap_pyfunction!(load_pdf_pages, m)?)?;
    m.add_function(wrap_pyfunction!(load_pdf_pages_many, m)?)?;
    m.add_function(wrap_pyfunction!(lancedb_create_or_open, m)?)?;
    m.add_function(wrap_pyfunction!(lancedb_search, m)?)?;
    Ok(())
}

