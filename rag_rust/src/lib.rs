use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

mod chunking;
mod cleaning;
mod dartboard;
mod embeddings;
mod pdf_ops;
mod runtime;
mod vector_store;

#[pyfunction]
fn load_embed_model_local() -> PyResult<()> {
    embeddings::load_embed_model_local_impl().map_err(PyRuntimeError::new_err)
}

#[pyfunction]
fn load_embed_model_zembed() -> PyResult<()> {
    embeddings::load_embed_model_zembed_impl().map_err(PyRuntimeError::new_err)
}

#[pyfunction]
fn embed_texts_rust_zembed(py: Python<'_>, texts: Vec<String>, embed_batch_size: usize) -> PyResult<Vec<Vec<f32>>> {
    let embeddings = py.detach(|| {
        embeddings::embed_texts_rust_zembed_impl(texts, embed_batch_size)
    });
    
    // 1. Map the internal String error into a PyRuntimeError
    // 2. Use '?' to extract the inner Vec<Vec<f32>> or return early if it's an Error
    let successfully_embedded = embeddings.map_err(PyRuntimeError::new_err)?;
    
    Ok(successfully_embedded)
}

#[pyfunction]
fn embed_texts_rust_local(py: Python<'_>, texts: Vec<String>, embed_batch_size: usize) -> PyResult<Vec<Vec<f32>>> {
    let embeddings = py.detach(|| {
         embeddings::embed_texts_rust_local_impl(texts, embed_batch_size)
    });
    
    let successfully_embedded = embeddings.map_err(PyRuntimeError::new_err)?;
    
    Ok(successfully_embedded)
}

#[pyfunction]
fn embed_query_rust_zembed(py: Python<'_>, query: String) -> PyResult<Vec<f32>> {
    let embeddings = py.detach(|| {
        embeddings::embed_query_rust_zembed_impl(query)
    });
    
    let successfully_embedded = embeddings.map_err(PyRuntimeError::new_err)?;
    
    Ok(successfully_embedded)
}

#[pyfunction]
fn semantic_window_chunker_advanced(text: String, max_chars: usize, window_size: usize) -> PyResult<Vec<String>> {
    chunking::semantic_window_chunker_advanced_impl(text, max_chars, window_size).map_err(PyRuntimeError::new_err)
}

#[pyfunction]
fn load_pdf_pages_many(paths: Vec<String>) -> PyResult<Vec<String>> {
    pdf_ops::load_pdf_pages_many_impl(paths).map_err(PyIOError::new_err)
}

#[pyfunction]
fn load_pdf_pages_markdown(paths: Vec<String>) -> PyResult<Vec<String>> {
    pdf_ops::load_pdf_pages_markdown_impl(paths).map_err(PyIOError::new_err)
}

#[pyfunction]
fn load_pdf_pages_pdfium_many(paths: Vec<String>) -> PyResult<Vec<String>> {
    pdf_ops::load_pdf_pages_pdfium_many_impl(paths).map_err(PyRuntimeError::new_err)
}

#[pyfunction]
fn extract_visual_elements(path: String, max_pages: Option<usize>) -> PyResult<Vec<(String, usize, String)>> {
    pdf_ops::extract_visual_elements_impl(path, max_pages).map_err(PyIOError::new_err)
}

#[pyfunction]
fn lancedb_create_or_open(
    db_dir: String,
    table_name: String,
    texts: Vec<String>,
    sources: Vec<String>,
    pages: Vec<i32>,
    vectors: Vec<Vec<f32>>,
    overwrite: bool,
) -> PyResult<()> {
    vector_store::lancedb_create_or_open_impl(db_dir, table_name, texts, sources, pages, vectors, overwrite)
        .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
fn lancedb_search(
    db_dir: String,
    table_name: String,
    query_vector: Vec<f32>,
    top_k: usize,
) -> PyResult<Vec<(String, String, i32, f32)>> {
    vector_store::lancedb_search_impl(db_dir, table_name, query_vector, top_k).map_err(PyRuntimeError::new_err)
}

#[pyfunction]
fn lancedb_search_filtered(
    db_dir: String,
    table_name: String,
    query_vector: Vec<f32>,
    top_k: usize,
    source_filter: Option<String>,
) -> PyResult<Vec<(String, String, i32, f32)>> {
    vector_store::lancedb_search_filtered_impl(db_dir, table_name, query_vector, top_k, source_filter)
        .map_err(PyRuntimeError::new_err)
}

#[pymodule]
fn rag_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_embed_model_zembed, m)?)?;
    m.add_function(wrap_pyfunction!(load_embed_model_local, m)?)?;
    m.add_function(wrap_pyfunction!(embed_texts_rust_local, m)?)?;
    m.add_function(wrap_pyfunction!(embed_texts_rust_zembed, m)?)?;
    m.add_function(wrap_pyfunction!(embed_query_rust_zembed, m)?)?;
    m.add_function(wrap_pyfunction!(semantic_window_chunker_advanced, m)?)?;
    m.add_function(wrap_pyfunction!(load_pdf_pages_many, m)?)?;
    m.add_function(wrap_pyfunction!(load_pdf_pages_markdown, m)?)?;
    m.add_function(wrap_pyfunction!(lancedb_create_or_open, m)?)?;
    m.add_function(wrap_pyfunction!(lancedb_search, m)?)?;
    m.add_function(wrap_pyfunction!(lancedb_search_filtered, m)?)?;
    m.add_function(wrap_pyfunction!(load_pdf_pages_pdfium_many, m)?)?;
    m.add_function(wrap_pyfunction!(extract_visual_elements, m)?)?;
    m.add_function(wrap_pyfunction!(dartboard::dartboard_rerank, m)?)?;
    Ok(())
}