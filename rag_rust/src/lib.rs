use pyo3::prelude::*;
// use std::fs; // Pour corriger l'erreur E0433
use pdf_oxide::PdfDocument;
use pdf_oxide::pipeline::{TextPipeline, TextPipelineConfig};
use pdf_oxide::pipeline::reading_order::ReadingOrderContext;

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
    let mut current_pos = 0;
    let text_chars: Vec<char> = text.chars().collect();

    while current_pos < text_chars.len() {
        let mut end_pos = std::cmp::min(current_pos + max_chars, text_chars.len());

        // Si on n'est pas à la fin, on cherche un point ou un retour à la ligne pour couper proprement
        if end_pos < text_chars.len() {
            let lookback_range = if end_pos > current_pos + (max_chars / 2) { 
                max_chars / 2 
            } else { 
                end_pos - current_pos 
            };

            // On cherche un délimiteur en arrière pour ne pas couper une phrase
            if let Some(pos) = text_chars[end_pos - lookback_range..end_pos]
                .iter()
                .rposition(|&c| c == '.' || c == '\n') 
            {
                end_pos = (end_pos - lookback_range) + pos + 1;
            }
        }

        let chunk: String = text_chars[current_pos..end_pos].iter().collect();
        chunks.push(chunk.trim().to_string());

        // On avance en soustrayant l'overlap
        current_pos = end_pos;
        if current_pos < text_chars.len() && current_pos > overlap {
            current_pos -= overlap;
        }
    }
    Ok(chunks)
}
#[pyfunction]
fn load_pdf_pages(path: String) -> PyResult<Vec<String>> {
    // 1. Ouverture du document avec la nouvelle API
    let mut doc = PdfDocument::open(&path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Impossible d'ouvrir le PDF: {:?}", e))
    })?;

    let mut pages_text = Vec::new();
    let config = TextPipelineConfig::default();
    let pipeline = TextPipeline::with_config(config.clone());

    // 2. Boucle sur les pages (l'index commence à 0)
    let num_pages = doc.page_count().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Impossible de lire le nombre de pages: {:?}", e))
    })?;
    for i in 0..num_pages {
        // Extraction des "spans" (blocs de texte)
        if let Ok(spans) = doc.extract_spans(i) {
            // Le pipeline traite les blocs pour respecter l'ordre de lecture (colonnes)
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

#[pymodule]
fn rag_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(top_k_cosine, m)?)?;
    m.add_function(wrap_pyfunction!(smart_chunker, m)?)?;
    m.add_function(wrap_pyfunction!(load_pdf_pages, m)?)?;
    Ok(())
}
