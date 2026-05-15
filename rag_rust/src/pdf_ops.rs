use std::collections::HashSet;

use pdf_oxide::pipeline::converters::OutputConverter;
use pdf_oxide::pipeline::reading_order::ReadingOrderContext;
use pdf_oxide::pipeline::{TextPipeline, TextPipelineConfig};
use pdf_oxide::{PdfDocument, ReadingOrder};
use pdfium_render::prelude::*;
use rayon::prelude::*;

use crate::cleaning::{clean_research_text, is_reference_page};

fn extract_pages_from_path(path: &str) -> Result<Vec<String>, String> {
    let mut doc = PdfDocument::open(path).map_err(|e| format!("Could not open PDF: {e:?}"))?;

    let config = TextPipelineConfig::default();
    let pipeline = TextPipeline::with_config(config.clone());
    let converter = pdf_oxide::pipeline::converters::MarkdownOutputConverter::new();

    let num_pages = doc
        .page_count()
        .map_err(|e| format!("Could not get page count: {e:?}"))?;

    let mut pages_text = Vec::new();
    for i in 0..num_pages {
        if let Ok(spans) = doc.extract_spans_with_reading_order(i, ReadingOrder::ColumnAware) {
            let context = ReadingOrderContext::default().with_page(i as u32);
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

pub fn load_pdf_pages_many_impl(paths: Vec<String>) -> Result<Vec<String>, String> {
    let mut results: Vec<(usize, Vec<String>)> = paths
        .par_iter()
        .enumerate()
        .map(|(idx, path)| extract_pages_from_path(path).map(|pages| (idx, pages)))
        .collect::<Result<Vec<_>, String>>()?;

    results.sort_by_key(|(idx, _)| *idx);
    let mut all_pages = Vec::new();
    for (_, pages) in results {
        all_pages.extend(pages);
    }
    Ok(all_pages)
}

pub fn load_pdf_pages_pdfium_many_impl(paths: Vec<String>) -> Result<Vec<String>, String> {
    let results: Vec<(usize, Vec<String>)> = paths
        .par_iter()
        .enumerate()
        .map(|(idx, path)| {
            let bind = Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./"))
                .or_else(|_| Pdfium::bind_to_system_library())
                .map_err(|e| format!("Failed to bind to PDFium: {e:?}"))?;

            let pdfium = Pdfium::new(bind);
            let doc = pdfium
                .load_pdf_from_file(path, None)
                .map_err(|e| format!("Could not open PDF {path}: {e:?}"))?;

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
        .collect::<Result<Vec<_>, String>>()?;

    let mut sorted_results = results;
    sorted_results.sort_by_key(|(idx, _)| *idx);
    Ok(sorted_results.into_iter().flat_map(|(_, p)| p).collect())
}

fn extract_markdown_from_path(path: &str) -> Result<Vec<String>, String> {
    let mut doc = PdfDocument::open(path).map_err(|e| format!("Could not open PDF: {e:?}"))?;
    let num_pages = doc
        .page_count()
        .map_err(|e| format!("Could not get page count: {e:?}"))?;

    let mut pages_md = Vec::new();
    for i in 0..num_pages {
        match doc.to_markdown(i, &Default::default()) {
            Ok(md) => {
                let trimmed = md.trim().to_string();
                if !trimmed.is_empty() {
                    pages_md.push(trimmed);
                }
            }
            Err(_) => continue,
        }
    }
    Ok(pages_md)
}

pub fn load_pdf_pages_markdown_impl(paths: Vec<String>) -> Result<Vec<String>, String> {
    let mut results: Vec<(usize, Vec<String>)> = paths
        .par_iter()
        .enumerate()
        .map(|(idx, path)| extract_markdown_from_path(path).map(|pages| (idx, pages)))
        .collect::<Result<Vec<_>, String>>()?;

    results.sort_by_key(|(idx, _)| *idx);
    let mut all_pages = Vec::new();
    for (_, pages) in results {
        all_pages.extend(pages);
    }
    Ok(all_pages)
}

fn extract_page_texts_with_numbers(path: &str) -> Result<Vec<(usize, String)>, String> {
    let mut doc = PdfDocument::open(path).map_err(|e| format!("Could not open PDF: {e:?}"))?;
    let config = TextPipelineConfig::default();
    let pipeline = TextPipeline::with_config(config);
    let num_pages = doc
        .page_count()
        .map_err(|e| format!("Could not get page count: {e:?}"))?;

    let mut pages_text = Vec::new();
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

pub fn extract_visual_elements_impl(
    path: String,
    max_pages: Option<usize>,
) -> Result<Vec<(String, usize, String)>, String> {
    let pages = extract_page_texts_with_numbers(&path)?;
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

