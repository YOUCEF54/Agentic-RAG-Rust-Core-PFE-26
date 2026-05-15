use crate::cleaning::is_low_signal_sentence;
use crate::embeddings::get_chunking_embedder;

const MIN_CHUNK_CHARS: usize = 80;

fn split_sentences(text: &str) -> Vec<String> {
    let abbrevs = [
        "dr", "al.", "mr", "mrs", "prof", "fig", "et al", "e.g", "i.e", "vs", "no", "vol",
    ];
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();

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
                    let is_abbrev =
                        abbrevs.iter().any(|&a| word_before == a) || word_before.len() == 1;

                    if !is_abbrev {
                        let s = paragraph[start..=i].trim().to_string();
                        if !s.is_empty() {
                            sentences.push(s);
                        }
                        start = after;
                    }
                }
            }
            i += 1;
        }
        let tail = paragraph[start..].trim().to_string();
        if !tail.is_empty() {
            sentences.push(tail);
        }
    }

    sentences
}

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

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product: f32 = v1.iter().zip(v2).map(|(a, b)| a * b).sum();
    let norm_v1: f32 = v1.iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
    let norm_v2: f32 = v2.iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
    dot_product / (norm_v1 * norm_v2)
}

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

pub fn semantic_window_chunker_advanced_impl(
    text: String,
    max_chars: usize,
    window_size: usize,
) -> Result<Vec<String>, String> {
    let stripped = strip_academic_header(&text);
    let max_chars = max_chars.max(MIN_CHUNK_CHARS);
    let window_size = window_size.max(2);
    let sentences: Vec<String> = split_sentences(stripped)
        .into_iter()
        .filter(|s| !is_low_signal_sentence(s))
        .collect();

    if sentences.is_empty() {
        return Ok(Vec::new());
    }

    if sentences.len() < window_size {
        let joined = sentences.join(" ");
        if joined.len() > max_chars {
            return Ok(simple_sliding_window(&joined, max_chars)
                .into_iter()
                .filter(|c| c.trim().len() >= MIN_CHUNK_CHARS)
                .collect());
        }
        return Ok(if joined.trim().len() >= MIN_CHUNK_CHARS {
            vec![joined]
        } else {
            Vec::new()
        });
    }

    let model_mutex = get_chunking_embedder()?;
    let mut model_guard = model_mutex.lock().map_err(|e| format!("Mutex lock failed: {e}"))?;

    let windows: Vec<String> = sentences.windows(window_size).map(|w| w.join(" ")).collect();
    let window_embeddings = model_guard
        .embed(windows, None)
        .map_err(|e| format!("Embedding failed: {e}"))?;

    let mut distances = Vec::new();
    for i in 0..window_embeddings.len() - 1 {
        let dist = 1.0 - cosine_similarity(&window_embeddings[i], &window_embeddings[i + 1]);
        distances.push(dist);
    }

    let mut break_points = Vec::new();
    let min_gap = 1;
    let mut last_bp: Option<usize> = None;
    let neighborhood = 3;

    for i in 0..distances.len() {
        let dist = distances[i];
        let start_idx = i.saturating_sub(neighborhood);
        let end_idx = (i + neighborhood + 1).min(distances.len());
        let local_slice = &distances[start_idx..end_idx];

        let local_mean: f32 = local_slice.iter().sum::<f32>() / local_slice.len() as f32;
        let variance: f32 =
            local_slice.iter().map(|&x| (x - local_mean).powi(2)).sum::<f32>() / local_slice.len() as f32;
        let local_std_dev = variance.sqrt();
        let local_threshold = local_mean + (1.2 * local_std_dev); // replaced 1.5 by 1.2 for "sharper" chunks

        if dist > local_threshold {
            let candidate = i + 1;
            let can_push = match last_bp {
                Some(prev) => candidate >= prev + min_gap,
                None => true,
            };
            if can_push && candidate < sentences.len() {
                break_points.push(candidate);
                last_bp = Some(candidate);
            }
        }
    }

    let mut chunks = Vec::new();
    let mut current_start = 0;

    for bp in break_points {
        if bp >= sentences.len() {
            continue;
        }
        let chunk_text = sentences[current_start..=bp].join(" ");
        if chunk_text.len() > max_chars {
            chunks.extend(simple_sliding_window(&chunk_text, max_chars));
        } else if chunk_text.trim().len() >= MIN_CHUNK_CHARS {
            chunks.push(chunk_text);
        }
        current_start = bp + 1;
    }

    if current_start < sentences.len() {
        let tail = sentences[current_start..].join(" ");
        if tail.len() > max_chars {
            chunks.extend(simple_sliding_window(&tail, max_chars));
        } else if tail.trim().len() >= MIN_CHUNK_CHARS {
            chunks.push(tail);
        }
    }

    Ok(chunks)
}

