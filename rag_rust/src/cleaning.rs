pub fn is_reference_page(text: &str) -> bool {
    let non_empty_lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if non_empty_lines.is_empty() {
        return false;
    }

    let ref_lines = non_empty_lines
        .iter()
        .filter(|l| looks_like_reference_line(l))
        .count();

    let has_ref_header = non_empty_lines
        .iter()
        .take(3)
        .any(|l| l.to_uppercase().starts_with("REFERENCES"));

    has_ref_header || (ref_lines * 100 / non_empty_lines.len() > 35)
}

fn is_reference_paragraph(para: &str) -> bool {
    let lines: Vec<&str> = para.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if lines.is_empty() {
        return false;
    }

    if lines.len() == 1 {
        let u = lines[0].to_uppercase();
        return u == "REFERENCES" || u == "BIBLIOGRAPHY";
    }

    let ref_lines = lines.iter().filter(|l| looks_like_reference_line(l)).count();
    ref_lines * 100 / lines.len() > 60
}

fn is_probable_pdf_result_line(line: &str) -> bool {
    let l = line.trim();
    if l.is_empty() {
        return false;
    }
    let lower = l.to_lowercase();

    if lower.ends_with(".pdf") && lower.contains(".pdf") && l.len() < 120 {
        return true;
    }

    if let Some(rest) = lower.strip_prefix("p.") {
        let cleaned: String = rest.chars().filter(|c| !c.is_whitespace()).collect();
        if !cleaned.is_empty() && cleaned.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
    }

    if l.len() <= 6 && l.starts_with("0.") && l.chars().skip(2).all(|c| c.is_ascii_digit()) {
        return true;
    }

    false
}

pub fn looks_like_reference_line(line: &str) -> bool {
    let l = line.trim();
    if l.is_empty() {
        return false;
    }

    if l.starts_with('[') && l.chars().nth(1).map(|c| c.is_ascii_digit()).unwrap_or(false) {
        return true;
    }

    let lower = l.to_lowercase();
    let year_hits = ["201", "202", "199"].iter().filter(|y| lower.contains(**y)).count();

    let citation_signals = [
        "arxiv:",
        "doi:",
        "proceedings",
        "transactions",
        "journal",
        "acm",
        "ieee",
        "springer",
        "et al.",
        "preprint",
        "conference",
    ];
    let signal_hits = citation_signals.iter().filter(|s| lower.contains(**s)).count();
    let semicolon_count = l.matches(';').count();

    (signal_hits >= 2 && year_hits >= 1)
        || (lower.contains("arxiv:") && year_hits >= 1)
        || (semicolon_count >= 2 && year_hits >= 1)
}

pub fn is_low_signal_line(line: &str) -> bool {
    let l = line.trim();
    if l.is_empty() {
        return true;
    }
    if is_probable_pdf_result_line(l) {
        return true;
    }

    let lower = l.to_lowercase();
    if lower.starts_with("arxiv:") {
        return true;
    }
    if lower == "references" || lower == "bibliography" {
        return true;
    }
    if looks_like_reference_line(l) && l.len() < 320 {
        return true;
    }

    false
}

pub fn is_low_signal_sentence(sentence: &str) -> bool {
    let s = sentence.trim();
    if s.is_empty() {
        return true;
    }
    if is_low_signal_line(s) {
        return true;
    }
    if s.len() < 24 {
        return true;
    }
    let lower = s.to_lowercase();
    if let Some(pdf_idx) = lower.find(".pdf") {
        if pdf_idx < 96 {
            return true;
        }
    }
    if lower.contains("arxiv:") && (s.matches(';').count() >= 1 || lower.contains("et al.")) {
        return true;
    }

    let alpha = s.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let digits = s.chars().filter(|c| c.is_ascii_digit()).count();
    let punct = s.chars().filter(|c| !c.is_alphanumeric() && !c.is_whitespace()).count();
    let len = s.chars().count().max(1);
    let digit_ratio = digits as f32 / len as f32;
    let punct_ratio = punct as f32 / len as f32;
    let alpha_ratio = alpha as f32 / len as f32;

    if s.len() > 120 {
        return false;
    }
    (alpha_ratio < 0.35 && (digit_ratio > 0.20 || punct_ratio > 0.30))
        || (looks_like_reference_line(s) && s.len() < 420)
}

pub fn clean_research_text(text: &str) -> String {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let cleaned_paragraphs: Vec<String> = paragraphs
        .iter()
        .filter(|para| !is_reference_paragraph(para.trim()))
        .map(|para| {
            let no_ctrl: String = para
                .chars()
                .filter(|c| !c.is_control() || c.is_whitespace())
                .collect();

            let lines: Vec<&str> = no_ctrl.lines().map(|l| l.trim()).collect();
            let mut buf = String::new();
            for (i, &line) in lines.iter().enumerate() {
                if is_low_signal_line(line) {
                    continue;
                }
                if line.ends_with('-') {
                    buf.push_str(&line[..line.len() - 1]);
                } else {
                    buf.push_str(line);
                    if i < lines.len() - 1 {
                        let last = line.chars().last().unwrap_or(' ');
                        if ".!?\":".contains(last) {
                            buf.push('\n');
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

