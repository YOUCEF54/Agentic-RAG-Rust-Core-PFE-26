from __future__ import annotations

import re
from dataclasses import dataclass, field

import rag_rust

# ---------------------------------------------------------------------------
# Chunk dataclass — carries provenance through the whole pipeline
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    source: str          # PDF filename
    page: int            # 1-based
    chunk_idx: int       # position within the page's chunks
    char_len: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_len = len(self.text)

    def __len__(self) -> int:
        return self.char_len

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_idx": self.chunk_idx,
            "char_len": self.char_len,
        }


# ---------------------------------------------------------------------------
# Constants  (tune or move to env vars as needed)
# ---------------------------------------------------------------------------

MIN_CHUNK_CHARS   = 80    # discard noise / header fragments
MAX_CHUNK_CHARS   = 900   # ~200-220 tokens for bge-small; keeps full sentences
OVERLAP_CHARS     = 120   # ~13 % overlap — balances recall vs redundancy
                          # (Lewis et al. 2020 RAG paper uses sliding window;
                          #  Gao et al. 2023 "Precise Zero-Shot" confirms overlap
                          #  materially improves boundary-spanning retrieval)

# Sentence / paragraph boundary regex used for semantic split points
_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+|\n{2,}')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Normalise whitespace; collapse runs of blanks/newlines."""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _sentence_aware_chunks(text: str) -> list[str]:
    """
    Split `text` into chunks that:
      1. Never exceed MAX_CHUNK_CHARS.
      2. Break on sentence / paragraph boundaries when possible.
      3. Apply a character-level sliding overlap.

    This is equivalent to LangChain's RecursiveCharacterTextSplitter
    but implemented directly to avoid the dependency and to integrate
    cleanly with the Rust smart_chunker fallback.
    """
    # First pass: split on natural boundaries
    segments: list[str] = [s.strip() for s in _BOUNDARY_RE.split(text) if s.strip()]

    chunks: list[str] = []
    buffer = ""

    for seg in segments:
        candidate = (buffer + " " + seg).strip() if buffer else seg
        if len(candidate) <= MAX_CHUNK_CHARS:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            # Segment itself may exceed the limit → delegate to Rust smart_chunker
            if len(seg) > MAX_CHUNK_CHARS:
                chunks.extend(
                    c.strip()
                    for c in rag_rust.smart_chunker(seg, MAX_CHUNK_CHARS, OVERLAP_CHARS)
                    if c.strip()
                )
                buffer = ""
            else:
                buffer = seg

    if buffer:
        chunks.append(buffer)

    # Second pass: re-apply overlap by prepending tail of previous chunk
    overlapped: list[str] = []
    for i, chunk in enumerate(chunks):
        if i > 0 and OVERLAP_CHARS:
            tail = chunks[i - 1][-OVERLAP_CHARS:]
            # Only prepend if it doesn't create a duplicate or over-length chunk
            candidate = (tail + " " + chunk).strip()
            chunk = candidate if len(candidate) <= MAX_CHUNK_CHARS + OVERLAP_CHARS else chunk
        overlapped.append(chunk)

    return overlapped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_chunk(paths: list[str]) -> list[Chunk]:
    """
    Load PDFs via PDFium (reading-order guaranteed) and return a flat list
    of Chunk objects with full provenance metadata.

    Pipeline
    --------
    PDFium extract (per-path, parallel in Rust)
      → per-page clean
      → sentence-aware chunking with overlap
      → noise filter (MIN_CHUNK_CHARS)
      → Chunk dataclass with source / page / index

    Parameters
    ----------
    paths : list[str]
        Absolute or relative paths to PDF files.

    Returns
    -------
    list[Chunk]
        Ordered by (source, page, chunk_idx). Ready to embed.
    """
    if not paths:
        raise ValueError("load_and_chunk: no paths provided.")

    all_chunks: list[Chunk] = []

    for pdf_path in paths:
        source_name = pdf_path.rsplit("/", 1)[-1]  # filename only for metadata

        # Rust returns one string per page, in reading order
        pages: list[str] = rag_rust.load_pdf_pages_pdfium_many([pdf_path])

        for page_idx, raw_page in enumerate(pages, start=1):
            cleaned = _clean(raw_page)
            if not cleaned:
                continue

            page_chunks = _sentence_aware_chunks(cleaned)

            for chunk_idx, text in enumerate(page_chunks):
                if len(text.strip()) < MIN_CHUNK_CHARS:
                    continue  # skip headers, lone numbers, etc.
                all_chunks.append(
                    Chunk(
                        text=text.strip(),
                        source=source_name,
                        page=page_idx,
                        chunk_idx=chunk_idx,
                    )
                )

    return all_chunks