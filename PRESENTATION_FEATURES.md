# Agentic-RAG-Rust-Core — Features & Presentation Guide

> **Project:** Agentic-RAG-Rust-Core-PFE-26  
> **Stack:** Python (orchestration) · Rust (PyO3 extension) · LanceDB · FastAPI · Ollama / OpenRouter  
> **Goal:** A high-performance, locally-runnable Retrieval-Augmented Generation (RAG) system with an agentic multi-step pipeline and a Rust-accelerated core.

---

## 1. System Architecture Overview

The system is split into two tightly coupled layers:

| Layer | Language | Role |
|---|---|---|
| **Orchestration** | Python | Agent pipeline, API server, caching, config |
| **Core Engine** | Rust (PyO3) | PDF loading, chunking, embedding, vector DB |

```
PDF Files
   │
   ▼
[Rust Core: PDF Loader]
   │  PDFium / pdf_oxide (parallel, per-thread)
   ▼
[Rust Core: Sliding Window Chunker]
   │  Sentence-aware, overlap-based splitting
   ▼
[Rust Core: ONNX Embedder (fastembed BGE)]
   │  GIL-released, batched
   ▼
[Rust Core: LanceDB Vector Store]
   │  Cosine similarity, filtered search
   ▼
[Python: Agentic Pipeline]
   QueryRefiner → Retriever → DynamicPassageSelector → Generator → Evaluator
   │
   ▼
[FastAPI REST API]  ←→  Frontend / curl
```

---

## 2. Features & Functionalities

### 2.1 PDF Ingestion & Text Extraction

**File:** `rag_rust/src/lib.rs`

- **Multi-file parallel loading** using Rayon (`par_iter`) — all PDFs processed simultaneously on separate threads.
- **Three extraction backends:**
  - `load_pdf_pages_pdfium_many` — PDFium C library (highest quality, complex layouts).
  - `load_pdf_pages_many` — pdf_oxide with ColumnAware reading order (multi-column papers).
  - `load_pdf_pages_markdown` — converts pages to structured Markdown (used by API).
- **Reference page filtering** — detects and skips bibliography/reference pages to avoid polluting the index.
- **Academic header stripping** — removes author/affiliation headers detected by `@` symbol.
- **Text cleaning** — removes control characters, fixes hyphenated line-breaks, normalizes whitespace.
- **Per-thread PDFium binding** — C-FFI safe: each Rayon thread owns its own binding.

---

### 2.2 Text Chunking

**File:** `rag_rust/src/lib.rs` — `sliding_window_chunker`, `smart_chunker`

#### Sliding Window Chunker (production)
- **Sentence-aware** — text split into sentences first (respects abbreviations: `Dr.`, `Fig.`, `et al.`).
- **Overlap in sentence units** — avoids the "over-drop" bug of character-based overlap.
- **Greedy window filling** — accumulates sentences up to `max_chars = 800`.
- **Minimum size guard** — chunks < 80 chars discarded.
- **Force-progress guarantee** — single overlong sentences are still emitted.

#### Smart Chunker (alternative)
- Character-based, looks back `max_chars/2` for a natural sentence/newline boundary.

---

### 2.3 Embedding Engine (Rust ONNX)

**File:** `rag_rust/src/lib.rs` — `load_embed_model`, `embed_texts_rust`

- **Model:** `BAAI/bge-small-en-v1.5` via fastembed-rs (Rust-native ONNX).
- **GIL release** — `py.allow_threads(...)` so Python is never blocked during embedding.
- **Singleton model** — loaded once into `OnceCell<Mutex<TextEmbedding>>`, reused forever.
- **BGE prefix strategy:**
  - Passages: `"passage: <text>"`
  - Queries: `"Represent this sentence for searching relevant passages: <query>"`
- **Batched processing** — `EMBED_BATCH_SIZE` auto-tuned by hardware calibration.
- **Batch size clamped** to `[2, 256]` to prevent hardware panics.

---

### 2.4 Vector Database (LanceDB — Rust)

**File:** `rag_rust/src/lib.rs` — `lancedb_create_or_open`, `lancedb_search`, `lancedb_search_filtered`

- **Serverless embedded DB** — no separate process needed.
- **Schema:** `(id, source, page, text, vector)` — every chunk retains its origin.
- **Cosine similarity** search via `DistanceType::Cosine`.
- **Filtered search** — restricts to a single source PDF using SQL predicate `source = 'file.pdf'`.
- **Overwrite or reuse** — full rebuild or open-existing table.
- **Apache Arrow** format for fast bulk insertion.
- **Tokio async runtime** (`OnceCell<Runtime>`) — blocks the calling thread for seamless sync Python integration.

---

### 2.5 PDF Hash Caching

**File:** `script_opt_ollama.py`

- SHA-256 hash of **all PDF bytes + file names + chunking params** (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBED_MODEL_NAME`).
- Cache hit → skip the entire PDF load + embed pipeline; chunks loaded from `.rag_cache.json` instantly.
- Cache miss → full pipeline runs and result is saved for next time.

---

### 2.6 Agentic Pipeline

**File:** `agents.py`

`UserProxy` orchestrates a **retry loop**: runs the full pipeline until score ≥ threshold or max attempts reached.

#### Agent 1 — QueryRefiner
- Rewrites raw user query for better retrieval using `qwen2.5:0.5b`.
- Strips preambles small models add (e.g., `"Here is the rewritten query:"`).
- Falls back to original query if rewrite is >3× longer or LLM fails.

#### Agent 2 — Retriever
- Embeds refined query, fetches **Top-15 candidates** from LanceDB (larger pool for DPS).
- Stores top-K subset as fallback if DPS is disabled.

#### Agent 3 — DynamicPassageSelector (DPS)
- Prompts `qwen2.5:3b` to select the **minimal sufficient subset** of passages from 15 candidates.
- Models inter-passage dependencies (reasons about combinations, not individual chunks).
- Selects between 1 and 8 passages. Falls back to top-3 if parsing fails.

#### Agent 4 — Generator
- Strict grounding prompt: answer using ONLY the retrieved chunks.
- Context formatted with chunk number, source file, and page.
- Uses `qwen2.5:1.5b` locally via Ollama.

#### Agent 5 — Evaluator
- **Stage 1 — Fast token-overlap precheck:** if answer token overlap with chunks < 15% → score 0.0, no LLM needed.
- **Stage 2 — LLM-as-judge:** scores faithfulness + relevance on a 0.0–1.0 scale.
- **Retry** if `score < 0.75` and `attempts < 3`.
- Conservative: unparseable LLM output defaults to score 0.0.

#### Trace System
- Every agent appends structured events to `state["trace"]`.
- Optional `emit` callback enables real-time streaming of agent steps.

---

### 2.7 FastAPI REST Server

**File:** `api.py`

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server health + active batch size |
| `POST` | `/documents` | Upload PDFs (multipart) |
| `GET` | `/documents` | List PDFs with metadata |
| `DELETE` | `/documents/{filename}` | Delete PDF, optionally rebuild index |
| `POST` | `/index` | Build/rebuild vector index |
| `GET` | `/index/status` | Index state: `idle/building/ready/stale/error` |
| `GET` | `/hardware/config` | Read hardware calibration config |
| `POST` | `/hardware/calibrate` | Run hardware benchmark + update config |
| `POST` | `/query` | RAG query (agentic / naive / retrieval-only) |
| `POST` | `/query/stream` | SSE streaming query (real-time agent events) |

- **CORS** configured for Vercel frontend + localhost.
- **Document metadata** persisted to `data/metadata.json` (size, SHA-256, upload time, pages).
- **SSE events:** `status`, `retrieved`, `agent_event`, `answer`, `final`, `error`.

---

### 2.8 Hardware Calibration

**File:** `get_hardware_config.py`

- Benchmarks local hardware to find the **optimal embedding batch size**.
- Tests batch sizes `[4, 8, 16, 24, 32]` (quick) or up to 128 (full).
- **Early stopping** when throughput drops. **Max runtime guard** of 25 seconds.
- Saves result to `hardware_config.json`; API auto-reloads on file change.
- Output: `optimal_batch_size`, `throughput_measured`, `cpu_info`, `calibration_date`.

---

### 2.9 Visual Element Extraction

**File:** `rag_rust/src/lib.rs` — `extract_visual_elements`

- Scans PDF text for keywords: `table`, `figure`, `fig.`, `graph`, `chart`, `diagram`.
- Extracts a 240-char snippet centered on each match.
- De-duplicates via `HashSet`. Returns `(kind, page, snippet)` tuples.

---

### 2.10 Benchmarking Suite

**File:** `bench_rag.py`

| Benchmark | What it measures |
|---|---|
| **Profiling loop** | Full pipeline (read→chunk→embed→insert→search), 100 runs, CSV output |
| **Retrieval benchmark** | Vector search throughput: 20 × 3 queries |
| **Embedding benchmark** | Raw embedding speed (chunks/sec) for 200 synthetic chunks |

---

### 2.11 Deployment

- **Docker** — `Dockerfile` + `.dockerignore` for containerized deployment.
- **Render** — `render.yaml` for one-click cloud deployment.
- **`.env` configuration** — all key parameters (models, paths, batch size, chunk params) are environment-variable driven.

---

## 3. Technology Stack

| Component | Technology |
|---|---|
| PDF extraction (quality) | PDFium (C library via Rust FFI) |
| PDF extraction (structured) | pdf_oxide + ColumnAware reading order |
| Parallel processing | Rayon |
| Embedding model | BAAI/bge-small-en-v1.5 (ONNX, fastembed-rs) |
| Vector database | LanceDB (embedded, Arrow-based) |
| Python-Rust bridge | PyO3 + Maturin |
| Async runtime (Rust) | Tokio (multi-thread) |
| API framework | FastAPI + Uvicorn |
| Local LLM serving | Ollama |
| Cloud LLM | OpenRouter API |
| Local models used | qwen2.5:0.5b / 1.5b / 3b, phi4-mini:3.8b |
| Containerization | Docker |

---

---

# 🎯 Presentation Slides — Suggested Content

> **12 slides + title.** Recommended tools: PowerPoint, LaTeX Beamer, or Reveal.js.

---

## Slide 0 — Title Slide

**Title:** Agentic RAG with a Rust-Accelerated Core  
**Subtitle:** A High-Performance, Locally-Runnable Question-Answering System over PDF Documents  
**Content:** Name · Supervisor · Institution · PFE 2026  
**Tagline:** *Python orchestration · Rust performance · Local LLMs · Zero cloud dependency*

---

## Slide 1 — Problem Statement

**Title:** Why Build a Custom RAG System?

- Standard RAG pipelines are slow: Python-only embedding loops are single-threaded.
- Existing tools require proprietary cloud APIs — no offline option.
- Naive RAG uses a fixed Top-K — no reasoning about which passages are actually needed.
- No quality control: a hallucinated answer looks identical to a correct one.

**Key message:** *We need a fast, local, quality-aware RAG system.*

---

## Slide 2 — System Architecture

**Title:** Architecture at a Glance

> Use a diagram showing the full pipeline:  
> `PDFs → [Rust: Load] → [Rust: Chunk] → [Rust: Embed] → [LanceDB] → [Agents] → [API] → Frontend`

- Hot path (load, chunk, embed, search) runs entirely in Rust.
- Python handles high-level logic and LLM calls only.
- Two modes: CLI (Ollama, fully local) and REST API (OpenRouter, cloud).

---

## Slide 3 — Rust Core: PDF Loading

**Title:** High-Performance PDF Ingestion

- **Parallel loading** with Rayon — all PDFs processed simultaneously.
- Two backends: PDFium (C library) and pdf_oxide (reading-order aware).
- Auto-filters: reference pages, author headers, control characters removed.
- Result: clean, page-attributed text ready in milliseconds.

**Show a real number:** *e.g., "24 pages loaded and cleaned in < 5 ms"*

---

## Slide 4 — Rust Core: Chunking & Embedding

**Title:** Sentence-Aware Chunking + ONNX Embedding

**Left — Chunking:**
- Splits text into sentences first (handles abbreviations: Dr., Fig., et al.)
- Fills windows up to 800 chars; overlap measured in sentences, not chars.
- Minimum 80-char guard; force-progress for oversized sentences.

**Right — Embedding:**
- BAAI/bge-small-en-v1.5 running ONNX locally via fastembed-rs.
- GIL released during embedding — Python is never blocked.
- Singleton model + hardware-calibrated batch size.

**Show a real number:** *e.g., "221 chunks embedded at 14 chunks/sec on CPU"*

---

## Slide 5 — LanceDB Vector Store

**Title:** Serverless Vector Storage

- Embedded — no server process required.
- Schema stores `(source filename, page number, text, vector)` per chunk.
- Provenance tracking → cited, verifiable answers.
- Cosine similarity search in milliseconds.
- Filtered search: restrict to a single PDF.
- Apache Arrow for fast bulk inserts.

---

## Slide 6 — The Agentic Pipeline

**Title:** 5-Agent Pipeline with Self-Correction

> Flow diagram:  
> `Query → QueryRefiner → Retriever (Top-15) → DPS → Generator → Evaluator → [Retry?] → Answer`

1. **QueryRefiner** — rewrites query for better retrieval.
2. **Retriever** — fetches 15 candidate passages.
3. **DynamicPassageSelector** — LLM picks the minimal useful subset.
4. **Generator** — produces a grounded answer from selected passages only.
5. **Evaluator** — scores faithfulness; retries if score < 0.75.

---

## Slide 7 — Dynamic Passage Selection

**Title:** Smarter Retrieval via DPS

- Classic RAG: fixed Top-K, no reasoning about which chunks matter.
- **DPS:** an LLM reasons over all 15 candidates and picks the minimal sufficient set.
- Models inter-passage dependencies (combination reasoning).
- Selects 1 to 8 passages. Falls back to top-3 if parsing fails.

**Example:** *"DPS selected 2 of 15 candidates: [1, 3]"* → cleaner context → better answer.

---

## Slide 8 — Evaluator & Hallucination Prevention

**Title:** Built-In Answer Quality Control

**Stage 1 — Fast token-overlap check (no LLM cost):**
- Answer tokens vs. chunk tokens (stop-words removed).
- Overlap < 15% → score = 0.0 (hallucination flagged instantly).

**Stage 2 — LLM-as-Judge:**
- Checks faithfulness (every claim supported?) + relevance.
- Responds with `SCORE: <0.0–1.0>`.

**Retry:** score < 0.75 + attempts < 3 → re-run full pipeline.  
**Conservative fallback:** unparseable output → score = 0.0.

---

## Slide 9 — REST API & Deployment

**Title:** Production-Ready FastAPI Server

| Endpoint | Purpose |
|---|---|
| `POST /documents` | Upload PDFs |
| `POST /index` | Build vector index |
| `POST /query` | Agentic Q&A |
| `POST /query/stream` | Real-time SSE streaming |
| `POST /hardware/calibrate` | Auto-tune batch size |

- Three query modes: `agentic`, `naive`, `retrieval_only`.
- CORS for Vercel frontend. Dockerized. Render-deployable.
- SSE streams agent events to the UI in real time.

---

## Slide 10 — Performance Results

**Title:** Measured Performance

| Stage | Time |
|---|---|
| PDF load + clean (24 pages) | ~5 ms |
| Chunking (221 chunks) | < 1 ms |
| Embedding 221 chunks (CPU) | ~15,000 ms |
| LanceDB insert | ~400 ms |
| Vector search (Top-15) | ~10 ms |

- **Hardware calibration** auto-selects optimal batch size (tested 4→32).
- **Caching:** unchanged PDFs → zero re-embed time on restart.
- **End-to-end query latency** (retrieval only): < 50 ms after index is ready.

---

## Slide 11 — Conclusion & Future Work

**Title:** Summary & Perspectives

**Achieved:**
- Fully local, end-to-end agentic RAG with a Rust-accelerated core.
- 5-agent pipeline: query refinement, DPS reranking, generation, self-evaluation.
- Production REST API with streaming, Docker, and cloud deployment.
- Built-in hallucination detection with no external evaluation service.

**Future Work:**
- GPU-accelerated embedding (CUDA ONNX backend).
- Multi-modal support (figures and tables as images).
- Persistent multi-turn conversation history.
- Hybrid BM25 + dense retrieval (sparse + dense fusion).
- Fine-tuned domain-specific evaluator model.

---

## Slide 12 — Live Demo

**Title:** Demo

**Show this flow:**
1. Upload a PDF via CLI or API.
2. Index built: show chunking stats + embedding time.
3. Ask a question → show DPS selection (which 2 of 15 were picked).
4. Show evaluator score + final grounded answer with source citation.
5. Ask a trick question the document doesn't contain → show correct refusal (no hallucination).

---

*Generated: 2026-04-19 | PFE-2026 | Agentic-RAG-Rust-Core*
