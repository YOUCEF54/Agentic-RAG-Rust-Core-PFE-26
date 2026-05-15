"""Professional Parameter Sweep Benchmark for Hybrid RAG
Tests all important combinations of:
- CHUNK_SIZE
- CHUNK_OVERLAP
- EMBED_BATCH_SIZE (Rust ONNX batch size)

Outputs:
- Clear console table
- Full CSV for Excel
- Top 5 best configurations (by chunks/sec)
"""

from __future__ import annotations
import csv
import os
import platform
import time
from pathlib import Path
from statistics import mean, stdev

import rag_rust



# ========================= CONFIG =========================
PDF_DIR = os.getenv("PDF_FOLDER", "pdfs")
DB_DIR = "lancedb"
TABLE_NAME = "pdf_chunks"

# Parameter grid to test (you can edit these)
# CHUNK_SIZES = [700, 800, 900, 1000]
CHUNK_SIZES = [1000]
OVERLAPS     = [150]
NUM_CHUNKS_TO_TEST = 512   # much larger than your normal 89
BATCH_SIZES  = [4, 8, 16, 24, 32, 48, 64, 128]   # what you care about

REPEATS_PER_CONFIG = 3          # for statistical stability
NUM_RUNS_FULL_PIPELINE = 1      # full PDF→DB test (keeps runtime reasonable)

CSV_FILE = "rag_benchmark_results.csv"


# =========================================================
def create_long_test_chunks(n: int, length: int = 750) -> list[str]:
    """Create fake chunks that are realistically long"""
    base_text = "This is a test chunk for stress testing embedding performance with long sequences. " * 20
    return [base_text[:length] for _ in range(n)]

test_chunks = lambda chunk_size : create_long_test_chunks(NUM_CHUNKS_TO_TEST, chunk_size)


def get_pdf_paths() -> list[str]:
    pdf_dir = Path(PDF_DIR)
    return [str(p) for p in sorted(pdf_dir.glob("*.pdf"))]

def print_header():
    print("="*80)
    print("HYBRID RAG PARAMETER SWEEP BENCHMARK")
    print("="*80)
    print(f"CPU          : {platform.processor()}")
    print(f"Python       : {platform.python_version()}")
    print(f"Date         : {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"PDFs         : {get_pdf_paths()}")
    print(f"Tested combos: {len(CHUNK_SIZES) * len(OVERLAPS) * len(BATCH_SIZES)}")
    print("="*80, "\n")

def run_embedding_benchmark(chunks: list[str], batch_size: int) -> dict:
    # os.environ["EMBED_BATCH_SIZE"] = str(batch_size)
    
    times = []
    for _ in range(REPEATS_PER_CONFIG):
        t0 = time.perf_counter()
        _ = rag_rust.embed_texts_rust([f"passage: {c}" for c in chunks], batch_size)
        times.append(time.perf_counter() - t0)
    
    avg_time = mean(times)
    std_time = stdev(times) if len(times) > 1 else 0
    
    return {
        "embed_time_s": avg_time,
        "chunks_per_sec": len(chunks) / avg_time,
        "std_chunks_per_sec": (len(chunks) / avg_time) * (std_time / avg_time) if avg_time > 0 else 0,
    }

def main():
    print_header()
    
    all_pdf_paths = get_pdf_paths()
    if not all_pdf_paths:
        raise FileNotFoundError(f"No PDFs in {PDF_DIR}")

    print("Loading PDF pages once...")
    pages = rag_rust.load_pdf_pages_many(all_pdf_paths)
    pages = [p for p in pages if p and p.strip()]
    print(f"   → {len(pages)} pages loaded\n")

    # CSV header
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "chunk_size", "overlap", "batch_size", "num_chunks", "avg_chunks_per_sec",
            "std_chunks_per_sec", "embed_time_s", "full_pipeline_s"
        ])

    results = []

    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            print(f"Testing chunk_size={chunk_size} | overlap={overlap}")

            # Chunk once for this (size, overlap) pair
            t0 = time.perf_counter()
            chunks = []
            for page in pages:
                chunks.extend(c for c in rag_rust.smart_chunker(page, chunk_size, overlap) if c)
            chunking_time = time.perf_counter() - t0

            print(f"   → {len(chunks)} chunks created ({chunking_time:.2f}s)")

            for batch_size in BATCH_SIZES:
                print(f"     → batch_size={batch_size} ... ", end="")

                # Pure embedding benchmark
                bench = run_embedding_benchmark(chunks, batch_size)

                # Full pipeline benchmark (PDF→chunk→embed→DB) - fewer repeats
                full_times = []
                for _ in range(NUM_RUNS_FULL_PIPELINE):
                    t0 = time.perf_counter()
                    embeddings = rag_rust.embed_texts_rust([f"passage: {c}" for c in chunks], batch_size)
                    rag_rust.lancedb_create_or_open(DB_DIR, TABLE_NAME, chunks, embeddings, True)
                    full_times.append(time.perf_counter() - t0)
                
                full_pipeline_s = mean(full_times)

                row = [
                    chunk_size, overlap, batch_size, len(chunks),
                    round(bench["chunks_per_sec"], 2),
                    round(bench["std_chunks_per_sec"], 2),
                    round(bench["embed_time_s"], 3),
                    round(full_pipeline_s, 3)
                ]

                with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)

                results.append({
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "batch_size": batch_size,
                    "num_chunks": len(chunks),
                    "chunks_per_sec": bench["chunks_per_sec"],
                    "full_pipeline_s": full_pipeline_s
                })

                print(f"{bench['chunks_per_sec']:.1f} chunks/s")

    # === FINAL SUMMARY ===
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE - TOP 5 CONFIGURATIONS")
    print("="*80)
    
    top5 = sorted(results, key=lambda x: x["chunks_per_sec"], reverse=True)[:5]
    for i, r in enumerate(top5, 1):
        print(f"{i:2}. chunk={r['chunk_size']:4} | overlap={r['overlap']:3} | "
              f"batch={r['batch_size']:3} → {r['chunks_per_sec']:6.1f} chunks/s "
              f"(full pipeline: {r['full_pipeline_s']:.1f}s)")

    print(f"\nFull results saved to: {CSV_FILE}")
    print("Open it in Excel → sort by 'avg_chunks_per_sec' descending")
    print("\nYour target config (800 / 100 / 16) should appear in the top results.")

if __name__ == "__main__":
    main()