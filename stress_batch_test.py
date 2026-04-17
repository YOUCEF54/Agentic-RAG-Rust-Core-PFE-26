import os
import time
import psutil
import rag_rust
from statistics import mean

# ========================= CONFIG =========================
CHUNK_SIZE = 800          # or 1000 to make it even heavier
NUM_CHUNKS_TO_TEST = 512   # much larger than your normal 89
BATCH_SIZES_TO_TEST = [4, 8, 16, 32, 64, 128]

REPEATS = 4
# =========================================================

def create_long_test_chunks(n: int, length: int = 750) -> list[str]:
    """Create fake chunks that are realistically long"""
    base_text = "This is a test chunk for stress testing embedding performance with long sequences. " * 20
    return [base_text[:length] for _ in range(n)]

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("=== FastEmbed Batch Size Stress Test ===")
print(f"Testing with {NUM_CHUNKS_TO_TEST} chunks of ~{CHUNK_SIZE} chars")
print(f"Hardware: {os.cpu_count()} logical cores | i7-6700HQ\n")

rag_rust.load_embed_model()

for batch_size in BATCH_SIZES_TO_TEST:
    print(f"\n→ Testing batch_size = {batch_size}")

    test_chunks = create_long_test_chunks(NUM_CHUNKS_TO_TEST, CHUNK_SIZE)
    
    times = []
    peak_mem = 0
    
    for r in range(REPEATS):
        start_mem = get_memory_mb()
        t0 = time.perf_counter()
        
        try:
            embeddings = rag_rust.embed_texts_rust([f"passage: {c}" for c in test_chunks],batch_size)
            dt = time.perf_counter() - t0
            times.append(dt)
            
            current_mem = get_memory_mb()
            peak_mem = max(peak_mem, current_mem)
            
            print(f"   Run {r+1}/{REPEATS}: {dt:.3f}s | Mem: {current_mem:.1f} MB")
            
            # Basic sanity check
            if len(embeddings) != NUM_CHUNKS_TO_TEST:
                print(f"   WARNING: Expected {NUM_CHUNKS_TO_TEST} embeddings, got {len(embeddings)}")
                
        except Exception as e:
            print(f"   FAILED with batch={batch_size}: {e}")
            break
    
    if times:
        avg_time = mean(times)
        print(f"   → Average: {avg_time:.3f}s ({NUM_CHUNKS_TO_TEST/avg_time:.1f} chunks/sec) | Peak Mem: {peak_mem:.1f} MB")

print("\nTest finished.")