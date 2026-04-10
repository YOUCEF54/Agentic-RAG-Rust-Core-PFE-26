# import os
# import time
# import rag_rust

# # --- UNLEASH THE CPU ---
# # Remove the "2" thread limits. Let it use all 8 logical processors.
# # os.environ["MKL_NUM_THREADS"] = "2"
# # os.environ["OMP_NUM_THREADS"] = "2"
# rag_rust.load_embed_model()

# # Warm-up (The first call will be slow as it loads models into threads)
# # rag_rust.embed_texts_rust(["warmup"] * 10)

# t = time.perf_counter()
# # Processing 101 texts
# results = rag_rust.embed_texts_rust(["""prompt tuning allows models to retain general knowledge while
# adapting to specialized content. This approach has shown
# promise across various NLP tasks; however, its potential for
# improving retrieval performance in technical question
# answering remains underexplored.
# Despite the progress in these areas, there remains a need for
# comprehensive frameworks that integrate synthetic query
# generation, refined parsing, and adapter tuning to optimize
# retrieval in technical domains. Our work addresses this gap by
# presenting Technical-Embeddings, a framework that combines
# these methodologies to enhance the accuracy and relevance of
# technical question answering systems. The contributions of this
# research build upon existing literature while pushing the
# boundaries of what is achievable in the retrieval of technical
# information"""] * 101)

# print(f"Final Time: {(time.perf_counter() - t) * 1000:.2f}ms")

import os
import time
import rag_rust


# Force the CPU to work before the timer starts
print("Waking up CPU...")
rag_rust.load_embed_model()
for _ in range(3):
    rag_rust.embed_texts_rust(["warmup"] * 64)


num_texts = 256
payload = (
    "ONNX Runtime already squeezes a lot of parallelism out of the CPU, so the fastest "
    "setup is often one process with many ORT threads. It keeps one shared copy of "
    "can reduce throughput because it increases memory-bandwidth pressure, cache misses, "
    "weights, stays cache-friendly, and avoids IPC overhead. Spinning up many processes "
    "and can trigger thread oversubscription (ORT + MKL/OpenMP/OpenBLAS) unless you cap threads."
)

t = time.perf_counter()
_ = rag_rust.embed_texts_rust([payload] * num_texts)
duration_s = time.perf_counter() - t
duration_ms = duration_s * 1000
chunks_per_s = num_texts / duration_s if duration_s > 0 else 0.0

print(f"Final Time: {duration_ms:.2f}ms")
print(f"Throughput: {chunks_per_s:.2f} chunks/s")

if duration_ms > 3000:
    print("NOTE: If performance is low, check CPU clock speed and power plan.")
