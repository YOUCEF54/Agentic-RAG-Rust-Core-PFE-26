
import os

import rag_rust
import time

# warm up — loads model into OnceCell
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"   
rag_rust.load_embed_model()

# larger batch with 810 characters
t = time.perf_counter()
rag_rust.embed_texts_rust(["""prompt tuning allows models to retain general knowledge while
adapting to specialized content. This approach has shown
promise across various NLP tasks; however, its potential for
improving retrieval performance in technical question
answering remains underexplored.
Despite the progress in these areas, there remains a need for
comprehensive frameworks that integrate synthetic query
generation, refined parsing, and adapter tuning to optimize
retrieval in technical domains. Our work addresses this gap by
presenting Technical-Embeddings, a framework that combines
these methodologies to enhance the accuracy and relevance of
technical question answering systems. The contributions of this
research build upon existing literature while pushing the
boundaries of what is achievable in the retrieval of technical
information"""] * 100)
print(f"100 chunks: {(time.perf_counter()-t)*1000:.2f}ms")