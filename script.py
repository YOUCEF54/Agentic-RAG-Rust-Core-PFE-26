"""Minimal RAG demo using PDFs from the local `pdfs` folder.

Usage:
1) Put your PDFs in the `pdfs` folder (or list them in `PDF_PATHS`).
2) Install dependencies: `pip install lancedb requests sentence-transformers`
3) Build/install the Rust extension in `rag_rust` (e.g. `maturin develop`)
4) Set `OPENROUTER_API_KEY` in your environment.
5) Run: `python script.py`
"""
import os
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
import lancedb
import rag_rust
import requests
import sys
import io

# Force l'encodage UTF-8 pour le terminal
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "Agentic-RAG-Rust-Core-PFE-26")
OPENROUTER_TIMEOUT = 60
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openrouter/free")
EMBED_MODEL_NAME = "bge-small-en-v1.5"
CHAT_TEMPERATURE = 0.2
EMBED_BATCH_SIZE = 32
EMBED_NORMALIZE = True
TOP_K = 3

# Storage settings
DB_DIR = 'lancedb'
TABLE_NAME = 'pdf_chunks'
REBUILD_DB = False

# PDF ingestion settings
PDF_DIR = 'pdfs'
PDF_PATHS = []  # Optional explicit list of PDF paths.
MAX_PAGES = None
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Runtime options
RUN_LLM = True
RUN_BENCHMARK = False
BENCHMARK_ITERS = 20
BENCHMARK_QUERIES = [
  "Summarize the main topic.",
  "What are the key conclusions?",
  "List important definitions.",
]

# Record the start time
start_time = time.perf_counter()

def log_run_info():
  print("=== Run Info ===")
  print(f"Chat model: {OPENROUTER_CHAT_MODEL}")
  print(f"Embedding model: {EMBED_MODEL_NAME}")
  print(f"Embedding batch size: {EMBED_BATCH_SIZE}")
  print(f"Embedding normalize: {EMBED_NORMALIZE}")
  print(f"Chunk size: {CHUNK_SIZE}")
  print(f"Chunk overlap: {CHUNK_OVERLAP}")
  print(f"MAX_PAGES: {MAX_PAGES}")
  try:
    embedder = get_embedder()
    device = getattr(embedder, "device", None) or getattr(embedder, "_target_device", None)
    dim = embedder.get_sentence_embedding_dimension()
    max_len = embedder.get_max_seq_length()
    print(f"Embedding device: {device}")
    print(f"Embedding dimension: {dim}")
    print(f"Embedding max_seq_length: {max_len}")
  except Exception as exc:
    print(f"Embedding model info not available: {exc}")
  print("================")


def openrouter_headers():
  key = os.getenv("OPENROUTER_API_KEY")
  if not key:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment.")

  key = key.strip().replace('"', '').replace("'", "")

  headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
  }
  if OPENROUTER_HTTP_REFERER:
    headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
  if OPENROUTER_TITLE:
    headers["X-Title"] = OPENROUTER_TITLE
  return headers

def openrouter_post(path, payload):
  url = f"{OPENROUTER_BASE_URL}{path}"
  response = requests.post(
    url,
    json=payload,
    headers=openrouter_headers(),
    timeout=OPENROUTER_TIMEOUT,
  )
  if not response.ok:
    detail = ""
    try:
      detail = response.json()
    except Exception:
      detail = response.text

    if response.status_code == 401:
      raise ValueError(
        "OpenRouter auth failed (401). Set OPENROUTER_API_KEY. "
        "For some free models you must also set OPENROUTER_HTTP_REFERER and OPENROUTER_TITLE."
      )
    raise RuntimeError(
      f"OpenRouter error {response.status_code}: {detail}"
    )
  return response.json()

_EMBEDDER = None

def get_embedder():
  global _EMBEDDER
  if _EMBEDDER is None:
    _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
  return _EMBEDDER

def embed_texts(texts):
  embedder = get_embedder()
  vectors = embedder.encode(texts, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=EMBED_NORMALIZE)
  return vectors.tolist()

def chat_complete(messages):
  data = openrouter_post(
    "/chat/completions",
    {
      "model": OPENROUTER_CHAT_MODEL,
      "messages": messages,
      "temperature": CHAT_TEMPERATURE,
    },
  )
  return data["choices"][0]["message"]["content"]

def load_pdf_texts():
  paths = []
  if PDF_PATHS:
    paths = [Path(p) for p in PDF_PATHS]
  else:
    pdf_dir = Path(PDF_DIR)
    if pdf_dir.exists():
      paths = list(pdf_dir.glob('*.pdf'))

  if not paths:
    raise FileNotFoundError(
      "No PDFs found. Set PDF_PATHS or put PDFs in the 'pdfs' folder."
    )

  try:
    pages = rag_rust.load_pdf_pages_many([str(p) for p in paths])
  except Exception as exc:
    raise RuntimeError("Failed to load PDFs via Rust (batch).") from exc

  if MAX_PAGES is not None:
    pages = pages[:MAX_PAGES]

  texts = [text for text in pages if text and text.strip()]
  return texts


def chunk_text(text):
  if not text or not text.strip():
    return []
  return [
    chunk
    for chunk in rag_rust.smart_chunker(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if chunk
  ]

def build_dataset():
  pdf_start = time.perf_counter()
  pages = load_pdf_texts()
  pdf_end = time.perf_counter()
  print(f'Loaded {len(pages)} PDF pages in {(pdf_end - pdf_start)*1000:.2f}ms')

  chunk_start = time.perf_counter()
  chunks = []
  for page_text in pages:
    chunks.extend(chunk_text(page_text))
  chunk_end = time.perf_counter()
  print(f'Chunked into {len(chunks)} chunks in {(chunk_end - chunk_start)*1000:.2f}ms')
  if chunks:
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    print(f'Avg chunk length: {avg_len:.1f} chars')
  return chunks

dataset = build_dataset()

db = lancedb.connect(DB_DIR)

def get_or_create_table():
  if not dataset:
    raise ValueError("Dataset is empty; cannot build the LanceDB table.")

  if not REBUILD_DB:
    try:
      return db.open_table(TABLE_NAME)
    except Exception:
      pass

  # Batch embedding for all chunks in one request.
  embed_start = time.perf_counter()
  embeddings = embed_texts(dataset)
  embed_end = time.perf_counter()
  embed_time = embed_end - embed_start
  if embed_time > 0:
    print(f'Embedding time: {embed_time*1000:.2f}ms ({len(dataset)/embed_time:.2f} chunks/s)')

  # Prepare rows for LanceDB.
  data = [
    {"id": i, "text": chunk, "vector": embedding}
    for i, (chunk, embedding) in enumerate(zip(dataset, embeddings))
  ]

  # Create or overwrite the table in one batch.
  table = db.create_table(
    TABLE_NAME,
    data=data,
    mode="overwrite" if REBUILD_DB else "create",
  )

  return table

build_start = time.perf_counter()
table = get_or_create_table()
build_end = time.perf_counter()
print(f'DB build/open time: {(build_end - build_start)*1000:.2f}ms')

def retrieve(query, top_n=3):
  query_embedding = embed_texts([query])[0]
  return table.search(query_embedding).limit(top_n).to_arrow().to_pylist()

log_run_info()

# input_query = input('Ask me a question: ').strip()
# input_query = "what are the documents required for admission?"
input_query = "what is Samyama?"

if not input_query:
  input_query = "What is this document about?"
query_start = time.perf_counter()
retrieved_knowledge = retrieve(input_query, top_n=TOP_K)
query_end = time.perf_counter()
print(f'Retrieval time: {(query_end - query_start)*1000:.2f}ms')

print('Retrieved knowledge:')
for row in retrieved_knowledge:
    distance = row.get('_distance', None)
    if distance is None:
      print(f' - {row["text"]}')
    else:
      print(f' - (distance: {distance:.4f}) {row["text"]}')

# 1. Prepare the context string outside the f-string
context_text = '\n'.join([f' - {row["text"]}' for row in retrieved_knowledge])

# 2. Use that variable inside the instruction prompt
if RUN_LLM:
  instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context_text}
'''

  llm_start = time.perf_counter()
  response_text = chat_complete([
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ])
  llm_end = time.perf_counter()

  print('Chatbot response:')
  print(response_text)
  print(f'LLM time: {(llm_end - llm_start)*1000:.2f}ms')

# Record the end time
end_time = time.perf_counter()
# Calculate and print the execution time
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time*1000:.4f}ms")

if RUN_BENCHMARK:
  total_queries = BENCHMARK_ITERS * len(BENCHMARK_QUERIES)
  bench_start = time.perf_counter()
  for _ in range(BENCHMARK_ITERS):
    for q in BENCHMARK_QUERIES:
      retrieve(q)
  bench_end = time.perf_counter()
  bench_total = bench_end - bench_start
  print(f'Benchmark: {bench_total:.2f}ms for {total_queries} queries')
  print(f'Avg per query: {bench_total / total_queries:.4f}s')
