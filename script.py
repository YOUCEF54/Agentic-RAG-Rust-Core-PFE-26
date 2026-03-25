"""Minimal RAG demo using PDFs from the local `pdfs` folder.

Usage:
1) Put your PDFs in the `pdfs` folder (or list them in `PDF_PATHS`).
2) Install dependencies: `pip install ollama lancedb pypdf`
3) Build/install the Rust extension in `rag_rust` (e.g. `maturin develop`)
4) Run: `python script.py`
"""
import time
from pathlib import Path

import ollama
import lancedb
import rag_rust

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'tinyllama'

# Storage settings
DB_DIR = 'lancedb'
TABLE_NAME = 'pdf_chunks'
REBUILD_DB = True

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
    from pypdf import PdfReader
  except Exception:
    try:
      from PyPDF2 import PdfReader
    except Exception as exc:
      raise ImportError(
        "Missing PDF dependency. Install one of: pypdf, PyPDF2"
      ) from exc

  texts = []
  for path in paths:
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
      if MAX_PAGES is not None and i >= MAX_PAGES:
        break
      text = page.extract_text() or ""
      if text.strip():
        texts.append(text)
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
  print(f'Loaded {len(pages)} PDF pages in {pdf_end - pdf_start:.2f}s')

  chunk_start = time.perf_counter()
  chunks = []
  for page_text in pages:
    chunks.extend(chunk_text(page_text))
  chunk_end = time.perf_counter()
  print(f'Chunked into {len(chunks)} chunks in {chunk_end - chunk_start:.2f}s')
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

  # Build (or rebuild) the table.
  first_embedding = ollama.embed(model=EMBEDDING_MODEL, input=dataset[0])['embeddings'][0]
  table = db.create_table(
    TABLE_NAME,
    data=[{"id": 0, "text": dataset[0], "vector": first_embedding}],
    mode="overwrite" if REBUILD_DB else "create",
  )

  for i, chunk in enumerate(dataset[1:], start=1):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    table.add([{"id": i, "text": chunk, "vector": embedding}])
    print(f'Added chunk {i+1}/{len(dataset)} to the database')

  return table

build_start = time.perf_counter()
table = get_or_create_table()
build_end = time.perf_counter()
print(f'DB build/open time: {build_end - build_start:.2f}s')

def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  return table.search(query_embedding).limit(top_n).to_arrow().to_pylist()

input_query = input('Ask me a question: ').strip()
if not input_query:
  input_query = "What is this document about?"
query_start = time.perf_counter()
retrieved_knowledge = retrieve(input_query)
query_end = time.perf_counter()
print(f'Retrieval time: {query_end - query_start:.2f}s')

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

  stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
      {'role': 'system', 'content': instruction_prompt},
      {'role': 'user', 'content': input_query},
    ],
    stream=True,
  )

  # print the response from the chatbot in real-time
  print('Chatbot response:')
  for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

# Record the end time
end_time = time.perf_counter()
# Calculate and print the execution time
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")

if RUN_BENCHMARK:
  total_queries = BENCHMARK_ITERS * len(BENCHMARK_QUERIES)
  bench_start = time.perf_counter()
  for _ in range(BENCHMARK_ITERS):
    for q in BENCHMARK_QUERIES:
      retrieve(q)
  bench_end = time.perf_counter()
  bench_total = bench_end - bench_start
  print(f'Benchmark: {bench_total:.2f}s for {total_queries} queries')
  print(f'Avg per query: {bench_total / total_queries:.4f}s')
