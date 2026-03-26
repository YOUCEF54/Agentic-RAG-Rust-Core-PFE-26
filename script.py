"""Agentic RAG demo using PDFs from the local `pdfs` folder.

Usage:
1) Put your PDFs in the `pdfs` folder (or list them in `PDF_PATHS`).
2) Install dependencies: `pip install ollama lancedb pypdf`
3) Build/install the Rust extension in `rag_rust` (e.g. `maturin develop`)
4) Run: `python script.py`
"""
import json
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
RUN_AGENT = True
RUN_BENCHMARK = False
BENCHMARK_ITERS = 20
BENCHMARK_QUERIES = [
  "Summarize the main topic.",
  "What are the key conclusions?",
  "List important definitions.",
]
TOP_K = 3
AGENT_MAX_STEPS = 4
AGENT_TEMPERATURE = 0.2

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
  # ÉTAPE 1 : Batch Embedding (Une seule requête HTTP au lieu de 5)
  # Ollama traite la liste beaucoup plus vite que les appels individuels
  response = ollama.embed(model=EMBEDDING_MODEL, input=dataset)
  embeddings = response['embeddings']

    # ÉTAPE 2 : Préparation des données en une liste de dictionnaires
  data = [
      {"id": i, "text": chunk, "vector": embedding}
      for i, (chunk, embedding) in enumerate(zip(dataset, embeddings))
  ]

  # ÉTAPE 3 : Création et insertion massive
  table = db.create_table(
      TABLE_NAME,
      data=data,
      mode="overwrite" if REBUILD_DB else "create",
  )


  return table

build_start = time.perf_counter()
table = get_or_create_table()
build_end = time.perf_counter()
print(f'DB build/open time: {build_end - build_start:.2f}s')

def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  return table.search(query_embedding).limit(top_n).to_arrow().to_pylist()

def format_retrieved(rows):
  lines = []
  for row in rows:
    distance = row.get('_distance', None)
    if distance is None:
      lines.append(f' - {row["text"]}')
    else:
      lines.append(f' - (distance: {distance:.4f}) {row["text"]}')
  return '\n'.join(lines)

AGENT_SYSTEM_PROMPT = """You are a helpful RAG agent.
You have one tool: retrieve(query) which returns relevant context from the PDFs.
When you need more information, respond ONLY with JSON:
{"action": "retrieve", "query": "..."}
When you are ready to answer, respond ONLY with JSON:
{"action": "answer", "final": "..."}
Use the retrieved context to answer, and do not make up facts.
"""

def parse_agent_action(text):
  try:
    return json.loads(text)
  except Exception:
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
      try:
        return json.loads(text[start:end + 1])
      except Exception:
        return None
  return None

def agentic_answer(question):
  messages = [
    {'role': 'system', 'content': AGENT_SYSTEM_PROMPT},
    {'role': 'user', 'content': question},
  ]

  for step in range(AGENT_MAX_STEPS):
    response = ollama.chat(
      model=LANGUAGE_MODEL,
      messages=messages,
      stream=False,
      options={'temperature': AGENT_TEMPERATURE},
    )
    content = response['message']['content'].strip()
    action = parse_agent_action(content)

    if not action or 'action' not in action:
      return content

    if action['action'] == 'retrieve':
      query = action.get('query', '').strip()
      if not query:
        return "I need a query to retrieve more context."
      query_start = time.perf_counter()
      rows = retrieve(query, top_n=TOP_K)
      query_end = time.perf_counter()
      print(f'Retrieval time: {query_end - query_start:.2f}s')

      context_text = format_retrieved(rows) if rows else ' - (no results)'
      messages.append({'role': 'assistant', 'content': content})
      messages.append({'role': 'system', 'content': f'Retrieved context:\n{context_text}'})
      continue

    if action['action'] == 'answer':
      return action.get('final', '').strip() or "No answer provided."

    return "Unsupported agent action."

  return "Reached the maximum number of agent steps without a final answer."

input_query = input('Ask me a question: ').strip()
if not input_query:
  input_query = "What is this document about?"

if RUN_LLM and RUN_AGENT:
  print('Chatbot response:')
  print(agentic_answer(input_query))
elif RUN_LLM:
  retrieved_knowledge = retrieve(input_query, top_n=TOP_K)
  context_text = format_retrieved(retrieved_knowledge)
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
