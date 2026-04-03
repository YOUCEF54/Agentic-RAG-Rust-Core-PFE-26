import json
import os
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')

import lancedb
import requests
from dotenv import load_dotenv
from pdf_oxide import PdfDocument
from semantic_text_splitter import TextSplitter       # pip install semantic-text-splitter
from sentence_transformers import SentenceTransformer
from agent import search

# ── Setup: load env + model ───────────────────────────────────────────────────
load_dotenv()
api_key    = os.getenv("API_KEY")
model_name = os.getenv("MODEL")
pdf_path   = os.getenv("PDF_PATH", "../rust-01/rag-app/data/documents.pdf")

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

QUESTION = (
    "Question: According to the abstract, what specific type of 'framework' "
    "does this paper propose to support knowledge management and decision-making?"
)

print(f"\n{'─' * 50}")
print("Running RAG pipeline...")

# 1. PDF read & extract ────────────────────────────────────────────────────
t0 = time.perf_counter()

doc = PdfDocument(pdf_path)
file_content = ""
page_count = doc.page_count() if callable(doc.page_count) else doc.page_count
for i in range(page_count):
    page_text = doc.extract_text(i)
    if page_text:
        file_content += page_text + "\n\n"

pdf_read_ms = (time.perf_counter() - t0) * 1000
print(f"  PDF Read & Extract:    {pdf_read_ms:.4f} ms")

# 2. Text chunking — same splitter as Rust (character-based, capacity=500) ─
t0 = time.perf_counter()

splitter = TextSplitter(500)
chunks = splitter.chunks(file_content)

chunking_ms = (time.perf_counter() - t0) * 1000
print(f"  Text Chunking:         {chunking_ms:.4f} ms")

# 3. Bulk embedding ────────────────────────────────────────────────────────
t0 = time.perf_counter()

vectors = embed_model.encode(chunks)

embedding_ms = (time.perf_counter() - t0) * 1000
print(f"  Model/Embedding:       {embedding_ms:.4f} ms")

# 4. DB insertion (overwrite mode — no drop/recreate needed) ───────────────
t0 = time.perf_counter()

db = lancedb.connect("./lance_db")
data = [
    {"id": str(i), "text": chunks[i], "vector": vectors[i].tolist()}
    for i in range(len(chunks))
]
table = db.create_table("docs", data, mode="overwrite")

db_insert_ms = (time.perf_counter() - t0) * 1000
print(f"  DB Init & Insertion:   {db_insert_ms:.4f} ms  ({table.count_rows()} rows)")

# 5. Vector search ─────────────────────────────────────────────────────────
t0 = time.perf_counter()

query_embedding = embed_model.encode(QUESTION)
results = table.search(query_embedding.tolist()).limit(2).to_list()
context = "\n\n".join(row["text"] for row in results)

search_ms = (time.perf_counter() - t0) * 1000
print(f"  Search:                {search_ms:.4f} ms")

# ── LLM call ───────────────────────────────────────────────────────────────────
print(f"\n{'─' * 50}")
print("Running LLM query...")

answer_prompt = (
    "Use the following context to answer the question.\n\n"
    f"Context:\n{context}\n\nQuestion: {QUESTION}"
)
headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
}
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "deos similarity search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the text we want to find similar chunks to it"
                    }
                },
                "required": ["query"]
            }
        }
    }
]
body    = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": answer_prompt}
        ],
        "tools": tools,
        "tool_choice": "auto"
}

response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers)


response_json  = response.json()
print("\nAnswer:", response_json["choices"][0]["message"]["content"])
fn_name = response_json["choices"][0]["message"]["tool_calls"][0]["function"]["name"] # search
fn_arguments = response_json["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] # the 
fn_arguments = json.loads(fn_arguments)


print("\nfunction name : ", fn_name) # is a string
print("\nargument(query): ",fn_arguments) # fn_arguments is a dict
match fn_name :
    case "search":
        if fn_arguments["query"].strip():
            tool_result = search(fn_arguments["query"])
            print("tool result : ",tool_result)
    case _:
        print("\n\n Didn't call teh tool") 
