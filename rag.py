import os
import time

import lancedb
import requests
from dotenv import load_dotenv
from pdf_oxide import PdfDocument
from sentence_transformers import SentenceTransformer


start_io = time.time()

pdf_path = os.getenv("PDF_PATH", "../rust-01/rag-app/data/documents.pdf")
doc = PdfDocument(pdf_path)
file_content = ""
page_count = doc.page_count() if callable(doc.page_count) else doc.page_count

for i in range(page_count):
    page_text = doc.extract_text(i)
    if page_text:
        file_content += page_text + "\n\n"

print(f"PDF Read & Extract: {(time.time() - start_io) * 1000:.4f}ms")


start_proc = time.time()

chunks = []
chunk_size = 500
text = file_content.strip()

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size].strip()
    if chunk:
        chunks.append(chunk)

print(f"Text Chunking: {(time.time() - start_proc) * 1000:.4f}ms")


print("Loading embedding model...")
start_model = time.time()

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoding chunks in bulk...")
vectors = model.encode(chunks)

print(f"Model/Embedding Stage: {(time.time() - start_model) * 1000:.4f}ms")


start_insert = time.time()

db = lancedb.connect("./lance_db")

ids = []
data = []

for i in range(len(chunks)):
    ids.append(str(i))

for i in range(len(chunks)):
    data.append(
        {
            "id": ids[i],
            "text": chunks[i],
            "vector": vectors[i].tolist(),
        }
    )

print("Inserting documents into LanceDB...")

try:
    db.drop_table("docs")
except Exception:
    pass

table = db.create_table("docs", data)

print(f"DB Init & Insertion: {(time.time() - start_insert) * 1000:.4f}ms")
print("off profiling: rows_count :", table.count_rows())


start_search = time.time()

question = "Question: According to the abstract, what specific type of 'framework' does this paper propose to support knowledge management and decision-making?"
query_embedding = model.encode(question)
results = table.search(query_embedding.tolist()).limit(2).to_list()

context_list = []
for row in results:
    context_list.append(row["text"])

context = "\n\n".join(context_list)

print(f"Search Stage: {(time.time() - start_search) * 1000:.4f}ms")
print(context)


start_llm = time.time()

load_dotenv()
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL")

answer = (
    "Use the following context to answer the question.\n\n"
    f"Context:\n{context}\n\nQuestion: {question}"
)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

body = {
    "model": model_name,
    "messages": [
        {"role": "user", "content": answer}
    ],
}

url = "https://openrouter.ai/api/v1/chat/completions"
response = requests.post(url, json=body, headers=headers)

print(f"LLM Request Stage: {(time.time() - start_llm) * 1000:.4f}ms")

try:
    print("\nAnswer:", response.json()["choices"][0]["message"]["content"])
except Exception:
    print("\nAnswer:", response.text)
