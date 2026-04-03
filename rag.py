import os
import time

import lancedb
import ollama
from dotenv import load_dotenv
from pdf_oxide import PdfDocument
from semantic_text_splitter import TextSplitter       # pip install semantic-text-splitter
from sentence_transformers import SentenceTransformer



load_dotenv()
api_key    = os.getenv("API_KEY")
OLLAMA_MODEL = "llama3.2"
pdf_path   = os.getenv("PDF_PATH", "../rust-01/rag-app/data/documents.pdf")

print("Loading embedding model (once)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

QUESTION = (
    "Question: According to the abstract, what specific type of 'framework' "
    "does this paper propose to support knowledge management and decision-making?"
)

context = ""

def build_vector_db(pdf_path: str):
    # 1. Read PDF
    doc = PdfDocument(pdf_path)
    file_content = ""
    page_count = doc.page_count() if callable(doc.page_count) else doc.page_count
    for i in range(page_count):
        page_text = doc.extract_text(i)
        if page_text:
            file_content += page_text + "\n\n"

    # 2. Chunk
    splitter = TextSplitter(1200)   # 1200 chars — same as the paper
    chunks = splitter.chunks(file_content)

    # 3. Embed
    vectors = embed_model.encode(chunks)

    # 4. Store in LanceDB
    db = lancedb.connect("./lance_db")
    data = [
        {"id": str(i), "text": chunks[i], "vector": vectors[i].tolist()}
        for i in range(len(chunks))
    ]
    table = db.create_table("docs", data, mode="overwrite")

    print(f"✅ Indexed {table.count_rows()} chunks")
    return table

def retriever(query: str, table, k: int = 3) -> list[str]:
    query_vector = embed_model.encode(query).tolist()
    results = table.search(query_vector).limit(k).to_list()
    chunks = [row["text"] for row in results]
    return chunks


table = build_vector_db(pdf_path)
query = "What framework does the paper propose ?"
chunks = retriever(query, table)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---\n{chunk[:300]}")


def generator(query: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)
    
    prompt = f"""Use ONLY the context below to answer the question.
                    If the context doesn't have enough info, say "Insufficient information".
                    Keep the answer beginner-friendly and cite which part of the context you used.
                    
                    Context:
                    {context}
                    
                   Question: {query}
               """
    
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]



def evaluator(query: str, answer: str) -> dict:
    prompt = f"""You are evaluating an AI-generated answer.
                
                Question: {query}
                Answer: {answer}
                
                Check these 3 things:
                1. Clarity — is it easy to understand?
                2. Relevance — does it actually answer the question?
                3. Completeness — is anything important missing?
                
                Reply in this exact format:
                VERDICT: satisfactory
                REASON: <one sentence>
                
                OR
                
                VERDICT: unsatisfactory
                REASON: <one sentence explaining what's missing>
                """

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response["message"]["content"].strip()

    verdict = "unsatisfactory"
    reason = ""

    for line in raw.splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip().lower()
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return {"verdict": verdict, "reason": reason}

def query_refiner(query: str, reason: str) -> str:
    prompt = f"""You are rewriting a search query to get a better answer.

Original query: {query}
Problem with the answer: {reason}

Write an improved version of the query that is more specific and addresses the problem.
Output ONLY the new query, nothing else.
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip()




answer = generator(query, chunks)
evaluation = evaluator(query, answer)
print("evaluation: \n", evaluation)









refined = query_refiner(query, "The answer was too vague about the framework type")
print(refined)
