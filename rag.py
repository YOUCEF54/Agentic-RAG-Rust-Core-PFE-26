import time
from sentence_transformers import SentenceTransformer
import lancedb
import numpy as np
import requests
import os
from dotenv import load_dotenv

# load search tool
from agent import search

# --- STAGE 1: File I/O ---
start_io = time.time()
with open("documents.txt", "r") as file:
    text = file.read()
print(f"File Read: {time.time() - start_io:.4f}s")

# --- STAGE 2: Text Processing ---
start_proc = time.time()
chunks = [chunk for chunk in text.split("\n\n") if chunk.strip()]
print(f"Text Chunking: {time.time() - start_proc:.4f}s")

# --- STAGE 3: Model Loading & Encoding ---
start_model = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2") # Loading weights is heavy

try:
    vectors = np.load('vectors.npy')
    print("Loaded vectors from disk.")
except FileNotFoundError:
    print("Encoding chunks (this might take a while)...")
    vectors = model.encode(chunks)
    np.save('vectors.npy', vectors)
print(f"Model/Embedding Stage: {time.time() - start_model:.4f}s")

# --- STAGE 4: Database Initialization ---
# start_db_init = time.time()
# client = chromadb.Client()
# collection = client.create_collection("my_docs")
# print(f"ChromaDB Init: {time.time() - start_db_init:.4f}s")

# Switching to Lancedb
db = lancedb.connect("./lance_db")

# --- STAGE 5: Data Conversion & Insertion ---
start_insert = time.time()

ids = [str(id) for id in range(len(chunks))]
embeddings = vectors.tolist() # Converting large NumPy arrays to lists is slow

data = [{'id': i, 'text': t, 'vector': v} for i, t, v in zip(ids,chunks,embeddings)]
# print(f"DB Insertion: {time.time() - start_insert:.4f}s")


# Creating a table in lanceDB
table = ""
try:
    table = db.create_table("docs", data)
except ValueError:
    print("Table Already Exists!")
    table = db["docs"]
print("table count: ", table.count_rows() )
print("table head: ", table.head() )


# Searching
query = "what is the person's name that was mentioned ?"
query_embedding = model.encode(query)
print(f"query :{query}\n query_embedding: {query_embedding.shape  }")

results = table.search( query_embedding.tolist() ).limit(2).to_list()
#print(f"results: {results} ")


context_list = [ row["text"] for row in results ]
context = "\n\n".join(context_list)

question = query 

#proompt = f"answer the question using the available tools \n\nContext: \n{context}\n Question:\n{question}"
# failed !proompt = f"answer the question using the available tools \n\n Question:\n{question}"
proompt = f"answer the question using the available tools ( hint: use the tool to look for author , the database is small so any search in the direction of the hint will be enough )\n\n Question:\n{question}"

print(f"proompt: \n{proompt}")

# Proomting OpenRouter
load_dotenv()
api_key = os.getenv("API_KEY") 
model = os.getenv("MODEL")

tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search in the vector database for semantic similarity",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the text that will be converted to vector and search the vector db using it"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

header = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
}

body = {
        "model": model,
        "messages": [
            {"role": "user", "content": proompt}
         ],
        "tools": tools,
        "tool_choice": "auto"
}




url  = "https://openrouter.ai/api/v1/chat/completions"
response = requests.post(url, json=body, headers=header)
print("\n\nResponse status: ", response.status_code)
print("\n\nResponse json: ", response.json())

