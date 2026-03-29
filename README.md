# Agentic-RAG-Rust-Core-PFE-26

Minimal RAG prototype that loads PDFs, chunks text in Rust, embeds with
`sentence-transformers` (bge-small-en-v1.5), stores vectors in LanceDB, and
answers questions from retrieved context via OpenRouter.

**Features**
- PDF ingestion from `pdfs/`
- Rust-based smart chunking (`rag_rust`)
- Rust batch PDF loader (one call, multi-file) with parallelism
- Vector search with LanceDB (handled in Rust)
- Local embeddings with `sentence-transformers`
- Optional LLM response generation with OpenRouter

**Project Structure**
- `api.py`: FastAPI server for the frontend
- `rag_rust/`: Rust extension module (PyO3)
- `pdfs/`: local PDF files (ignored by git)
- `lancedb/`: local vector store (ignored by git)

**Setup**
1. Create a virtual environment (optional).
2. Install Python deps:
   `pip install -r requirements.txt`
3. Set your OpenRouter key:
   `setx OPENROUTER_API_KEY "<YOUR_KEY>"`
   Optional (required for some free models):
   `setx OPENROUTER_HTTP_REFERER "http://localhost:3000,https://agentic-rag-rust-core-frontend-pfe.vercel.app/"`
   `setx OPENROUTER_TITLE "Agentic-RAG-Rust-Core-PFE-26"`
4. Build the Rust extension:
   `cd rag_rust`
   `maturin develop`
5. Put your PDFs in `pdfs/`.

**CORS (Frontend)**
- Dev quick fix:
  `setx CORS_ALLOW_ALL true`
- Or explicit:
  `setx CORS_ORIGINS "https://agentic-rag-rust-core-frontend-pfe.vercel.app/,http://localhost:3000,http://127.0.0.1:3000"`

**Hugging Face (optional)**
- If you hit 401 when downloading the embedding model:
  `setx HF_TOKEN "<YOUR_HF_TOKEN>"`

**Run (API)**
`uvicorn api:app --reload`

**Notes**
- Make sure `OPENROUTER_API_KEY` is set before running the API.
- Use `POST /index` to rebuild the vector DB.

**API Endpoints**
- `GET /health`
  - Health check.
- `POST /documents`
  - Upload one or more PDF files (multipart form).
- `GET /documents`
  - List uploaded PDFs.
- `POST /index`
  - Build or rebuild the vector index from uploaded PDFs.
  - Body: `{"rebuild": true, "max_pages": null}`
- `POST /query`
  - Query the index and optionally call OpenRouter for an answer.
  - Body:
    `{"question": "...", "top_k": 3, "chat_model": "openrouter/free", "use_llm": true}`

**Example Requests**
```bash
curl -X POST http://localhost:8000/documents \
  -F "files=@./data/pdfs/doc1.pdf" \
  -F "files=@./data/pdfs/doc2.pdf"

curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"rebuild": true, "max_pages": null}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the topic?", "top_k":3, "use_llm":true}'
```
