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
- `script.py`: main Python pipeline
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
   `setx OPENROUTER_HTTP_REFERER "http://localhost:3000"`
   `setx OPENROUTER_TITLE "Agentic-RAG-Rust-Core-PFE-26"`
4. Build the Rust extension:
   `cd rag_rust`
   `maturin develop`
5. Put your PDFs in `pdfs/`.

**Run**
`python script.py`

**Notes**
- Make sure `OPENROUTER_API_KEY` is set before running `script.py`.
- To rebuild the vector DB, set `REBUILD_DB = True` in `script.py`.
