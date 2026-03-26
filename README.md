# Agentic-RAG-Rust-Core-PFE-26

Agentic RAG prototype that loads PDFs, chunks text in Rust, embeds with Ollama,
stores vectors in LanceDB, and answers questions by iteratively retrieving context.

**Features**
- PDF ingestion from `pdfs/`
- Rust-based smart chunking (`rag_rust`)
- Vector search with LanceDB
- Agentic loop that decides when to retrieve
- Optional LLM response generation with Ollama

**Project Structure**
- `script.py`: main Python pipeline
- `rag_rust/`: Rust extension module (PyO3)
- `pdfs/`: local PDF files (ignored by git)
- `lancedb/`: local vector store (ignored by git)

**Setup**
1. Create a virtual environment (optional).
2. Install Python deps:
   `pip install -r requirements.txt`
3. Build the Rust extension:
   `cd rag_rust`
   `maturin develop`
4. Put your PDFs in `pdfs/`.

**Run**
`python script.py`

**Notes**
- If you do not have Ollama running, the retrieval still works but LLM output
  will fail.
- To rebuild the vector DB, set `REBUILD_DB = True` in `script.py`.
- To disable the agentic loop, set `RUN_AGENT = False` in `script.py`.
