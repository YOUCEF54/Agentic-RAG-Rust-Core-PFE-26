import os
from pathlib import Path
import io
import sys
import rag_rust

# --- Platform/Env ---
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PDF_DIR = os.getenv("PDF_FOLDER", "pdfs")
PDF_PATHS: list[str] = []

# ── Configuration du chunking sémantique ──────────────────────────────
CHUNK_CONFIG = {
    "max_chars": 2000,        # Limite de caractères par chunk (≈ 400-500 tokens)
    "window_size": 5,         # Nombre de phrases par fenêtre de comparaison
    "threshold_percentile": 0.85,  # Sensibilité : 0.85 = coupe dans les 15% de changements les plus forts
}

def get_pdf_paths() -> list[Path]:
    if PDF_PATHS:
        return [Path(p) for p in PDF_PATHS]
    pdf_dir = Path(PDF_DIR)
    return sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []


paths = get_pdf_paths()

if not paths:
    print("No PDFs found!")
else:
    print(f"Processing: {paths}")
    
    try:
        pages = rag_rust.load_pdf_pages_pdfium_many([str(p) for p in paths])
        print(f"Extracted {len(pages)} pages.")
        
        with open("pages.txt", "w", encoding="utf-8") as f:
            for page_idx, page in enumerate(pages):
                # ← NOUVEAU : chunking sémantique au lieu de sliding window basique
                chunks = rag_rust.semantic_window_chunker_advanced(
                    text=page,
                    max_chars=CHUNK_CONFIG["max_chars"],
                    window_size=CHUNK_CONFIG["window_size"],
                    threshold_percentile=CHUNK_CONFIG["threshold_percentile"]
                )
                
                print(f"Page {page_idx+1}: {len(chunks)} chunks (semantic)")
                
                for chunk_idx, chunk in enumerate(chunks):
                    f.write(f"--- Page {page_idx+1} Chunk {chunk_idx+1} ---\n{chunk}\n\n")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()