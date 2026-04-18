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

def get_pdf_paths() -> list[Path]:
    if PDF_PATHS:
        return [Path(p) for p in PDF_PATHS]
    pdf_dir = Path(PDF_DIR) # /pdfs
    return sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []


paths = get_pdf_paths()

if not paths:
    print("No PDFs found!")
else:
    # Process just the first PDF for testing
    #first_pdf = str(paths[0])
    print(f"Processing: {paths}")
    
    try:
        pages = rag_rust.load_pdf_pages_pdfium_many([str(p) for p in paths])
        print(f"Extracted {len(pages)} pages.")
        
        with open("pages.txt", "w", encoding="utf-8") as f:
            for i, page in enumerate(pages):
                f.write(f"--- Page {i+1} ---\n{page}...\n")
    except Exception as e:
        print(f"Error: {e}")
