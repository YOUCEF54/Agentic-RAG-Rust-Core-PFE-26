import argparse
from pathlib import Path

import rag_rust


def pick_default_pdf() -> Path | None:
    candidates = []
    for folder in ("pdfs", "data/pdfs"):
        path = Path(folder)
        if path.exists():
            candidates.extend(sorted(path.glob("*.pdf")))
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Test visual element extraction from PDFs.")
    parser.add_argument("--pdf", type=str, help="Path to a PDF file.")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages to scan.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf) if args.pdf else pick_default_pdf()
    if not pdf_path or not pdf_path.exists():
        raise SystemExit("No PDF found. Use --pdf to provide a file.")

    items = rag_rust.extract_visual_elements(str(pdf_path), args.max_pages)

    print(f"PDF: {pdf_path}")
    print(f"Found {len(items)} visual references.")
    for kind, page, snippet in items:
        print(f"- [{kind}] page {page}: {snippet}")


if __name__ == "__main__":
    main()
