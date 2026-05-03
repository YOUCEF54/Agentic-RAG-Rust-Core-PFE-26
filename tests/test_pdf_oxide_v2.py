from pdf_oxide import PdfDocument
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from collections import defaultdict
import rag_rust
import numpy as np

docs = np.array(rag_rust.load_pdf_pages_many(["pdfs/2604.01733v1.pdf"]))

print(docs[0])