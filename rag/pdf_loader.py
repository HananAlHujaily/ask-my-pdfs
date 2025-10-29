from typing import Dict, List
import os, glob, pathlib
from pypdf import PdfReader

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)

def load_folder(folder: str) -> dict:
    folder = os.path.expanduser(folder)
    docs = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
        try:
            docs[pathlib.Path(p).name] = read_pdf(p)
        except Exception as e:
            docs[pathlib.Path(p).name] = f"[PDF read error: {e}]"
    return docs
