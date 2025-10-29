from typing import List
import re

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = normalize_ws(text)
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n: break
        start = end - overlap if end - overlap > 0 else end
    return chunks
