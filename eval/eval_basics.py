"""Simple retrieval sanity checks.
Run: python -m eval.eval_basics
"""
import os
from dotenv import load_dotenv
from rag.embedder import Embedder
from rag.store import get_collection
from rag.retriever import top_k

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdfs")

def demo():
    coll = get_collection(COLLECTION_NAME, CHROMA_DIR)
    emb = Embedder(EMBEDDING_MODEL)
    for q in [
        "What is the main contribution?",
        "Summarize the experimental setup.",
        "List the key results."
    ]:
        qv = emb.encode([q])[0]
        hits = top_k(coll, qv, k=4)
        print("\nQ:", q)
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {h['source']}#{h['chunk']} score={h['score']:.3f}")
            print(h['text'][:160].replace("\n"," ") + " ...")

if __name__ == "__main__":
    demo()
