import os, hashlib
from typing import Dict, List, Tuple
import chromadb
from chromadb import PersistentClient

def hash_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

def get_collection(collection_name: str, persist_dir: str):
    os.makedirs(persist_dir, exist_ok=True)
    client = PersistentClient(path=persist_dir)
    return client.get_or_create_collection(collection_name)

def add_chunks(coll, chunks: List[str], sources: List[str], indices: List[int], embeddings: List[List[float]]):
    ids = [f"{src}-{i}-{hash_id(ch)}" for ch, src, i in zip(chunks, sources, indices)]
    metas = [{"source": s, "chunk": i} for s, i in zip(sources, indices)]
    coll.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metas)
    return ids
