# rag/retriever.py
from typing import List, Dict

def top_k(coll, query_embedding: list, k: int = 4) -> list:
    # NOTE: In Chroma 0.5.x, "ids" is NOT allowed in `include`.
    res = coll.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],  # <-- removed "ids"
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids   = res.get("ids", [[]])[0]  # ids are returned regardless of `include`

    hits = []
    for doc, id_, meta, dist in zip(docs, ids, metas, dists):
        hits.append({
            "id": id_,
            "text": doc,
            "score": float(dist),
            "source": meta.get("source", ""),
            "chunk": meta.get("chunk", 0),
        })
    return hits
