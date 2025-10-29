import os, click
from dotenv import load_dotenv
from rag.pdf_loader import load_folder
from rag.chunker import chunk_text
from rag.embedder import Embedder
from rag.store import get_collection, add_chunks
from rag.retriever import top_k
from rag.generator import template_answer, openai_answer

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdfs")
GENERATOR = os.getenv("GENERATOR", "none").lower()

@click.group()
def cli():
    pass

@cli.command()
@click.option("--path", required=True, help="Folder with PDFs")
def ingest(path):
    docs = load_folder(path)
    if not docs:
        click.echo("No PDFs found.")
        return
    coll = get_collection(COLLECTION_NAME, CHROMA_DIR)
    emb = Embedder(EMBEDDING_MODEL)
    total = 0
    for name, text in docs.items():
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        vecs = emb.encode(chunks)
        add_chunks(coll, chunks, [name]*len(chunks), list(range(len(chunks))), vecs)
        total += len(chunks)
        click.echo(f"Ingested {name}: {len(chunks)} chunks.")
    click.echo(f"Done. Total chunks: {total}")

@cli.command()
@click.option("--q", required=True, help="Your question")
@click.option("--k", default=TOP_K, show_default=True, help="Top-K retrieved chunks")
def query(q, k):
    from rag.store import get_collection
    coll = get_collection(COLLECTION_NAME, CHROMA_DIR)
    emb = Embedder(EMBEDDING_MODEL)
    qv = emb.encode([q])[0]
    hits = top_k(coll, qv, k=k)
    click.echo("Retrieved chunks:")
    for i, h in enumerate(hits, 1):
        click.echo(f"[{i}] {h['source']}#{h['chunk']} score={h['score']:.3f}")
        click.echo(h["text"][:200].replace("\n"," ") + " ...")
        click.echo("-"*60)
    ans = template_answer(q, hits) if GENERATOR != "openai" else openai_answer(q, hits)
    click.echo("\n---\nAnswer:\n" + ans)

if __name__ == "__main__":
    cli()
