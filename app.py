import os
import time
import streamlit as st
from dotenv import load_dotenv

from rag.pdf_loader import load_folder
from rag.chunker import chunk_text
from rag.embedder import Embedder
from rag.store import get_collection, add_chunks
from rag.retriever import top_k
from rag.generator import template_answer, openai_answer

# Load environment variables
load_dotenv()

# Config values from .env
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdfs")
GENERATOR = os.getenv("GENERATOR", "none").lower()

# Streamlit setup
st.set_page_config(page_title="Ask My PDFs", page_icon="📄", layout="wide")
st.title("📄 Ask My PDFs — RAG over your documents")

# Sidebar
with st.sidebar:
    st.header("Settings")
    folder = st.text_input(
        "Folder with PDFs",
        value="docs",
        help="Type/paste a local folder path containing PDF files."
    )
    k_value = st.slider("Top-K", 1, 10, TOP_K)  # ✅ renamed variable
    st.write(f"Embedding model: `{EMBEDDING_MODEL}`")
    st.write(f"Generator mode: `{GENERATOR}`")


# -----------------------------
# Helper functions (cached)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_docs_cached(path: str):
    """Load all PDFs from a folder (cached for performance)."""
    return load_folder(path)


@st.cache_resource(show_spinner=True)
def build_or_update_index(path: str):
    """Ingest PDFs, chunk, embed, and store in ChromaDB."""
    docs = load_docs_cached(path)
    if not docs:
        return None, 0

    coll = get_collection(COLLECTION_NAME, CHROMA_DIR)
    emb = Embedder(EMBEDDING_MODEL)
    total_chunks = 0

    for name, text in docs.items():
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue
        vecs = emb.encode(chunks)
        add_chunks(coll, chunks, [name] * len(chunks), list(range(len(chunks))), vecs)
        total_chunks += len(chunks)

    return coll, total_chunks


# -----------------------------
# Index PDFs
# -----------------------------
if st.button("Index / Re-index PDFs"):
    with st.spinner("Indexing..."):
        coll, total = build_or_update_index(folder)
    if not total:
        st.warning("No PDFs found or failed to extract text.")
    else:
        st.success(f"Indexed {total} chunks from PDFs in: {folder}")


# -----------------------------
# Ask a Question
# -----------------------------
query = st.text_input("Ask a question")
run = st.button("Retrieve & (optionally) Generate")

if run and query.strip():
    coll = get_collection(COLLECTION_NAME, CHROMA_DIR)
    emb = Embedder(EMBEDDING_MODEL)

    # Compute embedding & search
    t0 = time.time()
    qv = emb.encode([query.strip()])[0]
    hits = top_k(coll, qv, k=k_value)  # ✅ fixed variable name
    dt = time.time() - t0

    col1, col2 = st.columns([1, 1])

    # Retrieved chunks
    with col1:
        st.subheader("Retrieved Context")
        if not hits:
            st.info("No results found. Did you index your PDFs?")
        for i, h in enumerate(hits, 1):
            with st.expander(
                f"[{i}] {h['source']} (chunk {h['chunk']}) • score={h['score']:.3f}",
                expanded=(i == 1),
            ):
                st.write(h["text"])

    # Generated answer
    with col2:
        st.subheader("Answer")
        ans = template_answer(query, hits) if GENERATOR != "openai" else openai_answer(query, hits)
        st.write(ans)
        st.caption(f"Top-K = {k_value} • Retrieval time = {dt:.2f}s")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "💡 **Tips:** Tune `CHUNK_SIZE` / `CHUNK_OVERLAP` in `.env` • "
    "Use OCR for scanned PDFs • Clear Streamlit cache if structure changes."
)
