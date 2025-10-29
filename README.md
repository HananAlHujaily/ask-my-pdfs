📄 Ask My PDFs — Minimal RAG over Local Documents

This repository lets you query your own PDFs using a lightweight Retrieval-Augmented Generation (RAG) pipeline:
chunk → embed → store → retrieve → generate answers.

🚀 Features

🧠 Embeddings via sentence-transformers/all-MiniLM-L6-v2 (default, easily switchable)

💾 Vector store: ChromaDB
 (local & persistent by default)

🖥️ UI: Streamlit
 — fast, interactive, and simple

🧰 CLI support for ingestion and querying from the terminal

📑 PDF parsing with pypdf

🤖 LLM integration with OpenAI or (optionally) local models such as Ollama (coming soon)

⚙️ Quickstart
# 1. Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment template and configure
cp .env.example .env

# 4. Add your PDFs to ./docs  (or select any folder in the Streamlit UI)

# 5. Launch the app
streamlit run app.py

🧩 CLI Usage 
# Ingest a folder of PDFs into the vector store
python cli.py ingest --path ./docs

# Ask a question (retrieval-only mode)
python cli.py query --q "What are the main contributions?" --k 4

⚙️ Environment Configuration (.env)

Tune parameters easily in your .env file:

Variable	Description	Example
CHROMA_DIR	Path to vector store	./chroma_store
EMBEDDING_MODEL	Embedding model name	sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE	Chunk length in characters	900
CHUNK_OVERLAP	Overlap between chunks	120
TOP_K	Number of results retrieved	4
GENERATOR openai	or none
OPENAI_API_KEY	Your OpenAI API key	sk-XXXX...
COLLECTION_NAME	ChromaDB collection name	pdfs
🗂️ Project Structure
ask-my-pdfs/
├── app.py                 # Streamlit UI (folder picker, retrieval & generation)
├── cli.py                 # Command line ingestion & querying
├── rag/
│   ├── pdf_loader.py      # PDF extraction via pypdf
│   ├── chunker.py         # Text chunking logic
│   ├── embedder.py        # SentenceTransformers wrapper
│   ├── store.py           # ChromaDB persistence layer
│   ├── retriever.py       # Top-k similarity search
│   └── generator.py       # Template & OpenAI generation logic
├── eval/
│   └── eval_basics.py     # Retrieval sanity checks
├── docs/                  # Place your PDFs here
├── chroma_store/          # Auto-created local vector store
├── .env           # Configuration template
├── requirements.txt       # Dependencies
└── README.md

🧾 Requirements

The refined and fully compatible setup:

streamlit==1.37.0
chromadb==0.5.5
sentence-transformers==2.7.0
python-dotenv==1.0.1
openai>=1.0,<3
tiktoken==0.7.0
pypdf==4.2.0
click==8.1.7
httpx<0.28


This ensures compatibility with OpenAI SDK ≥1.0 and avoids proxies keyword conflicts in httpx 0.28+.

🧠 Notes

Works offline in retrieval-only mode (GENERATOR=none).

To enable OpenAI generation, set:

GENERATOR=openai
OPENAI_API_KEY=sk-...


For scanned PDFs, run OCR first (e.g., Tesseract
).

Clear Streamlit cache (streamlit cache clear) if structure or .env changes.

🧩 Coming Next

🦙 Local LLM support (Ollama, LM Studio) — offline generation mode

⚖️ RAG evaluation scripts (recall@k, context faithfulness)

🧹 “Clear Index” button directly in UI

📊 Retrieval diagnostics and statistics
