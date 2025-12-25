# Scientific RAG Prototype

A Retrieval-Augmented Generation (RAG) system for scientific PDF documents using LlamaIndex and ChromaDB.

## Features

- 📄 **PDF Extraction**: Uses LlamaParse for text, tables, equations, and figures
- 🧩 **Smart Chunking**: Preserves tables and equations as atomic units
- 🔢 **Multi-Modal Embedding**: Semantic enrichment for tables and equations
- 💾 **ChromaDB Storage**: Persistent vector storage
- 🔍 **Semantic Search**: Similarity-based retrieval with filtering
- 💬 **Streamlit Chat**: Interactive Q&A interface with citations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Ingest Documents

Place your PDFs in the `data/` folder, then run:

```bash
python3 scripts/ingest.py
```

### 4. Run the Chat App

```bash
streamlit run app.py
```

## Project Structure

```
Scientific_RAG_Prototype/
├── data/                    # Your PDF documents
├── chroma_db/               # Vector database (auto-created)
├── extracted/               # Cached extractions
├── src/
│   ├── extraction/          # PDF parsing
│   ├── processing/          # Chunking & embedding
│   ├── storage/             # ChromaDB management
│   ├── retrieval/           # Search logic
│   └── chatbot/             # Chat engine
├── scripts/
│   └── ingest.py            # Ingestion pipeline
├── app.py                   # Streamlit UI
├── config.py                # Configuration
└── requirements.txt
```

## Architecture

```
PDF Documents → LlamaParse → Smart Chunking → Embedding → ChromaDB
                                                              ↓
User Query → Semantic Search → Re-ranking → LLM → Response + Citations
```

## Configuration

Edit `config.py` to customize:

- `CHUNK_SIZE`: Text chunk size (default: 512)
- `SIMILARITY_TOP_K`: Number of results to retrieve (default: 10)
- `LLM_MODEL`: GPT model for generation (default: gpt-4-turbo-preview)
