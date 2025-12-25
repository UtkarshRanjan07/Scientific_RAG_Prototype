#!/usr/bin/env python3
"""
Ingestion script for Scientific RAG Prototype.
Processes all PDFs and stores them in ChromaDB.
Uses free local HuggingFace embeddings.
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from config import DATA_DIR
from src.extraction.pdf_parser import PDFParser, PyMuPDFParser
from src.processing.chunker import ScientificChunker
from src.processing.embedder import MultiModalEmbedder
from src.storage.vector_store import VectorStoreManager


def main():
    """Run the full ingestion pipeline."""
    
    print("=" * 60)
    print("Scientific RAG - Document Ingestion Pipeline")
    print("Using FREE local HuggingFace embeddings (no API costs)")
    print("=" * 60)
    
    # Step 1: Parse PDFs
    print("\n[1/4] Parsing PDF documents...")
    try:
        parser = PDFParser()
        documents = parser.parse_all_pdfs(DATA_DIR)
    except Exception as e:
        print(f"LlamaParse failed: {e}")
        print("Falling back to PyMuPDF...")
        parser = PyMuPDFParser()
        documents = parser.parse_all_pdfs(DATA_DIR)
    
    if not documents:
        print("No documents extracted! Check your PDF files.")
        sys.exit(1)
    
    print(f"   Extracted {len(documents)} document sections")
    
    # Step 2: Chunk documents
    print("\n[2/4] Chunking documents...")
    chunker = ScientificChunker()
    nodes = chunker.chunk_documents(documents)
    print(f"   Created {len(nodes)} chunks")
    
    # Count by type
    type_counts = {}
    for node in nodes:
        ct = node.metadata.get("content_type", "text")
        type_counts[ct] = type_counts.get(ct, 0) + 1
    print(f"   Content types: {type_counts}")
    
    # Step 3: Generate embeddings (FREE - runs locally!)
    print("\n[3/4] Generating embeddings (free local model)...")
    embedder = MultiModalEmbedder()
    embedded_nodes = embedder.embed_nodes(nodes)
    print(f"   Generated embeddings for {len(embedded_nodes)} nodes")
    
    # Step 4: Store in ChromaDB
    print("\n[4/4] Storing in ChromaDB...")
    store = VectorStoreManager()
    store.reset_collection()  # Start fresh
    index = store.add_nodes(embedded_nodes)
    
    stats = store.get_collection_stats()
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Total documents: {stats['document_count']}")
    
    print("\n" + "=" * 60)
    print("✓ Ingestion complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: streamlit run app.py")
    print("  2. Start asking questions about your papers!")


if __name__ == "__main__":
    main()
