"""
Configuration for Scientific RAG Prototype
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Paths ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
EXTRACTED_DIR = PROJECT_ROOT / "extracted"

# Create directories if they don't exist
CHROMA_DIR.mkdir(exist_ok=True)
EXTRACTED_DIR.mkdir(exist_ok=True)
FIGURES_DIR = PROJECT_ROOT / "extracted" / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# === API Keys ===
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY", "llx-uEJxq0UvkccZJeCuldZXrAA4DsvKOoe5cVjey3rRRRSeZedq")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Model Settings ===
# Using free HuggingFace embedding model (runs locally, no API costs)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Free, local, 384 dimensions
EMBEDDING_DIMENSIONS = 384
USE_LOCAL_EMBEDDINGS = True  # Set to False to use OpenAI embeddings

# LLM Settings - Using FREE Groq API with Llama 3.1
LLM_PROVIDER = "groq"  # Options: "groq" (free), "openai" (paid)
LLM_MODEL = "llama-3.1-8b-instant"  # Free Groq model
LLM_TEMPERATURE = 0.1

# === Chunking Settings ===
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# === Retrieval Settings ===
SIMILARITY_TOP_K = 5
RERANK_TOP_N = 3

# === ChromaDB Collections ===
COLLECTION_NAME = "scientific_papers"
