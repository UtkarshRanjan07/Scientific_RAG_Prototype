"""
ChromaDB Vector Store Manager for Scientific RAG.
"""
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


class VectorStoreManager:
    """
    Manages ChromaDB vector store for scientific document embeddings.
    """
    
    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_dir = persist_dir or CHROMA_DIR
        self.collection_name = collection_name
        
        # Initialize embedding model for retrieval
        self.embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            trust_remote_code=True,
        )
        
        # Initialize ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Create ChromaVectorStore
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.collection
        )
        
        # Storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
    
    def add_nodes(self, nodes: list[TextNode]) -> VectorStoreIndex:
        """
        Add nodes with embeddings to the vector store.
        
        Args:
            nodes: List of TextNode objects with embeddings
            
        Returns:
            VectorStoreIndex for retrieval
        """
        # Create index and add nodes
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
        )
        
        print(f"Added {len(nodes)} nodes to ChromaDB collection '{self.collection_name}'")
        return index
    
    def get_index(self) -> VectorStoreIndex:
        """
        Get existing VectorStoreIndex from ChromaDB.
        
        Returns:
            VectorStoreIndex connected to ChromaDB
        """
        return VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_dir": str(self.persist_dir),
        }
    
    def reset_collection(self) -> None:
        """Delete and recreate the collection."""
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.collection
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        print(f"Reset collection '{self.collection_name}'")
