"""
Multi-modal embedding for scientific content.
Uses free HuggingFace local embeddings.
"""
from typing import Optional
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import EMBEDDING_MODEL


class MultiModalEmbedder:
    """
    Embedding handler using free HuggingFace local embeddings.
    No API costs - runs entirely on your machine.
    """
    
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {embedding_model}")
        print("(This may take a minute on first run as the model downloads...)")
        
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            trust_remote_code=True,
        )
        print("Embedding model loaded!")
    
    def embed_nodes(self, nodes: list[TextNode]) -> list[TextNode]:
        """
        Embed all nodes using local HuggingFace model.
        
        Args:
            nodes: List of TextNode objects to embed
            
        Returns:
            List of TextNode objects with embeddings
        """
        print(f"Embedding {len(nodes)} nodes...")
        
        for i, node in enumerate(nodes):
            if i % 50 == 0:
                print(f"  Embedded {i}/{len(nodes)} nodes...")
            
            # Get embedding for the node text
            embedding = self.embed_model.get_text_embedding(node.text)
            node.embedding = embedding
        
        print(f"Completed embedding {len(nodes)} nodes!")
        return nodes
    
    def get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query string."""
        return self.embed_model.get_text_embedding(query)
    
    def get_embed_model(self):
        """Return the embedding model for use with LlamaIndex."""
        return self.embed_model
