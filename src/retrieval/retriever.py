"""
Scientific document retriever with hybrid search and re-ranking.
"""
from typing import Optional, List
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SimilarityPostprocessor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SIMILARITY_TOP_K, RERANK_TOP_N


class ScientificRetriever:
    """
    Retriever optimized for scientific documents.
    Uses semantic search with optional content-type filtering and re-ranking.
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = SIMILARITY_TOP_K,
        rerank_top_n: int = RERANK_TOP_N,
    ):
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.rerank_top_n = rerank_top_n
        
        # Base retriever
        self.retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
        )
        
        # Similarity postprocessor (filter low quality results)
        self.similarity_filter = SimilarityPostprocessor(
            similarity_cutoff=0.5
        )
    
    def retrieve(
        self,
        query: str,
        content_types: Optional[List[str]] = None,
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes for a query.
        
        Args:
            query: User query string
            content_types: Optional filter for content types 
                          (e.g., ["text", "table", "equation"])
        
        Returns:
            List of NodeWithScore objects
        """
        # Get initial results
        nodes = self.retriever.retrieve(query)
        
        # Apply similarity filter
        nodes = self.similarity_filter.postprocess_nodes(nodes)
        
        # Filter by content type if specified
        if content_types:
            nodes = [
                node for node in nodes
                if node.node.metadata.get("content_type", "text") in content_types
            ]
        
        return nodes[:self.rerank_top_n]
    
    def retrieve_with_context(
        self,
        query: str,
    ) -> dict:
        """
        Retrieve nodes with additional context information.
        
        Returns:
            Dict with nodes grouped by content type and source
        """
        nodes = self.retrieve(query)
        
        # Group by content type
        grouped = {
            "text": [],
            "table": [],
            "equation": [],
            "figure": [],
        }
        
        sources = set()
        
        for node in nodes:
            content_type = node.node.metadata.get("content_type", "text")
            grouped[content_type].append(node)
            sources.add(node.node.metadata.get("source", "unknown"))
        
        return {
            "nodes": nodes,
            "grouped": grouped,
            "sources": list(sources),
            "total_count": len(nodes),
        }
    
    def format_context_for_llm(self, nodes: List[NodeWithScore]) -> str:
        """
        Format retrieved nodes as context for LLM generation.
        
        Args:
            nodes: List of retrieved nodes
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, node in enumerate(nodes, 1):
            source = node.node.metadata.get("source", "Unknown")
            content_type = node.node.metadata.get("content_type", "text")
            page = node.node.metadata.get("page_num", "?")
            
            header = f"[Source {i}: {source}, Page {page}, Type: {content_type}]"
            context_parts.append(f"{header}\n{node.node.text}\n")
        
        return "\n---\n".join(context_parts)
