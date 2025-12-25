"""
Smart chunking for scientific documents.
Preserves tables, equations, and figures as atomic units.
"""
import re
from typing import Optional
from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    MarkdownNodeParser,
)
from llama_index.core.schema import TextNode

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP


class ScientificChunker:
    """
    Smart chunker for scientific content.
    - Regular text: Sentence-based chunking with overlap
    - Tables: Keep whole with surrounding context
    - Equations: Keep whole with surrounding context  
    - Figures: Keep whole with caption
    """
    
    def __init__(
        self, 
        chunk_size: int = CHUNK_SIZE, 
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Standard text splitter
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # Markdown-aware splitter
        self.markdown_splitter = MarkdownNodeParser()
        
        # Patterns for special content
        self.table_pattern = re.compile(r'\|[^\n]+\|\n(?:\|[-:]+\|)+\n(?:\|[^\n]+\|\n?)+', re.MULTILINE)
        self.equation_pattern = re.compile(r'\$\$[^$]+\$\$|\$[^$]+\$', re.MULTILINE)
        self.figure_pattern = re.compile(r'(?:Figure|Fig\.?)\s+\d+[.:][^\n]+(?:\n[^\n]+)?', re.IGNORECASE)
    
    def chunk_documents(self, documents: list[Document]) -> list[TextNode]:
        """
        Chunk documents into nodes, preserving special scientific content.
        
        Args:
            documents: List of Document objects from PDF parsing
            
        Returns:
            List of TextNode objects ready for embedding
        """
        all_nodes = []
        
        for doc in documents:
            nodes = self._chunk_single_document(doc)
            all_nodes.extend(nodes)
        
        print(f"Created {len(all_nodes)} chunks from {len(documents)} documents")
        return all_nodes
    
    def _chunk_single_document(self, doc: Document) -> list[TextNode]:
        """Process a single document into chunks."""
        text = doc.text
        metadata = doc.metadata.copy()
        
        nodes = []
        
        # Extract and preserve special content
        tables = self._extract_tables(text, metadata)
        equations = self._extract_equations(text, metadata)
        figures = self._extract_figures(text, metadata)
        
        nodes.extend(tables)
        nodes.extend(equations)
        nodes.extend(figures)
        
        # Remove special content from text before regular chunking
        clean_text = self._remove_special_content(text)
        
        # Chunk remaining text
        if clean_text.strip():
            text_doc = Document(text=clean_text, metadata=metadata)
            text_nodes = self.text_splitter.get_nodes_from_documents([text_doc])
            
            # Add content type to metadata
            for node in text_nodes:
                node.metadata["content_type"] = "text"
            
            nodes.extend(text_nodes)
        
        return nodes
    
    def _extract_tables(self, text: str, base_metadata: dict) -> list[TextNode]:
        """Extract tables as atomic chunks."""
        nodes = []
        
        for match in self.table_pattern.finditer(text):
            table_text = match.group()
            
            # Get surrounding context (2 lines before)
            start = match.start()
            context_start = text.rfind('\n', 0, max(0, start - 200))
            context = text[context_start:start].strip()
            
            node = TextNode(
                text=f"TABLE:\n{context}\n\n{table_text}",
                metadata={
                    **base_metadata,
                    "content_type": "table",
                    "context": context[:200],
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _extract_equations(self, text: str, base_metadata: dict) -> list[TextNode]:
        """Extract equations as atomic chunks with context."""
        nodes = []
        
        for match in self.equation_pattern.finditer(text):
            equation = match.group()
            
            # Get surrounding context
            start = max(0, match.start() - 150)
            end = min(len(text), match.end() + 150)
            
            # Find sentence boundaries
            context_before = text[start:match.start()]
            context_after = text[match.end():end]
            
            node = TextNode(
                text=f"EQUATION:\n{context_before.strip()}\n{equation}\n{context_after.strip()}",
                metadata={
                    **base_metadata,
                    "content_type": "equation",
                    "latex": equation,
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _extract_figures(self, text: str, base_metadata: dict) -> list[TextNode]:
        """Extract figure references and captions."""
        nodes = []
        
        for match in self.figure_pattern.finditer(text):
            figure_text = match.group()
            
            node = TextNode(
                text=f"FIGURE: {figure_text}",
                metadata={
                    **base_metadata,
                    "content_type": "figure",
                    "caption": figure_text,
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _remove_special_content(self, text: str) -> str:
        """Remove tables, equations, and figures from text for regular chunking."""
        # Remove tables
        text = self.table_pattern.sub('[TABLE REMOVED]', text)
        # Remove block equations (keep inline)
        text = re.sub(r'\$\$[^$]+\$\$', '[EQUATION REMOVED]', text)
        # Remove figure captions
        text = self.figure_pattern.sub('[FIGURE REMOVED]', text)
        
        return text
