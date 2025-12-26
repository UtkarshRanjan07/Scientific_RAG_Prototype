"""
Scientific RAG Chat Engine using LlamaIndex with FREE Groq API.
"""
from typing import Optional, List, Tuple
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.groq import Groq

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import LLM_MODEL, LLM_TEMPERATURE, SIMILARITY_TOP_K, GROQ_API_KEY


SYSTEM_PROMPT = """You are a scientific research assistant with expertise in analyzing research papers.

Your task is to answer questions based on the scientific papers in your knowledge base.

Guidelines:
1. **Always cite your sources** using the format [Source: filename, Page: X]
2. **For equations**, display them in LaTeX format using $$ delimiters
3. **For tables**, summarize the key insights and mention the source
4. **For figures**, describe what they show and their significance
5. **Be precise and scientific** in your language
6. **If you're unsure**, clearly state the limitations of your knowledge
7. **Provide context** - explain technical terms when appropriate

Remember: Only answer based on the provided context. If the information isn't in the documents, say so clearly."""


class ScientificChatEngine:
    """
    Chat engine for scientific document Q&A.
    Uses FREE Groq API with Llama 3.1 model.
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        llm_model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        similarity_top_k: int = SIMILARITY_TOP_K,
    ):
        self.index = index
        
        # Get API key
        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found! Get a FREE key at https://console.groq.com/keys"
            )
        
        # Initialize FREE Groq LLM
        self.llm = Groq(
            model=llm_model,
            api_key=api_key,
            temperature=temperature,
        )
        
        # Initialize memory
        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=4000
        )
        
        # Create chat engine
        self.chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=index.as_retriever(similarity_top_k=similarity_top_k),
            llm=self.llm,
            memory=self.memory,
            system_prompt=SYSTEM_PROMPT,
            verbose=True,
        )
    
    def chat(self, message: str) -> Tuple[str, List[dict], List[str]]:
        """
        Send a message and get a response with sources and images.
        
        Args:
            message: User message
            
        Returns:
            Tuple of (response text, list of source documents, list of image paths)
        """
        response = self.chat_engine.chat(message)
        
        # Extract sources and images
        sources, images = self._extract_sources_and_images(response, message)
        
        return str(response), sources, images

    def stream_chat(self, message: str):
        """
        Send a message and get a streaming response.
        
        Args:
            message: User message
            
        Returns:
            A generator that yields response chunks, and finally sources and images.
        """
        # GREETING OPTIMIZATION: Check for simple greetings
        greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "thanks", "thank you", "ok", "okay"]
        clean_msg = message.lower().strip().strip('?!. ')
        
        if clean_msg in greetings:
            # Create a mock streaming response
            class MockStreamingResponse:
                def __init__(self, text):
                    self.response = text
                    self.response_gen = (word + (" " if i < len(text.split())-1 else "") for i, word in enumerate(text.split()))
                    self.source_nodes = []
            
            reply = "Hello! How can I help you with your scientific papers today?"
            if clean_msg in ["thanks", "thank you"]:
                reply = "You're welcome! Let me know if you have more questions about the papers."
            
            return MockStreamingResponse(reply)

        response_gen = self.chat_engine.stream_chat(message)
        return response_gen

    def _extract_sources_and_images(self, response, message: str) -> Tuple[List[dict], List[str]]:
        """Helper to extract sources and images from response."""
        sources = []
        images = []
        import json
        
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                # Extract source info
                sources.append({
                    "source": node.node.metadata.get("source", "Unknown"),
                    "page": node.node.metadata.get("page_label") or node.node.metadata.get("page_num", "?"),
                    "content_type": node.node.metadata.get("content_type", "text"),
                    "score": round(node.score, 3) if node.score else None,
                    "text_preview": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                })
                
                # Extract images details from metadata - ONLY if user asks for them
                figure_keywords = ["figure", "image", "diagram", "plot", "chart", "graph", "visual", "show me", "picture"]
                should_extract_images = any(keyword in message.lower() for keyword in figure_keywords)
                
                if should_extract_images and "image_map" in node.node.metadata:
                    try:
                        image_map = json.loads(node.node.metadata["image_map"])
                        
                        current_page = None
                        if "page_label" in node.node.metadata:
                            try:
                                current_page = int(node.node.metadata["page_label"])
                            except: pass
                        elif "page_num" in node.node.metadata:
                            try:
                                current_page = int(node.node.metadata["page_num"])
                            except: pass
                            
                        if current_page:
                            pages_to_check = [str(current_page), str(current_page - 1), str(current_page + 1)]
                            for p in pages_to_check:
                                if p in image_map:
                                    images.extend(image_map[p])
                    except Exception:
                        pass
        
        # Deduplicate images
        unique_images = []
        for img in images:
            if img not in unique_images:
                unique_images.append(img)
                
        return sources, unique_images
    
    def reset(self):
        """Reset conversation history."""
        self.memory.reset()
        self.chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=self.index.as_retriever(similarity_top_k=SIMILARITY_TOP_K),
            llm=self.llm,
            memory=self.memory,
            system_prompt=SYSTEM_PROMPT,
            verbose=True,
        )
    
    def get_chat_history(self) -> List[dict]:
        """Get the current chat history."""
        history = []
        for msg in self.memory.get_all():
            history.append({
                "role": msg.role,
                "content": msg.content,
            })
        return history
