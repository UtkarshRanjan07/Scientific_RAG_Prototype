#!/usr/bin/env python3
"""
Streamlit Chat App for Scientific RAG Prototype.
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from config import GROQ_API_KEY
from src.storage.vector_store import VectorStoreManager
from src.chatbot.engine import ScientificChatEngine

# Set environment variable
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Page configuration
st.set_page_config(
    page_title="Scientific RAG Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS with IMPROVED CONTRAST
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #a0a0a0;
        margin-bottom: 2rem;
    }
    
    /* Chat messages - IMPROVED CONTRAST */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Make ALL text white/light for readability */
    .stChatMessage p, 
    .stChatMessage li,
    .stChatMessage span,
    .stChatMessage div {
        color: #ffffff !important;
    }
    
    /* User message styling */
    [data-testid="stChatMessageContent-user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    [data-testid="stChatMessageContent-user"] p {
        color: #ffffff !important;
    }
    
    /* Assistant message styling - HIGH CONTRAST */
    [data-testid="stChatMessageContent-assistant"] {
        background: rgba(40, 40, 60, 0.9) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
    }
    
    [data-testid="stChatMessageContent-assistant"] p,
    [data-testid="stChatMessageContent-assistant"] li,
    [data-testid="stChatMessageContent-assistant"] span {
        color: #f0f0f0 !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    
    /* Markdown text in chat */
    .stMarkdown p {
        color: #e8e8e8 !important;
    }
    
    /* Source cards */
    .source-card {
        background: rgba(102, 126, 234, 0.15) !important;
        border: 1px solid rgba(102, 126, 234, 0.4) !important;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #ffffff !important;
    }
    
    .source-card strong {
        color: #a8b4ff !important;
    }
    
    .source-card small {
        color: #c0c0c0 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e0e0e0 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #667eea !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.2) !important;
        color: #ffffff !important;
    }
    
    /* Chat input - CORRECTED STYLING */
    [data-testid="stChatInput"] {
        background-color: transparent !important;
        /* Let Streamlit handle position, just add padding for aesthetics */
        padding-bottom: 2rem !important;
    }
    
    /* The actual input container */
    [data-testid="stChatInput"] > div {
        background-color: #ffffff !important; /* White background */
        border: 1px solid #667eea !important;
        border-radius: 20px !important;
        color: #000000 !important; /* Force black text */
    }
    
    /* The textarea input itself */
    [data-testid="stChatInput"] textarea {
        color: #000000 !important; /* Force black text */
        caret-color: #000000 !important;
        background-color: transparent !important;
        /* Ensure font is visible */
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Placeholder text */
    [data-testid="stChatInput"] textarea::placeholder {
        color: #666666 !important;
        -webkit-text-fill-color: #666666 !important;
    }
    
    /* Send button */
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
    }
    
    [data-testid="stChatInput"] button:hover {
        opacity: 0.9 !important;
        transform: scale(1.05) !important;
    }
    
    /* Add bottom padding to main content so it doesn't get hidden behind fixed input */
    .stApp > div > div > div > div {
        padding-bottom: 100px !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Error/Info boxes */
    .stAlert {
        background: rgba(255, 100, 100, 0.2) !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chat_engine():
    """Load the chat engine (cached)."""
    try:
        store = VectorStoreManager()
        stats = store.get_collection_stats()
        
        if stats["document_count"] == 0:
            return None, "No documents found. Please run the ingestion script first."
        
        index = store.get_index()
        engine = ScientificChatEngine(index)
        return engine, None
        
    except Exception as e:
        return None, f"Error loading chat engine: {str(e)}"


def render_sidebar():
    """Render the sidebar with info and controls."""
    with st.sidebar:
        st.markdown("## 🔬 Scientific RAG")
        st.markdown("---")
        
        # Load stats
        try:
            store = VectorStoreManager()
            stats = store.get_collection_stats()
            
            st.markdown("### 📊 Knowledge Base")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats["document_count"])
            with col2:
                st.metric("Collection", "Active" if stats["document_count"] > 0 else "Empty")
                
        except Exception:
            st.warning("Could not load stats")
        
        st.markdown("---")
        
        # Controls
        st.markdown("### ⚙️ Controls")
        
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if "chat_engine" in st.session_state:
                st.session_state.chat_engine.reset()
            st.rerun()
        
        st.markdown("---")
        
        # Tips
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Ask about specific topics in your papers
        - Request summaries of methods or results
        - Ask about equations and their meanings
        - Query specific tables for data
        """)
        
        st.markdown("---")
        st.markdown("### ⚡ Powered by")
        st.markdown("**Groq** (Llama 3.1) + **ChromaDB**")


def render_sources(sources):
    """Render source citations."""
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(f"""
            <div class="source-card">
                <strong>Source {i}:</strong> {source['source']}<br>
                <strong>Page:</strong> {source['page']} | 
                <strong>Type:</strong> {source['content_type']} |
                <strong>Score:</strong> {source['score']}<br>
                <small>{source['text_preview']}</small>
            </div>
            """, unsafe_allow_html=True)


def render_images(images):
    """Render extracted figures with filtering for artifacts."""
    if not images:
        return
    
    # Import PIL for image analysis
    from PIL import Image
    import hashlib
    
    st.markdown("### 🖼️ Related Figures")
    
    # Filter and deduplicate images
    valid_images = []
    seen_hashes = set()
    
    for img_path in images:
        if not os.path.exists(img_path):
            continue
            
        try:
            with Image.open(img_path) as img:
                # 1. Filter by size (skip very small icons/logos)
                width, height = img.size
                if width < 150 or height < 150:
                    continue
                    
                # 2. Filter by aspect ratio (skip extreme headers/footers)
                aspect_ratio = width / height
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    continue
                
                # 3. Deduplicate by content hash
                # (to remove repeated logos like "Springer Nature" appearing on every page)
                with open(img_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    
                if file_hash in seen_hashes:
                    continue
                
                seen_hashes.add(file_hash)
                valid_images.append(img_path)
                
        except Exception:
            continue
    
    if not valid_images:
        # If all images were filtered out, show nothing (or maybe a message?)
        # Better to show nothing than garbage.
        return

    # Display in columns (max 3 per row)
    cols = st.columns(3)
    for i, img_path in enumerate(valid_images):
        with cols[i % 3]:
            st.image(img_path, use_container_width=True)


def main():
    """Main app function."""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Scientific Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your scientific papers</p>', unsafe_allow_html=True)
    
    # Sidebar
    render_sidebar()
    
    # Load chat engine
    engine, error = load_chat_engine()
    
    if error:
        st.error(error)
        st.info("""
        ### Getting Started
        
        1. Make sure you have set your API keys in the `.env` file
        2. Run the ingestion script:
           ```bash
           python scripts/ingest.py
           ```
        3. Restart this app
        """)
        return
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = engine
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message and message["images"]:
                render_images(message["images"])
            if "sources" in message and message["sources"]:
                render_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your scientific papers..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            try:
                # Get the streaming response generator
                # Note: stream_chat returns the raw LlamaIndex streaming response
                response_gen = st.session_state.chat_engine.stream_chat(prompt)
                
                # Use st.write_stream for the typing effect
                # response_gen.response_gen is the generator for the words
                full_response = st.write_stream(response_gen.response_gen)
                
                # After streaming is done, the response object has the sources
                # We extract them using the helper we added to the engine
                sources, images = st.session_state.chat_engine._extract_sources_and_images(response_gen, prompt)
                
                if images:
                    render_images(images)
                    
                render_sources(sources)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                    "images": images
                })
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "images": []
                })


if __name__ == "__main__":
    main()
