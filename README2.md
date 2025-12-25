# Scientific RAG Prototype - Architecture & Deep Dive

## 1. High-Level Architecture Diagram

```mermaid
graph TD
    subgraph "Data Ingestion Pipeline (One-Time)"
        A[Scientific GIFs] -->|LlamaParse| B(Markdown Text / Tables)
        A -->|PyMuPDF| C(Extracted Images)
        B --> D[Chunking]
        C --> D
        D -->|HuggingFace BGE| E[Vector Embeddings]
        E -->|Store| F[(ChromaDB)]
    end

    subgraph "Chat Application (Real-Time)"
        U[User] -->|Query| G[Streamlit UI]
        G -->|Query| H[Retriever]
        H -->|Search| F
        F -->|Top K Context| I[Context Window]
        I -->|Prompt| J[Groq LLM (Llama 3)]
        J -->|Response| G
        H -->|Image Metadata| K[Figure Logic]
        K -->|Display Images| G
    end
```

## 2. Project Overview

This project is a **Scientific Retrieval-Augmented Generation (RAG)** system designed to ingest complex scientific PDF documents and allow users to ask questions about them.

### What we are doing:
1.  **Parsing with Precision:** We use specialized tools to handle the complex structure of scientific papers (tables, equations, double-column layouts), which standard text extractors often fail at.
2.  **Multimodal-ish Context:** We extract not just text, but also tables (as Markdown) and Figures (as images linked to the text).
3.  **Cost-Free Operation:** We realized the system using entirely **FREE** components for its ongoing operation (Local Embeddings + Groq Free Tier).
4.  **Local Storage:** All data is stored locally in a vector database (ChromaDB), ensuring privacy and persistence.

---

## 3. Technology Stack & Addons

We utilize a "Best-in-Class" stack prioritizing performance and zero cost.

### Core Frameworks
*   **LlamaIndex:** The extensive data framework used to orchestrate ingestion, indexing, and retrieval.
*   **Streamlit:** The web framework used to build the interactive chat interface.

### Tools & Libraries (The "Why")

| Tool / Library | Purpose | Why we chose it |
| :--- | :--- | :--- |
| **LlamaParse** | **PDF Parser** | Unlike PyPDF2, LlamaParse uses a vision model to correctly understand layouts, **Math Blocks**, and **Tables**, converting them to clean Markdown. |
| **PyMuPDF (fitz)** | **Image Extraction** | LlamaParse extracts text, but PyMuPDF allows us to rip the actual image bytes from the PDF pages to display in the UI. |
| **HuggingFace** | **Embeddings** | We use the `BAAI/bge-small-en-v1.5` model locally. It is state-of-the-art for its size and completely **FREE** (vs OpenAI's paid embeddings). |
| **ChromaDB** | **Vector Database** | A light-weight, open-source vector store that saves data to the local disk (`./chroma_db`), allowing persistent storage without a server. |
| **Groq (Llama 3)** | **LLM Inference** | Groq provides an incredibly fast inference engine. We use `llama-3.1-8b-instant` which is **FREE** and much faster than GPT-3.5. |
| **Pillow (PIL)** | **Image Processing** | Used in the UI to filter out "junk" images (tiny icons, headers) so users only see relevant figures. |

---

## 4. File-by-File Script Explanation

Here is exactly what every script in the repository does:

### 📂 Root Directory
*   **`config.py`**: The "Control Center". It defines:
    *   Paths (`data/`, `extracted/`, `chroma_db/`).
    *   Model choices (`BAAI/bge-small`, `llama-3.1`).
    *   API keys loading.
*   **`app.py`**: The **Application Entry Point**.
    *   Sets up the Streamlit UI.
    *   Handles User Input.
    *   **Logic:** It calls the chat engine, receives the text response + image paths, runs the image filters (size/aspect ratio), and renders the result.
*   **`scripts/ingest.py`**: The **setup script**. You run this **once**.
    *   It orchestrates the flow: PDF -> Parser -> Chunker -> Embedder -> Database.
    *   It's designed to be idempotent (clears DB before running).

### 📂 `src/extraction`
*   **`pdf_parser.py`**:
    *   **Class `PDFParser`**: Wraps LlamaParse.
    *   **Method `_extract_images`**: Uses PyMuPDF to save images to `extracted/figures/`.
    *   **Method `parse_single_pdf`**: Combines the text from LlamaParse with the image paths, adding `page_num` metadata so the chatbot knows which image belongs to which page.

### 📂 `src/processing`
*   **`embedder.py`**:
    *   Configures the local HuggingFace embedding model.
    *   We switched this from OpenAI to avoid rate limits and costs.
*   **`chunker.py`**:
    *   Defines how we split the text. We use `SentenceSplitter` configured in `config.py` to ensure we don't break sentences in the middle.

### 📂 `src/storage`
*   **`vector_store.py`**:
    *   Manages the ChromaDB client.
    *   Handles saving and loading the index from the `./chroma_db` folder.

### 📂 `src/chatbot`
*   **`engine.py`**: The **Brain**.
    *   Initializes the `CondensePlusContextChatEngine` (which remembers chat history).
    *   **Method `chat`**:
        1.  Sends query to LLM.
        2.  LLM generates answer using retrieved text.
        3.  **Figure Logic**: It checks if the user asked for "figures". If yes, it looks at the metadata of the retrieved text chunks, finds the `page_num`, and looks up the corresponding images from that page (and adjacent pages) to return to the UI.

---

## 5. Future Improvements

If we wanted to take this project to the "Production/Enterprise" level, here is what we would add:

1.  **Hybrid Search (Sparse + Dense):**
    *   Currently, we use Semantic Search (Vector). This is great for concepts but bad for exact keywords (e.g., "Patient ID 123").
    *   *Addon:* **BM25** (Keyword search) combined with Vector search (using `Reciprocal Rank Fusion`).

2.  **Re-Ranking:**
    *   Sometimes the top 5 results are relevant but not the *most* relevant.
    *   *Addon:* **Cohere ReRerank**. It takes the top 20 results and sorts them intelligently before sending to the LLM. This drastically improves accuracy.

3.  **True Multimodal Embeddings (CLIP):**
    *   Currently, we find images based on the *text on the page*.
    *   *Improvement:* Embed the **images themselves** using a model like **CLIP**. This would allow you to search "Show me the graph with the steep curve" and the AI would look at the *pixels* of the image, not just the text next to it.

4.  **Auto-Updating Knowledge Base:**
    *   Add a "Watchdog" script that detects when you drop a new PDF into `data/` and automatically ingests just that file, without reprocessing everything.

5.  **Citation Linking:**
    *   Make the `[Source: pdf1, Page 2]` citations in the chat clickable, so clicking them opens the PDF to that specific page in the browser.
