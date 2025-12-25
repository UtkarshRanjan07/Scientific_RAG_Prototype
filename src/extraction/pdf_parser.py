"""
PDF Parser using LlamaParse for scientific document extraction.
Handles text, tables, equations, and figures.
"""
import os
import json
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from llama_parse import LlamaParse
from llama_index.core import Document

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import LLAMA_PARSE_API_KEY, DATA_DIR, EXTRACTED_DIR, FIGURES_DIR


class PDFParser:
    """
    Parser for scientific PDFs using LlamaParse.
    Extracts text, tables, equations, and figures with rich metadata.
    Includes image extraction via PyMuPDF.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or LLAMA_PARSE_API_KEY
        
        # Initialize LlamaParse with scientific document settings
        self.parser = LlamaParse(
            api_key=self.api_key,
            result_type="markdown",
            parsing_instruction="""
            You are parsing a scientific research paper. Please:
            1. Extract ALL text content preserving section structure (Abstract, Introduction, Methods, Results, Discussion, Conclusion, References)
            2. Convert ALL tables to proper markdown table format with captions
            3. Convert ALL mathematical equations to LaTeX format wrapped in $$ delimiters
            4. For figures, extract the caption and describe what the figure shows
            5. Preserve all citations and references
            6. Mark section headers clearly with markdown headers (# for main sections, ## for subsections)
            """,
            split_by_page=True,
            verbose=True
        )

    def _extract_images(self, pdf_path: Path) -> dict:
        """
        Extract images from PDF and return mapping of page_num -> list[image_paths].
        Uses PyMuPDF (fitz) to extract actual image data.
        """
        import fitz  # PyMuPDF
        
        image_map = {}
        try:
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc):
                image_list = page.get_images()
                
                if image_list:
                    page_images = []
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        try:
                            # Extract image bytes
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            ext = base_image["ext"]
                            
                            # Skip if image is too small (likely an icon or artifact)
                            if len(image_bytes) < 1000:  # < 1KB
                                continue
                                
                            # Create filename: {pdf_stem}_p{page}_img{index}.{ext}
                            image_filename = f"{pdf_path.stem}_p{page_num+1}_i{img_index+1}.{ext}"
                            image_path = FIGURES_DIR / image_filename
                            
                            # Save image if it doesn't exist
                            if not image_path.exists():
                                with open(image_path, "wb") as f:
                                    f.write(image_bytes)
                                
                            # Store RELATIVE path for usage in web app
                            # We'll serve FIGURES_DIR as static files
                            rel_path = f"extracted/figures/{image_filename}"
                            page_images.append(rel_path)
                            
                        except Exception as e:
                            continue
                    
                    if page_images:
                        image_map[page_num + 1] = page_images
                        
            doc.close()
        except Exception as e:
            print(f"Error extracting images from {pdf_path.name}: {e}")
            
        return image_map
    
    def parse_single_pdf(self, pdf_path: Path) -> list[Document]:
        """
        Parse a single PDF file and return LlamaIndex documents.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects with extracted content
        """
        try:
            # 1. Parse text with LlamaParse
            documents = self.parser.load_data(str(pdf_path))
            
            # 2. Extract images with PyMuPDF
            image_map = self._extract_images(pdf_path)
            
            # Enrich documents with metadata
            for i, doc in enumerate(documents):
                # Manually assign page number (1-based index)
                page_num = str(i + 1)
                
                doc.metadata.update({
                    "source": pdf_path.name,
                    "file_path": str(pdf_path),
                    "doc_id": pdf_path.stem,
                    "page_label": page_num, # Ensure page_label is present
                    "page_num": page_num,
                    # Store image map as JSON string
                    "image_map": json.dumps(image_map)
                })
            
            return documents
            
        except Exception as e:
            print(f"Error parsing {pdf_path.name}: {e}")
            return []
    
    def parse_all_pdfs(self, pdf_dir: Optional[Path] = None) -> list[Document]:
        """
        Parse all PDFs in the data directory.
        
        Args:
            pdf_dir: Directory containing PDFs (defaults to DATA_DIR)
            
        Returns:
            List of all Document objects
        """
        pdf_dir = pdf_dir or DATA_DIR
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        all_documents = []
        
        for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
            docs = self.parse_single_pdf(pdf_path)
            all_documents.extend(docs)
            
            # Save extracted content for debugging
            self._save_extracted_content(pdf_path, docs)
        
        print(f"Successfully extracted {len(all_documents)} documents")
        return all_documents
    
    def _save_extracted_content(self, pdf_path: Path, documents: list[Document]):
        """Save extracted content to disk for debugging/caching."""
        output_path = EXTRACTED_DIR / f"{pdf_path.stem}.json"
        
        content = {
            "source": pdf_path.name,
            "documents": [
                {
                    "text": doc.text[:1000] + "..." if len(doc.text) > 1000 else doc.text,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
        }
        
        with open(output_path, "w") as f:
            json.dump(content, f, indent=2)


# Fallback parser using PyMuPDF (if LlamaParse fails or for quick testing)
class PyMuPDFParser:
    """Fallback PDF parser using PyMuPDF for basic text extraction."""
    
    def parse_single_pdf(self, pdf_path: Path) -> list[Document]:
        """Extract text from PDF using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(str(pdf_path))
            documents = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                
                if text.strip():
                    documents.append(Document(
                        text=text,
                        metadata={
                            "source": pdf_path.name,
                            "page_num": page_num + 1,
                            "doc_id": pdf_path.stem,
                        }
                    ))
            
            doc.close()
            return documents
            
        except Exception as e:
            print(f"Error with PyMuPDF on {pdf_path.name}: {e}")
            return []
    
    def parse_all_pdfs(self, pdf_dir: Optional[Path] = None) -> list[Document]:
        """Parse all PDFs using PyMuPDF."""
        pdf_dir = pdf_dir or DATA_DIR
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        all_documents = []
        for pdf_path in tqdm(pdf_files, desc="Parsing PDFs (PyMuPDF)"):
            docs = self.parse_single_pdf(pdf_path)
            all_documents.extend(docs)
        
        return all_documents
