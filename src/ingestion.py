"""
ingestion.py - Document Ingestion Engine for HyRAG.

This module provides utility functions to load, inspect, clean, and extract
text along with metadata from enterprise PDF documents.
"""

import os
from typing import List, Dict, Any
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Reads a single PDF file page by page and returns a list of dictionaries.
    Each dictionary contains the extracted page text and its associated metadata.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Dict[str, Any]]: List of pages with 'text' and 'metadata'.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    reader = PdfReader(pdf_path)
    file_name = os.path.basename(pdf_path)
    total_pages = len(reader.pages)
    
    # Create a clean doc_id slug from filename
    doc_id = f"doc_{file_name.replace('.pdf', '').replace('-', '_').replace(' ', '_')}"

    extracted_pages = []

    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        # Clean basic whitespace
        cleaned_text = " ".join(raw_text.split())

        page_data = {
            "text": cleaned_text,
            "metadata": {
                "doc_id": doc_id,
                "file_name": file_name,
                "file_path": os.path.abspath(pdf_path),
                "page_number": page_idx + 1,  # 1-based page index
                "total_pages": total_pages,
            }
        }
        extracted_pages.append(page_data)

    return extracted_pages


def ingest_directory(dir_path: str) -> List[Dict[str, Any]]:
    """
    Scans a directory for all PDF files and extracts text + metadata from each.

    Args:
        dir_path (str): Path to the directory containing PDFs.

    Returns:
        List[Dict[str, Any]]: Consolidated list of all extracted page objects.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    all_documents = []
    pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]

    print(f"📁 Found {len(pdf_files)} PDF(s) in '{dir_path}' for ingestion.")

    for pdf_file in pdf_files:
        full_path = os.path.join(dir_path, pdf_file)
        pages = extract_text_from_pdf(full_path)
        all_documents.extend(pages)
        print(f"   ✓ Ingested '{pdf_file}': {len(pages)} page(s).")

    return all_documents


if __name__ == "__main__":
    # Test script locally when run directly
    raw_docs_dir = os.path.join("data", "raw_documents")
    try:
        docs = ingest_directory(raw_docs_dir)
        print(f"\n✅ Total pages ingested across all PDFs: {len(docs)}")
        if docs:
            print("\n🔍 Sample Page Metadata (Page 1 of first doc):")
            print(docs[0]["metadata"])
            print("\n📝 Sample Text Snippet (First 200 chars):")
            print(docs[0]["text"][:200] + "...")
    except Exception as e:
        print(f"❌ Error during ingestion test: {e}")
