"""
ingestion.py - Multi-Format Document Ingestion Engine for HyRAG.

Supports extracting text and metadata from enterprise documents:
PDF, TXT, MD, DOCX, and CSV formats.
"""

import os
import csv
import zipfile
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Any
from pypdf import PdfReader

logger = logging.getLogger("HyRAG.Ingestion")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv"}


def clean_text_whitespace(raw_text: str) -> str:
    """Normalizes whitespace and removes null characters."""
    if not raw_text:
        return ""
    return " ".join(raw_text.split())


def create_doc_id(file_name: str) -> str:
    """Generates a clean doc_id slug from a filename."""
    base = os.path.splitext(file_name)[0]
    clean = base.replace('-', '_').replace(' ', '_').replace('.', '_')
    return f"doc_{clean.lower()}"


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Reads a PDF page-by-page using pypdf."""
    reader = PdfReader(pdf_path)
    file_name = os.path.basename(pdf_path)
    total_pages = len(reader.pages)
    doc_id = create_doc_id(file_name)

    extracted_pages = []
    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned = clean_text_whitespace(raw_text)
        if cleaned:
            extracted_pages.append({
                "text": cleaned,
                "metadata": {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "file_path": os.path.abspath(pdf_path),
                    "file_type": "PDF",
                    "page_number": page_idx + 1,
                    "total_pages": total_pages,
                }
            })
    return extracted_pages


def extract_text_from_txt_or_md(file_path: str) -> List[Dict[str, Any]]:
    """Reads a TXT or Markdown file."""
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].upper().replace(".", "")
    doc_id = create_doc_id(file_name)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            content = f.read()

    cleaned = clean_text_whitespace(content)
    if not cleaned:
        return []

    return [{
        "text": cleaned,
        "metadata": {
            "doc_id": doc_id,
            "file_name": file_name,
            "file_path": os.path.abspath(file_path),
            "file_type": ext,
            "page_number": 1,
            "total_pages": 1,
        }
    }]


def extract_text_from_docx(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a DOCX file by reading word/document.xml.
    Requires no external dependencies beyond Python standard library.
    """
    file_name = os.path.basename(file_path)
    doc_id = create_doc_id(file_name)

    try:
        with zipfile.ZipFile(file_path) as docx:
            xml_content = docx.read('word/document.xml')
        tree = ET.fromstring(xml_content)
        
        # Word XML namespace for text elements is w:t
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        text_elements = tree.findall('.//w:t', namespaces)
        raw_text = " ".join([elem.text for elem in text_elements if elem.text])
        cleaned = clean_text_whitespace(raw_text)

        if not cleaned:
            return []

        return [{
            "text": cleaned,
            "metadata": {
                "doc_id": doc_id,
                "file_name": file_name,
                "file_path": os.path.abspath(file_path),
                "file_type": "DOCX",
                "page_number": 1,
                "total_pages": 1,
            }
        }]
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX '{file_path}': {e}")
        return []


def extract_text_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """Reads a CSV file and converts records into structured text passages."""
    file_name = os.path.basename(file_path)
    doc_id = create_doc_id(file_name)

    rows_text = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers:
                for idx, row in enumerate(reader):
                    row_pairs = [f"{headers[i]}: {val}" for i, val in enumerate(row) if i < len(headers) and val]
                    if row_pairs:
                        rows_text.append(f"[Record {idx+1}] " + ", ".join(row_pairs))
            else:
                for idx, row in enumerate(reader):
                    rows_text.append(f"[Record {idx+1}] " + ", ".join(row))

        full_text = clean_text_whitespace(" | ".join(rows_text))
        if not full_text:
            return []

        return [{
            "text": full_text,
            "metadata": {
                "doc_id": doc_id,
                "file_name": file_name,
                "file_path": os.path.abspath(file_path),
                "file_type": "CSV",
                "page_number": 1,
                "total_pages": 1,
            }
        }]
    except Exception as e:
        logger.error(f"Failed to extract CSV '{file_path}': {e}")
        return []


def extract_text_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Dispatcher function to extract text from any supported file format:
    PDF, TXT, MD, DOCX, CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt_or_md(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".csv":
        return extract_text_from_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported: {SUPPORTED_EXTENSIONS}")


def ingest_directory(dir_path: str) -> List[Dict[str, Any]]:
    """
    Scans a directory for all supported files (PDF, TXT, MD, DOCX, CSV)
    and extracts text + metadata from each.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    all_documents = []
    files = [f for f in os.listdir(dir_path) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]

    print(f"[HyRAG] Found {len(files)} document(s) in '{dir_path}' for ingestion.")

    for doc_file in files:
        full_path = os.path.join(dir_path, doc_file)
        try:
            pages = extract_text_from_file(full_path)
            all_documents.extend(pages)
            print(f"   ✓ Ingested '{doc_file}': {len(pages)} page(s).")
        except Exception as e:
            print(f"   ❌ Failed to ingest '{doc_file}': {e}")

    return all_documents


if __name__ == "__main__":
    raw_docs_dir = os.path.join("data", "raw_documents")
    try:
        docs = ingest_directory(raw_docs_dir)
        print(f"\nTotal pages ingested across all files: {len(docs)}")
    except Exception as e:
        print(f"Error during ingestion test: {e}")
