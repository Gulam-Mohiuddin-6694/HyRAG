"""
chunking.py - Document Chunking & Text Splitting Module for HyRAG.

This module splits ingested page-level text into logically coherent chunks
using Recursive Character Text Splitting while maintaining strict metadata lineage.
"""

import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_text_splitter(
    chunk_size: int = 1000, 
    chunk_overlap: int = 150
) -> RecursiveCharacterTextSplitter:
    """
    Creates and returns a RecursiveCharacterTextSplitter instance.

    Separators priority:
    1. "\n\n" (Paragraphs)
    2. "\n"   (Lines)
    3. " "    (Words)
    4. ""     (Characters fallback)
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""]
    )


def chunk_single_page(
    page_dict: Dict[str, Any], 
    text_splitter: RecursiveCharacterTextSplitter
) -> List[Dict[str, Any]]:
    """
    Splits a single page document into chunks while attaching detailed chunk metadata.

    Args:
        page_dict (Dict[str, Any]): Dictionary containing 'text' and 'metadata' of a page.
        text_splitter (RecursiveCharacterTextSplitter): Active text splitter instance.

    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries containing text and chunk metadata.
    """
    raw_text = page_dict.get("text", "")
    page_metadata = page_dict.get("metadata", {})

    if not raw_text.strip():
        return []

    # Split text into string chunks
    text_chunks = text_splitter.split_text(raw_text)
    
    doc_id = page_metadata.get("doc_id", "unknown_doc")
    page_num = page_metadata.get("page_number", 0)

    chunk_objects = []

    for idx, chunk_text in enumerate(text_chunks):
        chunk_idx = idx + 1
        chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx}"

        # Combine page metadata with chunk-specific metadata
        chunk_metadata = {
            **page_metadata,
            "chunk_id": chunk_id,
            "chunk_index": chunk_idx,
            "char_count": len(chunk_text),
        }

        chunk_objects.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "metadata": chunk_metadata
        })

    return chunk_objects


def chunk_dataset(
    pages_list: List[Dict[str, Any]], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Processes a complete dataset of ingested pages into text chunks.

    Args:
        pages_list (List[Dict[str, Any]]): List of page objects.
        chunk_size (int): Max characters per chunk (default 1000).
        chunk_overlap (int): Overlap in characters (default 150).

    Returns:
        List[Dict[str, Any]]: Consolidated list of chunk objects across all documents.
    """
    print(f"✂️  Starting chunking process (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    splitter = create_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    all_chunks = []

    for page in pages_list:
        page_chunks = chunk_single_page(page, splitter)
        all_chunks.extend(page_chunks)

    print(f"✅ Chunking Complete: Processed {len(pages_list)} page(s) into {len(all_chunks)} chunk(s).")
    return all_chunks


if __name__ == "__main__":
    # Integration test with Phase 4 Ingestion Engine
    from src.ingestion import ingest_directory

    raw_docs_dir = os.path.join("data", "raw_documents")
    try:
        print("--- PHASE 4: INGESTION ---")
        pages = ingest_directory(raw_docs_dir)

        print("\n--- PHASE 5: CHUNKING ---")
        chunks = chunk_dataset(pages, chunk_size=1000, chunk_overlap=150)

        if chunks:
            print("\n🔍 Sample Chunk Metadata (First chunk of dataset):")
            print(chunks[0]["metadata"])
            print("\n📝 Sample Chunk Text:")
            print(chunks[0]["text"])

    except Exception as e:
        print(f"❌ Integration Test Error: {e}")
