"""
chunking.py - Token-Aware Document Chunking Engine for HyRAG.

This module splits ingested page-level text into logically coherent chunks
using Token-Aware Recursive Character Text Splitting while maintaining strict metadata lineage.
"""

import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


# Default Tokenizer & Parameters optimized for BAAI/bge-small-en-v1.5
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE_TOKENS = 256
DEFAULT_CHUNK_OVERLAP_TOKENS = 32


def create_token_text_splitter(
    model_name: str = DEFAULT_MODEL_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS, 
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS
) -> RecursiveCharacterTextSplitter:
    """
    Creates and returns a Token-Aware RecursiveCharacterTextSplitter instance.

    Uses Hugging Face AutoTokenizer to measure text length in exact tokens
    while preserving paragraph and sentence boundaries.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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

    # Split text into string chunks (measured in tokens)
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
    chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS, 
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS
) -> List[Dict[str, Any]]:
    """
    Processes a complete dataset of ingested pages into token-aware text chunks.

    Args:
        pages_list (List[Dict[str, Any]]): List of page objects.
        chunk_size (int): Max tokens per chunk (default 256 tokens).
        chunk_overlap (int): Overlap in tokens (default 32 tokens).

    Returns:
        List[Dict[str, Any]]: Consolidated list of chunk objects across all documents.
    """
    print(f"✂️  Starting Token-Aware chunking (chunk_size={chunk_size} tokens, overlap={chunk_overlap} tokens)...")
    splitter = create_token_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    all_chunks = []

    for page in pages_list:
        page_chunks = chunk_single_page(page, splitter)
        all_chunks.extend(page_chunks)

    print(f"✅ Token Chunking Complete: Processed {len(pages_list)} page(s) into {len(all_chunks)} chunk(s).")
    return all_chunks


if __name__ == "__main__":
    from src.ingestion import ingest_directory

    raw_docs_dir = os.path.join("data", "raw_documents")
    try:
        print("--- PHASE 4: INGESTION ---")
        pages = ingest_directory(raw_docs_dir)

        print("\n--- PHASE 5 (UPGRADED): TOKEN-AWARE CHUNKING ---")
        chunks = chunk_dataset(pages, chunk_size=256, chunk_overlap=32)

        if chunks:
            print("\n🔍 Sample Chunk Metadata (First chunk of dataset):")
            print(chunks[0]["metadata"])
            print("\n📝 Sample Chunk Text:")
            print(chunks[0]["text"])

    except Exception as e:
        print(f"❌ Integration Test Error: {e}")
