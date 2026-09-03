"""
pipeline_manager.py - Unified Ingestion, Indexing & Lifecycle Pipeline Manager for HyRAG.

Coordinates the end-to-end Graph RAG pipeline upon document upload,
updating FAISS vector indexes, BM25 corpus, NetworkX Knowledge Graph,
and the SQLite document registry.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.ingestion import extract_text_from_file, create_doc_id
from src.chunking import chunk_dataset
from src.embeddings import (
    generate_chunk_embeddings,
    build_faiss_index,
    save_faiss_index,
    load_embedding_model
)
from src.graph_store import NetworkXGraphStore, BaseGraphStore
from src.graph_extractor import process_and_index_chunks_to_graph
from src.entity_resolution import EntityResolver
from src.database import register_document, mark_document_deleted, get_document_by_id

logger = logging.getLogger("HyRAG.PipelineManager")


def process_new_upload(
    file_bytes: bytes,
    file_name: str,
    uploaded_by: str,
    embedding_model: SentenceTransformer,
    graph_store: BaseGraphStore,
    faiss_index: Optional[faiss.IndexFlatIP] = None,
    existing_chunks: Optional[List[Dict[str, Any]]] = None,
    raw_docs_dir: str = os.path.join("data", "raw_documents"),
    faiss_storage_dir: str = os.path.join("storage", "faiss_index"),
    graph_storage_path: str = os.path.join("storage", "graph_store", "knowledge_graph.json")
) -> Dict[str, Any]:
    """
    Executes the real-time Graph RAG pipeline for an uploaded document:
    1. Saves file to data/raw_documents
    2. Extracts text & metadata
    3. Chunks text with token boundaries (256 tokens, 32 overlap)
    4. Generates dense embeddings & updates FAISS index
    5. Extracts entities & relations into Knowledge Graph
    6. Registers document in SQLite database with status 'COMPLETED'
    """
    os.makedirs(raw_docs_dir, exist_ok=True)
    file_path = os.path.join(raw_docs_dir, file_name)

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    file_size = len(file_bytes)
    ext = os.path.splitext(file_name)[1].upper().replace(".", "")
    doc_id = create_doc_id(file_name)

    logger.info(f"Processing uploaded document '{file_name}' (ID: {doc_id}, Size: {file_size} bytes)")

    # 1. Extract Text
    pages = extract_text_from_file(file_path)
    if not pages:
        raise ValueError(f"No extractable text found in file '{file_name}'.")

    # 2. Token-Aware Chunking
    new_chunks = chunk_dataset(pages, chunk_size=256, chunk_overlap=32)
    if not new_chunks:
        raise ValueError(f"Chunking failed for file '{file_name}'.")

    # 3. Dense Embeddings & FAISS Vector Index Update
    new_embeddings = generate_chunk_embeddings(new_chunks, embedding_model, batch_size=64)

    # Load existing chunks if not supplied
    meta_path = os.path.join(faiss_storage_dir, "chunks_metadata.json")
    if existing_chunks is None:
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
        else:
            existing_chunks = []

    # Filter out previous chunks for this doc_id to avoid duplicate indexing
    combined_chunks = [c for c in existing_chunks if c.get("metadata", {}).get("doc_id") != doc_id]
    combined_chunks.extend(new_chunks)

    # Rebuild FAISS index cleanly to maintain exact 1:1 chunk alignment
    full_embeddings = generate_chunk_embeddings(combined_chunks, embedding_model, batch_size=64)
    faiss_index = build_faiss_index(full_embeddings)
    save_faiss_index(faiss_index, combined_chunks, output_dir=faiss_storage_dir)

    # 4. Knowledge Graph Extraction & Update
    entity_resolver = EntityResolver(similarity_threshold=0.88, embedding_model=embedding_model)
    process_and_index_chunks_to_graph(
        chunks=new_chunks,
        graph_store=graph_store,
        entity_resolver=entity_resolver,
        use_llm=True
    )
    graph_store.save(graph_storage_path)

    g_stats = graph_store.stats()

    # 5. Register in SQLite Database
    register_document(
        doc_id=doc_id,
        file_name=file_name,
        file_path=os.path.abspath(file_path),
        file_type=ext,
        file_size_bytes=file_size,
        uploaded_by=uploaded_by,
        page_count=len(pages),
        chunk_count=len(new_chunks),
        entity_count=g_stats.get("total_nodes", 0),
        edge_count=g_stats.get("total_edges", 0),
        status="PROCESSED"
    )

    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "file_type": ext,
        "page_count": len(pages),
        "chunk_count": len(new_chunks),
        "total_chunks_indexed": len(combined_chunks),
        "total_graph_nodes": g_stats.get("total_nodes", 0),
        "total_graph_edges": g_stats.get("total_edges", 0),
        "status": "PROCESSED"
    }


def delete_document_from_system(
    doc_id: str,
    embedding_model: SentenceTransformer,
    graph_store: BaseGraphStore,
    raw_docs_dir: str = os.path.join("data", "raw_documents"),
    faiss_storage_dir: str = os.path.join("storage", "faiss_index"),
    graph_storage_path: str = os.path.join("storage", "graph_store", "knowledge_graph.json")
) -> Dict[str, Any]:
    """
    Cascades document deletion:
    1. Removes document provenance and orphan graph facts
    2. Prunes document chunks and rebuilds FAISS index
    3. Removes physical file
    4. Updates SQLite document registry
    """
    logger.info(f"Deleting document '{doc_id}' from system...")

    # 1. Cascade Graph deletion
    prune_stats = graph_store.delete_document_provenance(doc_id)
    graph_store.save(graph_storage_path)

    # 2. Prune FAISS Vector Chunks
    meta_path = os.path.join(faiss_storage_dir, "chunks_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        
        remaining_chunks = [c for c in all_chunks if c.get("metadata", {}).get("doc_id") != doc_id]

        if remaining_chunks:
            full_embeddings = generate_chunk_embeddings(remaining_chunks, embedding_model, batch_size=64)
            new_faiss_index = build_faiss_index(full_embeddings)
            save_faiss_index(new_faiss_index, remaining_chunks, output_dir=faiss_storage_dir)
        else:
            # Empty index
            dim = 384
            empty_index = faiss.IndexFlatIP(dim)
            save_faiss_index(empty_index, [], output_dir=faiss_storage_dir)

    # 3. Remove physical file if present
    doc_record = get_document_by_id(doc_id)
    if doc_record and doc_record.get("file_path") and os.path.exists(doc_record["file_path"]):
        try:
            os.remove(doc_record["file_path"])
        except Exception as e:
            logger.warning(f"Could not remove physical file for '{doc_id}': {e}")

    # 4. Mark Deleted in DB
    mark_document_deleted(doc_id)

    return {
        "doc_id": doc_id,
        "pruned_edges": prune_stats.get("pruned_edges", 0),
        "pruned_nodes": prune_stats.get("pruned_nodes", 0),
        "status": "DELETED"
    }


def reprocess_document_in_system(
    doc_id: str,
    uploaded_by: str,
    embedding_model: SentenceTransformer,
    graph_store: BaseGraphStore,
    raw_docs_dir: str = os.path.join("data", "raw_documents"),
    faiss_storage_dir: str = os.path.join("storage", "faiss_index"),
    graph_storage_path: str = os.path.join("storage", "graph_store", "knowledge_graph.json")
) -> Dict[str, Any]:
    """
    Re-processes an existing document:
    1. Reads existing file from storage
    2. Runs full extraction, chunking, embedding, and Knowledge Graph extraction
    """
    doc_record = get_document_by_id(doc_id)
    if not doc_record:
        raise ValueError(f"Document ID '{doc_id}' not found in registry.")

    file_path = doc_record.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Original file for '{doc_id}' not found at '{file_path}'.")

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    return process_new_upload(
        file_bytes=file_bytes,
        file_name=doc_record["file_name"],
        uploaded_by=uploaded_by,
        embedding_model=embedding_model,
        graph_store=graph_store,
        raw_docs_dir=raw_docs_dir,
        faiss_storage_dir=faiss_storage_dir,
        graph_storage_path=graph_storage_path
    )
