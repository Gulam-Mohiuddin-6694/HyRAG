"""
embeddings.py - Vector Embedding Generator & FAISS Index Manager for HyRAG.

This module loads the BAAI/bge-small-en-v1.5 embedding model, converts text chunks
into 384-dimensional dense vectors, manages the local FAISS vector database,
and provides semantic search capabilities.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Ensure safe console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger("HyRAG.Embeddings")
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Loads and returns the SentenceTransformer embedding model.

    Args:
        model_name (str): HuggingFace model identifier.

    Returns:
        SentenceTransformer: Loaded model instance.
    """
    print(f"[HyRAG] Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print(f"[HyRAG] Embedding model loaded! (Vector Dimensions: {model.get_embedding_dimension()})")
    return model


def generate_chunk_embeddings(
    chunks: List[Dict[str, Any]], 
    model: SentenceTransformer,
    batch_size: int = 64
) -> np.ndarray:
    """
    Generates L2-normalized 384-dimensional embeddings for a list of chunks.

    Args:
        chunks (List[Dict[str, Any]]): List of chunk objects.
        model (SentenceTransformer): Active embedding model.
        batch_size (int): Batch size for encoding (default 64).

    Returns:
        np.ndarray: 2D numpy array of shape (N, 384) with float32 dtype.
    """
    if not chunks:
        raise ValueError("Cannot generate embeddings for an empty list of chunks.")

    texts = [c["text"] for c in chunks]
    print(f"[HyRAG] Generating embeddings for {len(texts)} chunk(s) (batch_size={batch_size})...")

    # Generate L2-normalized embeddings for cosine similarity matching
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")
    print(f"[HyRAG] Embeddings matrix generated! Shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Builds a FAISS IndexFlatIP (Inner Product) index for exact cosine similarity search.

    Args:
        embeddings (np.ndarray): 2D array of shape (N, dim) with float32 dtype.

    Returns:
        faiss.IndexFlatIP: Initialized and populated FAISS index.
    """
    dim = embeddings.shape[1]
    print(f"[HyRAG] Building FAISS IndexFlatIP (Dimension: {dim})...")
    
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"[HyRAG] FAISS Index Built! Total vectors stored: {index.ntotal}")
    return index


def save_faiss_index(
    index: faiss.IndexFlatIP, 
    chunks: List[Dict[str, Any]], 
    output_dir: str = os.path.join("storage", "faiss_index")
) -> None:
    """
    Saves the FAISS index binary file and the corresponding chunks metadata JSON file.

    Args:
        index (faiss.IndexFlatIP): Populated FAISS index.
        chunks (List[Dict[str, Any]]): Original chunk objects.
        output_dir (str): Directory where index files will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(output_dir, "index.faiss")
    metadata_path = os.path.join(output_dir, "chunks_metadata.json")

    # Save FAISS binary index
    faiss.write_index(index, index_path)
    
    # Save matching metadata dictionary list
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[HyRAG] FAISS Index successfully saved to '{index_path}'")
    print(f"[HyRAG] Chunks Metadata successfully saved to '{metadata_path}'")


def search_faiss_index(
    query: str, 
    model: SentenceTransformer, 
    index: faiss.IndexFlatIP, 
    chunks: List[Dict[str, Any]], 
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Searches the FAISS vector index for top_k chunks most relevant to a query.

    Args:
        query (str): User question or search phrase.
        model (SentenceTransformer): Active BGE embedding model.
        index (faiss.IndexFlatIP): Populated FAISS index.
        chunks (List[Dict[str, Any]]): Metadata list corresponding 1:1 to index positions.
        top_k (int): Number of top search results to return (default 3).

    Returns:
        List[Dict[str, Any]]: Top-K matching chunks with similarity score and metadata attached.
    """
    query_vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx < len(chunks):
            matched_chunk = chunks[idx].copy()
            matched_chunk["similarity_score"] = float(score)
            results.append(matched_chunk)

    return results
