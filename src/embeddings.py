"""
embeddings.py - Vector Embedding Generator & FAISS Index Manager for HyRAG.

This module loads the BAAI/bge-small-en-v1.5 embedding model, converts text chunks
into 384-dimensional dense vectors, manages the local FAISS vector database,
and provides semantic search capabilities.
"""

import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Loads and returns the SentenceTransformer embedding model.

    Args:
        model_name (str): HuggingFace model identifier.

    Returns:
        SentenceTransformer: Loaded model instance.
    """
    print(f"🔄 Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print(f"✅ Embedding model loaded! (Vector Dimensions: {model.get_embedding_dimension()})")
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
    print(f"⚡ Generating embeddings for {len(texts)} chunk(s) (batch_size={batch_size})...")

    # Generate L2-normalized embeddings for cosine similarity matching
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")
    print(f"✅ Embeddings matrix generated! Shape: {embeddings.shape}")
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
    print(f"🏗️  Building FAISS IndexFlatIP (Dimension: {dim})...")
    
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"✅ FAISS Index Built! Total vectors stored: {index.ntotal}")
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

    print(f"💾 FAISS Index successfully saved to '{index_path}'")
    print(f"💾 Chunks Metadata successfully saved to '{metadata_path}'")


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


if __name__ == "__main__":
    from src.ingestion import ingest_directory
    from src.chunking import chunk_dataset

    raw_docs_dir = os.path.join("data", "raw_documents")
    
    try:
        print("--- PHASE 4: INGESTION ---")
        pages = ingest_directory(raw_docs_dir)

        print("\n--- PHASE 5: TOKEN CHUNKING ---")
        chunks = chunk_dataset(pages)

        print("\n--- PHASE 6: EMBEDDINGS & FULL FAISS INDEX BUILD ---")
        model = load_embedding_model()

        # Step 8: Full Dataset Processing (3,501 chunks)
        full_embeddings = generate_chunk_embeddings(chunks, model, batch_size=64)
        faiss_index = build_faiss_index(full_embeddings)
        
        # Save FAISS Index & Metadata locally
        save_faiss_index(faiss_index, chunks)

        # Step 9 Verification: Run a live semantic search test query!
        test_query = "What is Amazon's policy on workplace gifts and entertainment?"
        print(f"\n🔍 Live Semantic Search Test Query: '{test_query}'")
        search_results = search_faiss_index(test_query, model, faiss_index, chunks, top_k=2)

        print(f"\n🎯 Top {len(search_results)} Search Results:")
        for idx, res in enumerate(search_results):
            print(f"\n--- Result #{idx+1} (Similarity Score: {res['similarity_score']:.4f}) ---")
            print(f"📄 Source: {res['metadata']['file_name']} (Page {res['metadata']['page_number']})")
            print(f"📌 Chunk ID: {res['chunk_id']}")
            print(f"📝 Text Snippet: {res['text'][:250]}...")

    except Exception as e:
        print(f"❌ Error during full embedding pipeline execution: {e}")
