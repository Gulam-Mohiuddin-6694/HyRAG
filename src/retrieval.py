"""
retrieval.py - Sparse BM25 & Reciprocal Rank Fusion (RRF) Hybrid Retrieval Engine.

This module implements BM25 keyword search, FAISS dense search integration,
and Reciprocal Rank Fusion (RRF) for hallucination-aware hybrid retrieval.
"""

import os
import json
import re
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.embeddings import load_embedding_model, search_faiss_index


def tokenize_text(text: str) -> List[str]:
    """
    Simple whitespace and alphanumeric tokenizer for BM25.
    Converts text to lowercase and extracts word tokens.
    """
    return re.findall(r'\w+', text.lower())


def build_bm25_index(chunks: List[Dict[str, Any]]) -> Tuple[BM25Okapi, List[List[str]]]:
    """
    Tokenizes chunk texts and builds a BM25Okapi index.

    Args:
        chunks (List[Dict[str, Any]]): List of chunk objects.

    Returns:
        Tuple[BM25Okapi, List[List[str]]]: Initialized BM25 index and tokenized corpus.
    """
    print(f"🏗️  Building BM25 Sparse Index for {len(chunks)} chunk(s)...")
    tokenized_corpus = [tokenize_text(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"✅ BM25 Index Built Successfully!")
    return bm25, tokenized_corpus


def search_bm25(
    query: str, 
    bm25: BM25Okapi, 
    chunks: List[Dict[str, Any]], 
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Searches the BM25 index and returns top_k matching chunks with sparse BM25 scores.

    Args:
        query (str): Search query phrase.
        bm25 (BM25Okapi): Active BM25 index instance.
        chunks (List[Dict[str, Any]]): List of chunk objects corresponding 1:1 with corpus.
        top_k (int): Number of top results to return.

    Returns:
        List[Dict[str, Any]]: Top matching chunk dictionaries with 'bm25_score' attached.
    """
    tokenized_query = tokenize_text(query)
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Sort indices by score in descending order
    top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        score = doc_scores[idx]
        if score > 0:
            chunk_copy = chunks[idx].copy()
            chunk_copy["bm25_score"] = float(score)
            results.append(chunk_copy)

    return results


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Combines dense FAISS results and sparse BM25 results using Reciprocal Rank Fusion (RRF).

    Formula: RRF_Score(d) = 1 / (k + rank_dense(d)) + 1 / (k + rank_sparse(d))

    Args:
        dense_results (List[Dict[str, Any]]): Ranked search results from FAISS.
        sparse_results (List[Dict[str, Any]]): Ranked search results from BM25.
        k (int): RRF smoothing constant (default 60).
        top_k (int): Number of top hybrid results to return (default 5).

    Returns:
        List[Dict[str, Any]]: Deduplicated hybrid search results sorted by RRF score.
    """
    rrf_map: Dict[str, Dict[str, Any]] = {}

    # Process Dense FAISS Rankings
    for rank, chunk in enumerate(dense_results):
        chunk_id = chunk["chunk_id"]
        dense_rank = rank + 1  # 1-based rank
        score = 1.0 / (k + dense_rank)

        if chunk_id not in rrf_map:
            rrf_map[chunk_id] = {
                "chunk": chunk.copy(),
                "rrf_score": 0.0,
                "dense_rank": dense_rank,
                "sparse_rank": None
            }
        rrf_map[chunk_id]["rrf_score"] += score

    # Process Sparse BM25 Rankings
    for rank, chunk in enumerate(sparse_results):
        chunk_id = chunk["chunk_id"]
        sparse_rank = rank + 1  # 1-based rank
        score = 1.0 / (k + sparse_rank)

        if chunk_id not in rrf_map:
            rrf_map[chunk_id] = {
                "chunk": chunk.copy(),
                "rrf_score": 0.0,
                "dense_rank": None,
                "sparse_rank": sparse_rank
            }
            rrf_map[chunk_id]["rrf_score"] += score
        else:
            rrf_map[chunk_id]["sparse_rank"] = sparse_rank
            rrf_map[chunk_id]["rrf_score"] += score

    # Sort merged results by RRF score descending
    sorted_items = sorted(rrf_map.values(), key=lambda item: item["rrf_score"], reverse=True)

    hybrid_results = []
    for item in sorted_items[:top_k]:
        res_chunk = item["chunk"]
        res_chunk["rrf_score"] = item["rrf_score"]
        res_chunk["dense_rank"] = item["dense_rank"]
        res_chunk["sparse_rank"] = item["sparse_rank"]
        hybrid_results.append(res_chunk)

    return hybrid_results


if __name__ == "__main__":
    # Integration Test: Ingestion -> Chunking -> FAISS + BM25 -> Hybrid RRF
    raw_docs_dir = os.path.join("data", "raw_documents")
    storage_dir = os.path.join("storage", "faiss_index")
    
    try:
        print("--- LOADING PRE-INDEXED FAISS & CHUNKS METADATA ---")
        index_path = os.path.join(storage_dir, "index.faiss")
        metadata_path = os.path.join(storage_dir, "chunks_metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("FAISS index files not found. Run 'python -m src.embeddings' first!")

        faiss_index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        model = load_embedding_model()

        print("\n--- BUILDING BM25 SPARSE INDEX ---")
        bm25, corpus = build_bm25_index(chunks)

        # Step 8 Verification: Compare FAISS, BM25, and Hybrid RRF on Enterprise Queries!
        test_queries = [
            "Section 2.2.2 Stockholder-Demanded Special Meetings",
            "What is AWS Identity and Access Management MFA policy?",
            "What is Amazon policy on employee conflict of interest?"
        ]

        for q in test_queries:
            print(f"\n========================================================")
            print(f"❓ TEST QUERY: '{q}'")
            print(f"========================================================")

            # 1. FAISS Search
            dense_res = search_faiss_index(q, model, faiss_index, chunks, top_k=5)
            
            # 2. BM25 Search
            sparse_res = search_bm25(q, bm25, chunks, top_k=5)
            
            # 3. Hybrid RRF Fusion
            hybrid_res = reciprocal_rank_fusion(dense_res, sparse_res, k=60, top_k=3)

            print(f"\n🏆 TOP 3 HYBRID (RRF) RESULTS:")
            for rank, h in enumerate(hybrid_res):
                d_rank = f"#{h['dense_rank']}" if h['dense_rank'] else "N/A"
                s_rank = f"#{h['sparse_rank']}" if h['sparse_rank'] else "N/A"
                print(f"   Rank #{rank+1} | RRF Score: {h['rrf_score']:.6f} | (FAISS: {d_rank}, BM25: {s_rank})")
                print(f"   Source: {h['metadata']['file_name']} (Page {h['metadata']['page_number']}) | Chunk ID: {h['chunk_id']}")
                print(f"   Text Snippet: {h['text'][:200]}...\n")

    except Exception as e:
        print(f"❌ Error during hybrid retrieval test: {e}")
