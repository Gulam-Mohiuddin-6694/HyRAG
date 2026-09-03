"""
retrieval.py - Tri-Hybrid Retrieval Engine (Dense FAISS + Sparse BM25 + Graph RAG).

This module implements BM25 keyword search, FAISS dense search integration,
Graph RAG multi-hop retrieval, and Tri-Hybrid Reciprocal Rank Fusion (RRF).
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.embeddings import load_embedding_model, search_faiss_index
from src.graph_store import BaseGraphStore
from src.graph_retrieval import search_knowledge_graph

logger = logging.getLogger("HyRAG.Retrieval")


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
    graph_results: Optional[List[Dict[str, Any]]] = None,
    k: int = 60,
    top_k: int = 5,
    w_dense: float = 1.0,
    w_sparse: float = 1.0,
    w_graph: float = 1.2
) -> List[Dict[str, Any]]:
    """
    Combines dense FAISS, sparse BM25, and Graph-derived chunks using
    Tri-Hybrid Reciprocal Rank Fusion (RRF).

    Formula: RRF(d) = w_dense/(k + rank_dense) + w_sparse/(k + rank_sparse) + w_graph/(k + rank_graph)

    Args:
        dense_results: Ranked search results from FAISS.
        sparse_results: Ranked search results from BM25.
        graph_results: Optional ranked search results from Graph RAG.
        k: RRF smoothing constant (default 60).
        top_k: Number of top hybrid results to return (default 5).
        w_dense: Weight for dense vector retrieval.
        w_sparse: Weight for sparse BM25 retrieval.
        w_graph: Weight for graph evidence chunks.

    Returns:
        List[Dict[str, Any]]: Deduplicated hybrid search results sorted by RRF score.
    """
    rrf_map: Dict[str, Dict[str, Any]] = {}

    def _add_source(results_list: List[Dict[str, Any]], rank_key: str, weight: float):
        for rank, chunk in enumerate(results_list):
            chunk_id = chunk["chunk_id"]
            pos_rank = rank + 1  # 1-based rank
            score = weight / (k + pos_rank)

            if chunk_id not in rrf_map:
                rrf_map[chunk_id] = {
                    "chunk": chunk.copy(),
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "graph_rank": None
                }
            rrf_map[chunk_id][rank_key] = pos_rank
            rrf_map[chunk_id]["rrf_score"] += score

    # Process Dense FAISS
    _add_source(dense_results, "dense_rank", w_dense)

    # Process Sparse BM25
    _add_source(sparse_results, "sparse_rank", w_sparse)

    # Process Graph Supporting Chunks (if provided)
    if graph_results:
        _add_source(graph_results, "graph_rank", w_graph)

    # Sort merged results by RRF score descending
    sorted_items = sorted(rrf_map.values(), key=lambda item: item["rrf_score"], reverse=True)

    hybrid_results = []
    for item in sorted_items[:top_k]:
        res_chunk = item["chunk"]
        res_chunk["rrf_score"] = item["rrf_score"]
        res_chunk["dense_rank"] = item["dense_rank"]
        res_chunk["sparse_rank"] = item["sparse_rank"]
        res_chunk["graph_rank"] = item["graph_rank"]
        hybrid_results.append(res_chunk)

    return hybrid_results


def tri_hybrid_search(
    query: str,
    model: SentenceTransformer,
    faiss_index: faiss.IndexFlatIP,
    bm25_index: BM25Okapi,
    chunks: List[Dict[str, Any]],
    graph_store: Optional[BaseGraphStore] = None,
    top_k_chunks: int = 5,
    rrf_k: int = 60,
    max_hops: int = 2,
    min_relation_confidence: float = 0.60
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes full Tri-Hybrid Retrieval (Vector + Keyword + Graph RAG).

    Returns:
        Tuple:
            - List[Dict[str, Any]]: Ranked top hybrid chunks with fusion metadata.
            - Dict[str, Any]: Graph search outcome (seed entities, subgraph, facts, supporting chunks).
    """
    chunk_map = {c["chunk_id"]: c for c in chunks}

    # 1. Dense Vector Search
    dense_results = search_faiss_index(query, model, faiss_index, chunks, top_k=10)

    # 2. Sparse BM25 Search
    sparse_results = search_bm25(query, bm25_index, chunks, top_k=10)

    # 3. Graph RAG Search
    graph_res = {
        "seed_entities": [],
        "subgraph": {"nodes": [], "edges": [], "supporting_chunk_ids": [], "paths": []},
        "supporting_chunk_ids": [],
        "facts_summary": ""
    }
    graph_chunk_results: List[Dict[str, Any]] = []

    if graph_store is not None:
        try:
            graph_res = search_knowledge_graph(
                query=query,
                graph_store=graph_store,
                embedding_model=model,
                max_hops=max_hops,
                min_relation_confidence=min_relation_confidence
            )

            # Map graph supporting chunk IDs to full chunk objects
            for cid in graph_res.get("supporting_chunk_ids", []):
                if cid in chunk_map:
                    g_chunk = chunk_map[cid].copy()
                    g_chunk["is_graph_provenance"] = True
                    graph_chunk_results.append(g_chunk)

        except Exception as e:
            logger.error(f"Graph RAG search encountered error, falling back gracefully: {e}")

    # 4. Tri-Hybrid Reciprocal Rank Fusion
    hybrid_chunks = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        graph_results=graph_chunk_results if graph_chunk_results else None,
        k=rrf_k,
        top_k=top_k_chunks,
        w_dense=1.0,
        w_sparse=1.0,
        w_graph=1.25
    )

    return hybrid_chunks, graph_res
