"""
graph_retrieval.py - Multi-Hop Graph Retrieval & Bounded Traversal Engine for HyRAG.

This module performs query entity linking, bounded 1-hop/multi-hop subgraph traversals,
and gathers provenance-backed supporting text chunks for hybrid evidence fusion.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np

from src.graph_store import BaseGraphStore
from src.entity_resolution import normalize_entity_name

logger = logging.getLogger("HyRAG.GraphRetrieval")

# Default Traversal Hyperparameters
DEFAULT_MAX_HOPS = 2
DEFAULT_TOP_K_ENTITIES = 5
DEFAULT_TOP_K_RELATIONSHIPS = 15
DEFAULT_MAX_GRAPH_NODES = 20
DEFAULT_MIN_RELATION_CONFIDENCE = 0.60


def link_query_entities(
    query: str,
    graph_store: BaseGraphStore,
    top_k: int = DEFAULT_TOP_K_ENTITIES,
    embedding_model=None
) -> List[Dict[str, Any]]:
    """
    Identifies and ranks seed entity nodes in the graph corresponding to the user's query.
    Uses alias dictionary lookup, substring matching, and semantic vector similarity.
    """
    matched_nodes: Dict[str, Tuple[Dict[str, Any], float]] = {}  # node_id -> (node_data, score)
    norm_query = normalize_entity_name(query)
    query_words = set(norm_query.split())

    if not hasattr(graph_store, "graph") or graph_store.graph.number_of_nodes() == 0:
        return []

    # 1. Exact alias & token matching
    for node_id, data in graph_store.graph.nodes(data=True):
        name = data.get("name", "")
        norm_name = normalize_entity_name(name)
        aliases = [normalize_entity_name(a) for a in data.get("aliases", [])] + [norm_name]

        # Exact substring or alias match
        for alias in aliases:
            if alias and (alias in norm_query or re.search(r'\b' + re.escape(alias) + r'\b', norm_query)):
                score = 1.0 + (len(alias) / 50.0)  # Favor more specific entity matches
                matched_nodes[node_id] = (data, score)
                break
            # Word overlap match for multi-word entities
            alias_words = set(alias.split())
            if len(alias_words) > 1 and alias_words.issubset(query_words):
                score = 0.85
                if node_id not in matched_nodes or score > matched_nodes[node_id][1]:
                    matched_nodes[node_id] = (data, score)

    # 2. Embedding semantic search if few entities matched
    if embedding_model and len(matched_nodes) < top_k:
        try:
            all_nodes = list(graph_store.graph.nodes(data=True))
            if all_nodes:
                node_names = [d.get("name", nid) for nid, d in all_nodes]
                q_vec = embedding_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
                n_vecs = embedding_model.encode(node_names, normalize_embeddings=True, convert_to_numpy=True)
                sims = np.dot(n_vecs, q_vec)

                top_indices = np.argsort(sims)[::-1][:top_k]
                for idx in top_indices:
                    sim = float(sims[idx])
                    if sim >= 0.65:
                        nid, data = all_nodes[idx]
                        if nid not in matched_nodes:
                            matched_nodes[nid] = (data, sim)
        except Exception as e:
            logger.warning(f"Semantic entity linking fallback failed: {e}")

    # Sort by score descending
    sorted_matches = sorted(matched_nodes.values(), key=lambda x: x[1], reverse=True)[:top_k]
    return [item[0] for item in sorted_matches]


def search_knowledge_graph(
    query: str,
    graph_store: BaseGraphStore,
    embedding_model=None,
    max_hops: int = DEFAULT_MAX_HOPS,
    top_k_entities: int = DEFAULT_TOP_K_ENTITIES,
    max_graph_nodes: int = DEFAULT_MAX_GRAPH_NODES,
    min_relation_confidence: float = DEFAULT_MIN_RELATION_CONFIDENCE
) -> Dict[str, Any]:
    """
    Executes a bounded multi-hop Knowledge Graph search for a query.

    Returns:
        Dict[str, Any]:
            - 'seed_entities': List of linked starting nodes
            - 'subgraph': Subgraph dictionary (nodes, edges, supporting_chunk_ids, paths)
            - 'supporting_chunk_ids': List of chunk IDs referenced by the subgraph
            - 'facts_summary': Formatted string of retrieved graph facts for prompt inclusion
    """
    # 1. Link seed entities
    seed_entities = link_query_entities(
        query=query,
        graph_store=graph_store,
        top_k=top_k_entities,
        embedding_model=embedding_model
    )

    if not seed_entities:
        return {
            "seed_entities": [],
            "subgraph": {"nodes": [], "edges": [], "supporting_chunk_ids": [], "paths": []},
            "supporting_chunk_ids": [],
            "facts_summary": ""
        }

    seed_ids = [e["id"] for e in seed_entities]

    # 2. Extract bounded multi-hop subgraph
    subgraph = graph_store.get_subgraph(
        seed_node_ids=seed_ids,
        max_hops=max_hops,
        max_nodes=max_graph_nodes,
        min_confidence=min_relation_confidence
    )

    # 3. Format facts summary
    facts_lines = []
    for path in subgraph.get("paths", []):
        facts_lines.append(f"• {path}")

    facts_summary = "\n".join(facts_lines) if facts_lines else "No direct relational facts discovered."

    return {
        "seed_entities": seed_entities,
        "subgraph": subgraph,
        "supporting_chunk_ids": subgraph.get("supporting_chunk_ids", []),
        "facts_summary": facts_summary
    }
