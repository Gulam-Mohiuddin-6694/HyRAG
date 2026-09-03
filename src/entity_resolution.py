"""
entity_resolution.py - Entity Resolution, Canonicalization & Deduplication for HyRAG.

This module resolves raw entity surface forms to canonical entities using
normalized naming rules, known acronym/alias mappings, entity type constraints,
and vector embedding cosine similarity gating.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

logger = logging.getLogger("HyRAG.EntityResolution")

# Standard enterprise aliases & acronym mappings
KNOWN_ACRONYMS = {
    "aws": "amazon web services",
    "amazon aws": "amazon web services",
    "aws cloud": "amazon web services",
    "iam": "identity and access management",
    "aws iam": "identity and access management",
    "mfa": "multi-factor authentication",
    "2fa": "two-factor authentication",
    "s3": "simple storage service",
    "aws s3": "simple storage service",
    "ec2": "elastic compute cloud",
    "aws ec2": "elastic compute cloud",
    "waf": "web application firewall",
    "aws waf": "web application firewall",
    "vpc": "virtual private cloud",
    "kms": "key management service",
    "aws kms": "key management service",
    "ceo": "chief executive officer",
    "cfo": "chief financial officer",
    "cto": "chief technology officer",
    "ciso": "chief information security officer",
    "coo": "chief operating officer",
    "sla": "service level agreement",
    "gdpr": "general data protection regulation",
    "hipaa": "health insurance portability and accountability act",
    "soc 2": "service organization control 2",
    "pci dss": "payment card industry data security standard",
    "rbac": "role-based access control",
    "abac": "attribute-based access control",
}


def normalize_entity_name(name: str) -> str:
    """
    Normalizes an entity name by stripping punctuation, extra whitespace,
    and trailing parenthetical acronyms.
    """
    if not name:
        return ""
    # Strip brackets/parentheses e.g. "Amazon Web Services (AWS)" -> "Amazon Web Services"
    cleaned = re.sub(r'\(.*?\)', '', name)
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    # Remove punctuation except hyphens inside words
    cleaned = re.sub(r'[^\w\s-]', ' ', cleaned)
    # Collapse multiple whitespaces
    cleaned = " ".join(cleaned.strip().lower().split())
    return cleaned


def resolve_canonical_name(name: str, entity_type: str = "Concept") -> Tuple[str, str, List[str]]:
    """
    Resolves raw name into (canonical_id, canonical_name, aliases).
    
    Args:
        name: Raw entity text extracted from document or query.
        entity_type: Entity classification type.

    Returns:
        Tuple of (canonical_id, canonical_name, list_of_aliases)
    """
    norm = normalize_entity_name(name)
    aliases = [name.strip()]

    # Check known acronym expansion
    if norm in KNOWN_ACRONYMS:
        expanded = KNOWN_ACRONYMS[norm]
        canonical_name = expanded.title()
        aliases.append(name.strip())
        aliases.append(norm)
        canonical_id = f"ent_{expanded.replace(' ', '_').replace('-', '_')}"
    else:
        # Check reverse acronym mapping
        for acr, exp in KNOWN_ACRONYMS.items():
            if norm == exp:
                aliases.append(acr.upper())
                break
        canonical_name = name.strip().title()
        canonical_id = f"ent_{norm.replace(' ', '_').replace('-', '_')}"

    # Disambiguation safeguard for short/ambiguous names
    if len(norm) <= 2 and norm not in KNOWN_ACRONYMS:
        canonical_id = f"ent_{entity_type.lower()}_{norm}"

    return canonical_id, canonical_name, list(set(aliases))


class EntityResolver:
    """
    Resolves candidate entities against existing Graph Store entities
    using type compatibility and optional embedding similarity.
    """

    def __init__(self, similarity_threshold: float = 0.88, embedding_model=None):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

    def resolve_or_merge(
        self,
        candidate_name: str,
        candidate_type: str,
        existing_nodes: List[Dict[str, Any]]
    ) -> Tuple[str, str, str, List[str]]:
        """
        Determines whether a candidate entity matches an existing entity node.

        Returns:
            Tuple: (node_id, canonical_name, entity_type, aliases)
        """
        cand_id, cand_canon_name, cand_aliases = resolve_canonical_name(candidate_name, candidate_type)
        cand_norm = normalize_entity_name(candidate_name)

        # 1. Exact canonical ID or alias match
        for node in existing_nodes:
            node_id = node.get("id")
            node_type = node.get("entity_type", "")
            node_aliases = [normalize_entity_name(a) for a in node.get("aliases", [])]

            if node_id == cand_id or cand_norm in node_aliases:
                # Merge if types are compatible or one is generic
                if node_type == candidate_type or node_type in ("Concept", "Unknown") or candidate_type in ("Concept", "Unknown"):
                    merged_aliases = list(set(node.get("aliases", []) + cand_aliases))
                    return node_id, node.get("name", cand_canon_name), node_type or candidate_type, merged_aliases

        # 2. Embedding-based semantic similarity check (if embedding model is present)
        if self.embedding_model and existing_nodes and len(cand_norm) > 3:
            candidate_vec = self.embedding_model.encode([cand_canon_name], normalize_embeddings=True, convert_to_numpy=True)[0]
            for node in existing_nodes:
                node_type = node.get("entity_type", "")
                if node_type == candidate_type:
                    node_name = node.get("name", "")
                    node_vec = self.embedding_model.encode([node_name], normalize_embeddings=True, convert_to_numpy=True)[0]
                    sim = float(np.dot(candidate_vec, node_vec))

                    if sim >= self.similarity_threshold:
                        logger.info(f"Entity Resolution Merged '{candidate_name}' into '{node_name}' (Cosine Sim: {sim:.3f})")
                        merged_aliases = list(set(node.get("aliases", []) + cand_aliases))
                        return node["id"], node_name, node_type, merged_aliases

        # 3. If no confident match, return candidate as new canonical entity
        return cand_id, cand_canon_name, candidate_type, cand_aliases
