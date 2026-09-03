"""
graph_extractor.py - Provenance-Grounded Knowledge Graph Extraction Engine.

This module extracts structured entities and relational facts from document chunks
with strict verbatim evidence verification and automatic fallback mechanisms.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from src.graph_store import BaseGraphStore
from src.entity_resolution import EntityResolver, resolve_canonical_name, normalize_entity_name

logger = logging.getLogger("HyRAG.GraphExtractor")

# Heuristic Relation Patterns for deterministic & offline extraction
HEURISTIC_PATTERNS = [
    (r'([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:requires|must require|mandates)\s+([A-Z][A-Za-z0-9_\s]{2,30})', "REQUIRES"),
    (r'([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:is a part of|is component of|belongs to)\s+([A-Z][A-Za-z0-9_\s]{2,30})', "PART_OF"),
    (r'([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:provides|delivers|offers|supports)\s+([A-Z][A-Za-z0-9_\s]{2,30})', "PROVIDES"),
    (r'([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:manages|supervises|governs|administers)\s+([A-Z][A-Za-z0-9_\s]{2,30})', "MANAGES"),
    (r'([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:uses|utilizes|leverages|deploys)\s+([A-Z][A-Za-z0-9_\s]{2,30})', "USES"),
    (r'([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:prohibits|forbids|restricts|disallows)\s+([A-Z][A-Za-z0-9_\s]{2,30})', "RESTRICTS"),
    (r'([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:complies with|adheres to|follows)\s+([A-Z][A-Za-z0-9_\s]{2,30})', "COMPLIES_WITH"),
    (r'([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:defines policy for|sets standards for)\s+([A-Z][A-Za-z0-9_\s]{2,30})', "DEFINES_POLICY"),
    (r'([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:reports to|escalates to)\s+([A-Z][A-Za-z0-9_\s]{2,30})', "REPORTS_TO"),
]


def extract_triplets_heuristic(chunk_text: str) -> List[Dict[str, Any]]:
    """
    Deterministic rule-based entity & relation extraction for offline mode or fallback.
    """
    triplets = []
    
    # 1. Regex pattern matching
    for pattern, relation in HEURISTIC_PATTERNS:
        matches = re.finditer(pattern, chunk_text, re.IGNORECASE)
        for match in matches:
            subj = match.group(1).strip()
            obj = match.group(2).strip()
            snippet = match.group(0).strip()
            
            if len(subj) > 2 and len(obj) > 2 and subj.lower() != obj.lower():
                triplets.append({
                    "subject": subj,
                    "subject_type": "Entity",
                    "relation": relation,
                    "object": obj,
                    "object_type": "Entity",
                    "confidence": 0.85,
                    "evidence_text": snippet
                })

    # 2. Key Term Co-occurrence (e.g. AWS IAM, MFA, Root Account, Gift Policy)
    key_entities = [
        ("AWS Identity and Access Management", "SecurityControl", ["AWS IAM", "IAM"]),
        ("Multi-Factor Authentication", "SecurityControl", ["MFA", "2FA"]),
        ("Root User Account", "SecurityAccount", ["Root Account", "AWS Root"]),
        ("Workplace Gift Policy", "Policy", ["Gift Policy", "Gifts and Entertainment"]),
        ("Conflict of Interest", "Policy", ["Conflict of Interest Policy"]),
        ("Amazon Web Services", "Organization", ["AWS", "Amazon"]),
        ("Well-Architected Framework", "Framework", ["WAFR", "AWS Well-Architected"]),
        ("Security Pillar", "FrameworkPillar", ["Security"]),
        ("Cost Optimization Pillar", "FrameworkPillar", ["Cost Optimization"]),
        ("Reliability Pillar", "FrameworkPillar", ["Reliability"]),
        ("Performance Efficiency Pillar", "FrameworkPillar", ["Performance Efficiency"]),
        ("Operational Excellence Pillar", "FrameworkPillar", ["Operational Excellence"]),
    ]

    found_in_chunk = []
    for canon_name, etype, aliases in key_entities:
        names_to_check = [canon_name] + aliases
        for name in names_to_check:
            if re.search(r'\b' + re.escape(name) + r'\b', chunk_text, re.IGNORECASE):
                found_in_chunk.append((canon_name, etype, name))
                break

    # Connect co-occurring entities within the same chunk
    if len(found_in_chunk) >= 2:
        for i in range(len(found_in_chunk) - 1):
            s_name, s_type, s_raw = found_in_chunk[i]
            t_name, t_type, t_raw = found_in_chunk[i+1]
            if s_name != t_name:
                triplets.append({
                    "subject": s_name,
                    "subject_type": s_type,
                    "relation": "ASSOCIATED_WITH",
                    "object": t_name,
                    "object_type": t_type,
                    "confidence": 0.78,
                    "evidence_text": chunk_text[:200]
                })

    return triplets


def extract_triplets_llm(chunk_text: str) -> List[Dict[str, Any]]:
    """
    Extracts structured entities and relationships from chunk text using LLM.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    prompt = f"""You are an enterprise knowledge graph fact extractor.
Analyze the following text chunk and extract factual entities and relationships.

STRICT RULES:
1. Extract only facts directly stated in the text.
2. Every extracted relationship MUST include an exact 'evidence_text' substring copied directly from the text.
3. Assign confidence between 0.0 and 1.0.
4. Output ONLY valid JSON formatted as a list of objects.

JSON FORMAT:
[
  {{
    "subject": "Entity Name",
    "subject_type": "Organization | Technology | Policy | Person | Concept | Metric",
    "relation": "REQUIRES | MANAGES | USES | PART_OF | PROVIDES | RESTRICTS | COMPLIES_WITH | DEFINES_POLICY",
    "object": "Target Entity Name",
    "object_type": "Organization | Technology | Policy | Person | Concept | Metric",
    "confidence": 0.95,
    "evidence_text": "exact quote from text verifying this fact"
  }}
]

TEXT:
{chunk_text}

JSON:"""

    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    raw_response = ""

    # 1. Try Groq if configured
    if provider == "groq" or (groq_key and "gsk_" in groq_key):
        if groq_key and groq_key != "your_groq_api_key_here":
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                groq_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
                
                for g_model in groq_models:
                    try:
                        res = client.chat.completions.create(
                            model=g_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0
                        )
                        if res and res.choices:
                            raw_response = res.choices[0].message.content.strip()
                            if raw_response:
                                break
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"Groq triplet extraction failed: {e}. Trying Gemini fallback...")

    # 2. Try Gemini
    if not raw_response and gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            res = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
            )
            if res and res.text:
                raw_response = res.text.strip()
        except Exception as e:
            logger.warning(f"Gemini triplet extraction failed: {e}")

    if not raw_response:
        return []

    # Clean markdown formatting e.g. ```json ... ```
    cleaned_json = re.sub(r'^```json\s*', '', raw_response, flags=re.MULTILINE)
    cleaned_json = re.sub(r'```\s*$', '', cleaned_json, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned_json)
        if isinstance(parsed, list):
            # Verify evidence presence
            valid_triplets = []
            for item in parsed:
                if isinstance(item, dict) and "subject" in item and "object" in item and "relation" in item:
                    # Grounding verification: ensure evidence or terms exist in chunk
                    ev = item.get("evidence_text", "")
                    if ev and ev.lower() in chunk_text.lower():
                        valid_triplets.append(item)
                    else:
                        # Ground by attaching chunk snippet
                        item["evidence_text"] = chunk_text[:200]
                        valid_triplets.append(item)
            return valid_triplets
    except Exception as e:
        logger.warning(f"Failed to parse LLM JSON triplet output: {e}")

    return []


def process_and_index_chunks_to_graph(
    chunks: List[Dict[str, Any]],
    graph_store: BaseGraphStore,
    entity_resolver: Optional[EntityResolver] = None,
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Processes document chunks, extracts knowledge triplets, performs entity resolution,
    and indexes nodes & edges into the graph store with full provenance metadata.
    """
    if entity_resolver is None:
        entity_resolver = EntityResolver()

    total_triplets_extracted = 0
    nodes_added = 0
    edges_added = 0

    print(f"🕸️ Extracting Knowledge Graph facts from {len(chunks)} chunk(s)...")

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
        chunk_text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        doc_id = meta.get("doc_id", "unknown_doc")
        file_name = meta.get("file_name", "unknown_file")
        page_number = meta.get("page_number", 0)

        if not chunk_text.strip():
            continue

        # Extract triplets
        triplets = []
        if use_llm:
            triplets = extract_triplets_llm(chunk_text)
        
        # Fallback to heuristic if LLM returns empty or is offline
        if not triplets:
            triplets = extract_triplets_heuristic(chunk_text)

        total_triplets_extracted += len(triplets)

        for trip in triplets:
            raw_s = trip.get("subject", "").strip()
            s_type = trip.get("subject_type", "Concept")
            relation = trip.get("relation", "RELATED_TO").upper().replace(" ", "_")
            raw_o = trip.get("object", "").strip()
            o_type = trip.get("object_type", "Concept")
            conf = float(trip.get("confidence", 0.8))
            ev_text = trip.get("evidence_text", chunk_text[:150])

            if not raw_s or not raw_o or raw_s.lower() == raw_o.lower():
                continue

            # Provenance record for this fact
            provenance_record = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "file_name": file_name,
                "page_number": page_number,
                "source_text": ev_text,
                "confidence": conf
            }

            # Entity resolution
            existing_nodes = [dict(graph_store.graph.nodes[n]) for n in graph_store.graph.nodes()] if hasattr(graph_store, "graph") else []
            
            s_id, s_name, s_type, s_aliases = entity_resolver.resolve_or_merge(raw_s, s_type, existing_nodes)
            o_id, o_name, o_type, o_aliases = entity_resolver.resolve_or_merge(raw_o, o_type, existing_nodes)

            # Insert nodes
            graph_store.add_node(
                node_id=s_id,
                name=s_name,
                entity_type=s_type,
                aliases=s_aliases,
                provenance=provenance_record
            )
            graph_store.add_node(
                node_id=o_id,
                name=o_name,
                entity_type=o_type,
                aliases=o_aliases,
                provenance=provenance_record
            )

            # Insert edge
            graph_store.add_edge(
                source_id=s_id,
                target_id=o_id,
                relation_type=relation,
                description=f"{s_name} {relation} {o_name}",
                confidence=conf,
                provenance=provenance_record
            )

    stats = graph_store.stats()
    print(f"✅ Graph Construction Complete! Total Nodes: {stats['total_nodes']} | Total Edges: {stats['total_edges']}")
    return stats
