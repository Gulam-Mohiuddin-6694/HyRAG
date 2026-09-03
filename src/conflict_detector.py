"""
conflict_detector.py - Multi-Document Contradiction & Fact Conflict Detection Engine for HyRAG.

This module inspects retrieved knowledge graph relations and supporting chunks
to identify conflicting claims across different documents (e.g. diverging policies,
ownership, or management).
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("HyRAG.ConflictDetector")


def detect_graph_conflicts(subgraph_edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Scans subgraph edges for conflicting multi-document facts sharing the same
    (subject, relation_type) but pointing to distinct targets or opposing values.
    
    Returns:
        List of conflict dictionaries detailing opposing claims and their document provenance.
    """
    # Group edges by (source, relation_type)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    
    # Relations where multiplicity typically indicates conflicting information
    functional_relations = {
        "MANAGED_BY", "REPORTS_TO", "OWNS", "HEADED_BY", "LEAD_BY",
        "DEFINES_LIMIT", "EFFECTIVE_DATE", "SUPERVISED_BY", "MAXIMUM_VALUE"
    }

    for edge in subgraph_edges:
        src = edge.get("source")
        rel = edge.get("relation_type", "")
        tgt = edge.get("target")
        
        key = (src, rel)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(edge)

    conflicts = []
    for (src, rel), edges in grouped.items():
        if len(edges) > 1:
            # Check if targets differ across different documents
            distinct_targets = set(e.get("target") for e in edges)
            if len(distinct_targets) > 1:
                # Group provenance by target
                target_provenance = {}
                for e in edges:
                    t_name = e.get("target_name", e.get("target"))
                    prov_list = e.get("provenance", [])
                    target_provenance[t_name] = prov_list

                src_name = edges[0].get("source_name", src)
                conflicts.append({
                    "subject": src_name,
                    "relation": rel,
                    "competing_targets": list(distinct_targets),
                    "details": target_provenance,
                    "warning": f"Conflicting information detected: '{src_name}' has multiple contradictory '{rel}' relationships across documents: {list(target_provenance.keys())}"
                })

    return conflicts


def format_conflict_prompt_notice(conflicts: List[Dict[str, Any]]) -> str:
    """
    Formats detected conflicts into a prominent warning block for the LLM prompt.
    """
    if not conflicts:
        return ""

    lines = ["\n⚠️ ATTENTION: CONFLICTING MULTI-DOCUMENT FACTS DETECTED:"]
    for c in conflicts:
        lines.append(f"- {c['warning']}")
        for target, prov_list in c["details"].items():
            if prov_list:
                p = prov_list[0]
                lines.append(f"   • Claims '{target}': [Source: {p.get('file_name', 'Doc')}, Page {p.get('page_number', '?')}]")
    lines.append("INSTRUCTION: You MUST explicitly inform the user of this contradiction and cite both sources. Do NOT arbitrarily choose one over the other.")
    return "\n".join(lines) + "\n"
