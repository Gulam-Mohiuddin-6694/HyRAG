"""
graph_store.py - Persistent Knowledge Graph Storage Engine for HyRAG.

This module provides an extensible, provenance-first Graph Store implementation
built on NetworkX with JSON persistence, multi-hop sub-graph queries, and
cascading document lifecycle pruning.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx

logger = logging.getLogger("HyRAG.GraphStore")


class BaseGraphStore(ABC):
    """Abstract interface for HyRAG graph stores."""

    @abstractmethod
    def add_node(
        self,
        node_id: str,
        name: str,
        entity_type: str,
        aliases: Optional[List[str]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        **attrs
    ) -> str:
        """Adds or updates an entity node with source provenance."""
        pass

    @abstractmethod
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        description: Optional[str] = None,
        confidence: float = 1.0,
        provenance: Optional[Dict[str, Any]] = None,
        **attrs
    ) -> str:
        """Adds or updates a directed relationship edge with source provenance."""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves node data by canonical node_id."""
        pass

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        max_hops: int = 1,
        min_confidence: float = 0.0,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Finds neighboring nodes up to max_hops away."""
        pass

    @abstractmethod
    def get_subgraph(
        self,
        seed_node_ids: List[str],
        max_hops: int = 2,
        max_nodes: int = 25,
        min_confidence: float = 0.0
    ) -> Dict[str, Any]:
        """Extracts bounded contextual subgraph around seed entities."""
        pass

    @abstractmethod
    def delete_document_provenance(self, doc_id: str) -> Dict[str, int]:
        """Cascades deletion of document facts and prunes orphan nodes/edges."""
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        """Persists the graph index to disk."""
        pass

    @abstractmethod
    def load(self, file_path: str) -> bool:
        """Loads graph index from disk."""
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Returns statistics of the current graph."""
        pass


class NetworkXGraphStore(BaseGraphStore):
    """
    In-memory NetworkX MultiDiGraph implementation with atomic JSON persistence,
    strict multi-source provenance tracking, and bounded multi-hop traversals.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.alias_to_id: Dict[str, str] = {}  # Normalized alias/name -> node_id

    def _normalize_name(self, name: str) -> str:
        return " ".join(name.strip().lower().split())

    def add_node(
        self,
        node_id: str,
        name: str,
        entity_type: str,
        aliases: Optional[List[str]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        **attrs
    ) -> str:
        norm_name = self._normalize_name(name)
        if not node_id:
            node_id = f"ent_{norm_name.replace(' ', '_').replace('-', '_')}"

        if self.graph.has_node(node_id):
            node_data = self.graph.nodes[node_id]
            # Merge aliases
            existing_aliases = set(node_data.get("aliases", []))
            if aliases:
                for a in aliases:
                    existing_aliases.add(a)
                    self.alias_to_id[self._normalize_name(a)] = node_id
            node_data["aliases"] = list(existing_aliases)

            # Append provenance record if unique
            if provenance:
                prov_list = node_data.get("provenance", [])
                chunk_id = provenance.get("chunk_id")
                if not any(p.get("chunk_id") == chunk_id for p in prov_list if chunk_id):
                    prov_list.append(provenance)
                node_data["provenance"] = prov_list

            # Update other attributes
            for k, v in attrs.items():
                if k not in node_data:
                    node_data[k] = v
        else:
            all_aliases = set(aliases or [])
            all_aliases.add(name)
            prov_list = [provenance] if provenance else []

            self.graph.add_node(
                node_id,
                id=node_id,
                name=name,
                entity_type=entity_type,
                aliases=list(all_aliases),
                provenance=prov_list,
                **attrs
            )

            for a in all_aliases:
                self.alias_to_id[self._normalize_name(a)] = node_id

        self.alias_to_id[norm_name] = node_id
        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        description: Optional[str] = None,
        confidence: float = 1.0,
        provenance: Optional[Dict[str, Any]] = None,
        **attrs
    ) -> str:
        if not self.graph.has_node(source_id):
            raise ValueError(f"Source node '{source_id}' does not exist in graph.")
        if not self.graph.has_node(target_id):
            raise ValueError(f"Target node '{target_id}' does not exist in graph.")

        rel_key = f"{source_id}->{relation_type}->{target_id}"

        # Check if identical directed relation exists
        edge_found = False
        if self.graph.has_edge(source_id, target_id):
            for k, edge_data in self.graph.get_edge_data(source_id, target_id).items():
                if edge_data.get("relation_type") == relation_type:
                    edge_found = True
                    # Update edge confidence (keep maximum)
                    edge_data["confidence"] = max(edge_data.get("confidence", 0.0), confidence)
                    if description and not edge_data.get("description"):
                        edge_data["description"] = description
                    if provenance:
                        prov_list = edge_data.get("provenance", [])
                        chunk_id = provenance.get("chunk_id")
                        if not any(p.get("chunk_id") == chunk_id for p in prov_list if chunk_id):
                            prov_list.append(provenance)
                        edge_data["provenance"] = prov_list
                    break

        if not edge_found:
            prov_list = [provenance] if provenance else []
            self.graph.add_edge(
                source_id,
                target_id,
                key=rel_key,
                id=rel_key,
                relation_type=relation_type,
                description=description or f"{source_id} {relation_type} {target_id}",
                confidence=float(confidence),
                provenance=prov_list,
                **attrs
            )

        return rel_key

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if self.graph.has_node(node_id):
            return dict(self.graph.nodes[node_id])
        return None

    def find_node_id_by_alias(self, query_name: str) -> Optional[str]:
        norm = self._normalize_name(query_name)
        return self.alias_to_id.get(norm)

    def get_neighbors(
        self,
        node_id: str,
        max_hops: int = 1,
        min_confidence: float = 0.0,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        if not self.graph.has_node(node_id):
            return []

        visited = {node_id}
        current_level = {node_id}
        result_nodes = []

        for _ in range(max_hops):
            next_level = set()
            for current in current_level:
                # Outgoing
                if direction in ("out", "both"):
                    for _, neighbor, data in self.graph.out_edges(current, data=True):
                        if data.get("confidence", 1.0) >= min_confidence and neighbor not in visited:
                            visited.add(neighbor)
                            next_level.add(neighbor)
                            node_data = self.get_node(neighbor)
                            if node_data:
                                result_nodes.append(node_data)
                # Incoming
                if direction in ("in", "both"):
                    for neighbor, _, data in self.graph.in_edges(current, data=True):
                        if data.get("confidence", 1.0) >= min_confidence and neighbor not in visited:
                            visited.add(neighbor)
                            next_level.add(neighbor)
                            node_data = self.get_node(neighbor)
                            if node_data:
                                result_nodes.append(node_data)

            current_level = next_level
            if not current_level:
                break

        return result_nodes

    def get_subgraph(
        self,
        seed_node_ids: List[str],
        max_hops: int = 2,
        max_nodes: int = 25,
        min_confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Extracts bounded subgraph around seed nodes including connected edges and provenance.
        """
        valid_seeds = [s for s in seed_node_ids if self.graph.has_node(s)]
        if not valid_seeds:
            return {"nodes": [], "edges": [], "supporting_chunk_ids": [], "paths": []}

        subgraph_nodes: Set[str] = set()
        queue = [(s, 0) for s in valid_seeds]
        visited_hops = {s: 0 for s in valid_seeds}

        while queue and len(subgraph_nodes) < max_nodes:
            current, hops = queue.pop(0)
            subgraph_nodes.add(current)

            if hops < max_hops:
                # Check outgoing
                for _, neighbor, edge_data in self.graph.out_edges(current, data=True):
                    if edge_data.get("confidence", 1.0) >= min_confidence:
                        if neighbor not in visited_hops or visited_hops[neighbor] > hops + 1:
                            visited_hops[neighbor] = hops + 1
                            queue.append((neighbor, hops + 1))

                # Check incoming
                for neighbor, _, edge_data in self.graph.in_edges(current, data=True):
                    if edge_data.get("confidence", 1.0) >= min_confidence:
                        if neighbor not in visited_hops or visited_hops[neighbor] > hops + 1:
                            visited_hops[neighbor] = hops + 1
                            queue.append((neighbor, hops + 1))

        # Build node and edge response payloads
        nodes_list = []
        for nid in list(subgraph_nodes)[:max_nodes]:
            ndata = dict(self.graph.nodes[nid])
            nodes_list.append(ndata)

        edges_list = []
        supporting_chunks: Set[str] = set()
        node_set = set(n["id"] for n in nodes_list)

        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if u in node_set and v in node_set:
                if data.get("confidence", 1.0) >= min_confidence:
                    edge_dict = dict(data)
                    edge_dict["source"] = u
                    edge_dict["target"] = v
                    edge_dict["source_name"] = self.graph.nodes[u].get("name", u)
                    edge_dict["target_name"] = self.graph.nodes[v].get("name", v)
                    edges_list.append(edge_dict)

                    for prov in edge_dict.get("provenance", []):
                        if "chunk_id" in prov:
                            supporting_chunks.add(prov["chunk_id"])

        for n in nodes_list:
            for prov in n.get("provenance", []):
                if "chunk_id" in prov:
                    supporting_chunks.add(prov["chunk_id"])

        # Construct readable relational path strings
        paths = []
        for e in edges_list:
            s_name = e.get("source_name", e["source"])
            r_type = e.get("relation_type", "RELATED_TO")
            t_name = e.get("target_name", e["target"])
            conf = e.get("confidence", 1.0)
            
            # Format primary provenance tag
            prov_tag = ""
            if e.get("provenance"):
                p0 = e["provenance"][0]
                fname = p0.get("file_name", "doc")
                page = p0.get("page_number", 0)
                prov_tag = f" [Source: {fname}, Page {page}]"

            paths.append(f"({s_name}) ──[{r_type}]──> ({t_name}) (Confidence: {conf:.2f}){prov_tag}")

        return {
            "nodes": nodes_list,
            "edges": edges_list,
            "supporting_chunk_ids": list(supporting_chunks),
            "paths": paths
        }

    def delete_document_provenance(self, doc_id: str) -> Dict[str, int]:
        """
        Removes all provenance links belonging to doc_id.
        Prunes edges and nodes that have zero remaining provenance records.
        """
        pruned_edges = 0
        pruned_nodes = 0

        # 1. Clean Edges
        edges_to_remove = []
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            prov_list = data.get("provenance", [])
            filtered_prov = [p for p in prov_list if p.get("doc_id") != doc_id]
            if len(filtered_prov) == 0:
                edges_to_remove.append((u, v, k))
            else:
                data["provenance"] = filtered_prov

        for u, v, k in edges_to_remove:
            self.graph.remove_edge(u, v, key=k)
            pruned_edges += 1

        # 2. Clean Nodes
        nodes_to_remove = []
        for n, data in self.graph.nodes(data=True):
            prov_list = data.get("provenance", [])
            filtered_prov = [p for p in prov_list if p.get("doc_id") != doc_id]
            if len(filtered_prov) == 0 and self.graph.degree(n) == 0:
                nodes_to_remove.append(n)
            else:
                data["provenance"] = filtered_prov

        for n in nodes_to_remove:
            # Clean alias lookup table
            node_aliases = self.graph.nodes[n].get("aliases", [])
            for a in node_aliases:
                norm = self._normalize_name(a)
                if self.alias_to_id.get(norm) == n:
                    del self.alias_to_id[norm]
            self.graph.remove_node(n)
            pruned_nodes += 1

        return {"pruned_edges": pruned_edges, "pruned_nodes": pruned_nodes}

    def save(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        nodes_data = {n: dict(data) for n, data in self.graph.nodes(data=True)}
        
        edges_data = []
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            e_dict = dict(data)
            e_dict["source"] = u
            e_dict["target"] = v
            e_dict["key"] = k
            edges_data.append(e_dict)

        payload = {
            "version": "1.0",
            "nodes": nodes_data,
            "edges": edges_data,
            "alias_to_id": self.alias_to_id
        }

        temp_path = f"{file_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # Atomic replacement
        if os.path.exists(file_path):
            os.replace(temp_path, file_path)
        else:
            os.rename(temp_path, file_path)

        logger.info(f"Knowledge Graph saved to '{file_path}' ({len(nodes_data)} nodes, {len(edges_data)} edges)")

    def load(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.graph.clear()
            self.alias_to_id.clear()

            nodes_data = payload.get("nodes", {})
            for nid, ndata in nodes_data.items():
                self.graph.add_node(nid, **ndata)

            edges_data = payload.get("edges", [])
            for edata in edges_data:
                u = edata.pop("source")
                v = edata.pop("target")
                key = edata.pop("key", f"{u}->{edata.get('relation_type')}->{v}")
                self.graph.add_edge(u, v, key=key, **edata)

            self.alias_to_id = payload.get("alias_to_id", {})
            return True
        except Exception as e:
            logger.error(f"Failed to load Knowledge Graph from '{file_path}': {e}")
            return False

    def stats(self) -> Dict[str, Any]:
        node_types: Dict[str, int] = {}
        for _, d in self.graph.nodes(data=True):
            t = d.get("entity_type", "Unknown")
            node_types[t] = node_types.get(t, 0) + 1

        rel_types: Dict[str, int] = {}
        for _, _, d in self.graph.edges(data=True):
            r = d.get("relation_type", "RELATED_TO")
            rel_types[r] = rel_types.get(r, 0) + 1

        doc_ids = set()
        for _, d in self.graph.nodes(data=True):
            for p in d.get("provenance", []):
                if "doc_id" in p:
                    doc_ids.add(p["doc_id"])
        for _, _, d in self.graph.edges(data=True):
            for p in d.get("provenance", []):
                if "doc_id" in p:
                    doc_ids.add(p["doc_id"])

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "entity_types": node_types,
            "relation_types": rel_types,
            "total_documents": len(doc_ids)
        }
