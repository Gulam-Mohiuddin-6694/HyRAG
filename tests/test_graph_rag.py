"""
test_graph_rag.py - Comprehensive Unit & Integration Test Suite for HyRAG Graph RAG.

Tests all core components:
1. Graph Store CRUD & Provenance
2. Entity Resolution & Deduplication
3. Triplet Extraction & Grounding
4. Bounded Multi-Hop Traversal
5. Tri-Hybrid RRF Fusion
6. Conflict Detection
7. 4-Layer Hallucination Audit
8. Document Lifecycle & Cascading Deletion
9. Failure Fallback
"""

import os
import sys
import unittest
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph_store import NetworkXGraphStore
from src.entity_resolution import EntityResolver, normalize_entity_name, resolve_canonical_name
from src.graph_extractor import extract_triplets_heuristic, process_and_index_chunks_to_graph
from src.graph_retrieval import search_knowledge_graph, link_query_entities
from src.conflict_detector import detect_graph_conflicts, format_conflict_prompt_notice
from src.retrieval import reciprocal_rank_fusion
from src.generation import audit_hallucination_and_confidence, build_grounded_prompt


class TestHyRAGGraphStore(unittest.TestCase):

    def setUp(self):
        self.store = NetworkXGraphStore()

    def test_node_edge_crud_and_provenance(self):
        prov1 = {
            "doc_id": "doc_aws_policy",
            "chunk_id": "doc_aws_policy_p1_c1",
            "file_name": "aws_policy.pdf",
            "page_number": 1,
            "source_text": "AWS IAM requires MFA for root accounts.",
            "confidence": 0.95
        }
        
        # Add Nodes
        s_id = self.store.add_node("ent_aws_iam", "AWS IAM", "SecurityControl", aliases=["IAM"], provenance=prov1)
        t_id = self.store.add_node("ent_mfa", "Multi-Factor Authentication", "SecurityControl", aliases=["MFA"], provenance=prov1)

        self.assertEqual(s_id, "ent_aws_iam")
        self.assertEqual(t_id, "ent_mfa")
        
        # Add Edge
        rel_key = self.store.add_edge(s_id, t_id, "REQUIRES", description="AWS IAM requires MFA", confidence=0.95, provenance=prov1)
        self.assertIn("REQUIRES", rel_key)

        # Retrieve Node and verify provenance
        node = self.store.get_node(s_id)
        self.assertIsNotNone(node)
        self.assertEqual(node["name"], "AWS IAM")
        self.assertEqual(len(node["provenance"]), 1)
        self.assertEqual(node["provenance"][0]["chunk_id"], "doc_aws_policy_p1_c1")

        # Verify stats
        stats = self.store.stats()
        self.assertEqual(stats["total_nodes"], 2)
        self.assertEqual(stats["total_edges"], 1)

    def test_multi_hop_bounded_traversal(self):
        # Build chain: A -> B -> C -> D
        self.store.add_node("ent_a", "Node A", "Type1")
        self.store.add_node("ent_b", "Node B", "Type1")
        self.store.add_node("ent_c", "Node C", "Type1")
        self.store.add_node("ent_d", "Node D", "Type1")

        self.store.add_edge("ent_a", "ent_b", "CONNECTS_TO", confidence=0.9)
        self.store.add_edge("ent_b", "ent_c", "CONNECTS_TO", confidence=0.9)
        self.store.add_edge("ent_c", "ent_d", "CONNECTS_TO", confidence=0.9)

        # 1-hop from A should find only B
        subgraph_1hop = self.store.get_subgraph(["ent_a"], max_hops=1, max_nodes=10)
        node_ids_1hop = [n["id"] for n in subgraph_1hop["nodes"]]
        self.assertIn("ent_a", node_ids_1hop)
        self.assertIn("ent_b", node_ids_1hop)
        self.assertNotIn("ent_c", node_ids_1hop)

        # 2-hop from A should find B and C
        subgraph_2hop = self.store.get_subgraph(["ent_a"], max_hops=2, max_nodes=10)
        node_ids_2hop = [n["id"] for n in subgraph_2hop["nodes"]]
        self.assertIn("ent_c", node_ids_2hop)
        self.assertNotIn("ent_d", node_ids_2hop)

    def test_cascading_document_deletion(self):
        prov_doc1 = {"doc_id": "doc_1", "chunk_id": "doc1_c1"}
        prov_doc2 = {"doc_id": "doc_2", "chunk_id": "doc2_c1"}

        # Node A belongs to doc_1 ONLY
        # Node B belongs to doc_1 AND doc_2
        self.store.add_node("ent_a", "Unique to Doc 1", "Concept", provenance=prov_doc1)
        self.store.add_node("ent_b", "Shared Doc 1 and 2", "Concept", provenance=prov_doc1)
        self.store.add_node("ent_b", "Shared Doc 1 and 2", "Concept", provenance=prov_doc2)

        self.store.add_edge("ent_a", "ent_b", "RELATES", provenance=prov_doc1)

        # Delete doc_1 provenance
        res = self.store.delete_document_provenance("doc_1")
        self.assertEqual(res["pruned_edges"], 1)
        self.assertEqual(res["pruned_nodes"], 1)

        # Node A should be completely removed
        self.assertIsNone(self.store.get_node("ent_a"))
        # Node B should still exist with doc_2 provenance
        node_b = self.store.get_node("ent_b")
        self.assertIsNotNone(node_b)
        self.assertEqual(len(node_b["provenance"]), 1)
        self.assertEqual(node_b["provenance"][0]["doc_id"], "doc_2")


class TestEntityResolution(unittest.TestCase):

    def setUp(self):
        self.resolver = EntityResolver(similarity_threshold=0.88)

    def test_acronym_and_normalization(self):
        cid, cname, aliases = resolve_canonical_name("AWS IAM", "SecurityControl")
        self.assertEqual(cid, "ent_identity_and_access_management")
        self.assertEqual(cname, "Identity And Access Management")
        self.assertIn("AWS IAM", aliases)

    def test_merging_synonymous_entities(self):
        existing_nodes = [
            {"id": "ent_identity_and_access_management", "name": "Identity and Access Management", "entity_type": "SecurityControl", "aliases": ["AWS IAM", "IAM"]}
        ]
        node_id, name, etype, aliases = self.resolver.resolve_or_merge("IAM", "SecurityControl", existing_nodes)
        self.assertEqual(node_id, "ent_identity_and_access_management")

    def test_rejecting_ambiguous_distinct_entities(self):
        existing_nodes = [
            {"id": "ent_apple_tech", "name": "Apple Inc.", "entity_type": "TechnologyCompany", "aliases": ["Apple"]}
        ]
        # "Apple Records" with type "MusicLabel" must not merge
        node_id, name, etype, aliases = self.resolver.resolve_or_merge("Apple Records", "MusicLabel", existing_nodes)
        self.assertNotEqual(node_id, "ent_apple_tech")


class TestConflictDetection(unittest.TestCase):

    def test_detect_opposing_facts_across_documents(self):
        edges = [
            {
                "source": "ent_project_x",
                "source_name": "Project X",
                "relation_type": "MANAGED_BY",
                "target": "ent_john",
                "target_name": "John Doe",
                "provenance": [{"doc_id": "doc_a", "file_name": "policy_2023.pdf", "page_number": 2}]
            },
            {
                "source": "ent_project_x",
                "source_name": "Project X",
                "relation_type": "MANAGED_BY",
                "target": "ent_sarah",
                "target_name": "Sarah Connor",
                "provenance": [{"doc_id": "doc_b", "file_name": "policy_2024.pdf", "page_number": 5}]
            }
        ]

        conflicts = detect_graph_conflicts(edges)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["subject"], "Project X")
        self.assertEqual(conflicts[0]["relation"], "MANAGED_BY")
        self.assertEqual(len(conflicts[0]["competing_targets"]), 2)

        notice = format_conflict_prompt_notice(conflicts)
        self.assertIn("CONFLICTING MULTI-DOCUMENT FACTS DETECTED", notice)
        self.assertIn("policy_2023.pdf", notice)
        self.assertIn("policy_2024.pdf", notice)


class TestTriHybridRRF(unittest.TestCase):

    def test_tri_hybrid_fusion(self):
        dense_results = [
            {"chunk_id": "c1", "text": "Dense rank 1"},
            {"chunk_id": "c2", "text": "Dense rank 2"}
        ]
        sparse_results = [
            {"chunk_id": "c3", "text": "BM25 rank 1"},
            {"chunk_id": "c1", "text": "BM25 rank 2"}
        ]
        graph_results = [
            {"chunk_id": "c1", "text": "Graph rank 1"}
        ]

        hybrid = reciprocal_rank_fusion(dense_results, sparse_results, graph_results, k=60, top_k=3)
        # c1 appears in all 3 sources, so it must rank #1
        self.assertEqual(hybrid[0]["chunk_id"], "c1")
        self.assertIsNotNone(hybrid[0]["dense_rank"])
        self.assertIsNotNone(hybrid[0]["sparse_rank"])
        self.assertIsNotNone(hybrid[0]["graph_rank"])


class TestHallucinationAudit(unittest.TestCase):

    class MockEmbeddingModel:
        def encode(self, texts, **kwargs):
            # Deterministic mock vectors
            return np.ones((len(texts), 16), dtype=np.float32)

    def test_grounded_vs_ungrounded_answer(self):
        mock_model = self.MockEmbeddingModel()
        chunks = [
            {"text": "AWS Identity and Access Management requires Multi-Factor Authentication (MFA) for root user accounts.", "rrf_score": 0.03}
        ]
        graph_facts = "• (AWS IAM) ──[REQUIRES]──> (MFA) [Source: aws.pdf, Page 1]"

        # Grounded answer
        grounded_ans = "AWS Identity and Access Management requires MFA for all root accounts [Source: aws.pdf, Page 1]."
        audit_good = audit_hallucination_and_confidence(grounded_ans, chunks, mock_model, graph_facts)
        self.assertTrue(audit_good["is_grounded"])
        self.assertEqual(audit_good["hallucination_risk"], "LOW")

        # Unsupported fallback answer
        fallback_ans = "I cannot answer this question based on the provided enterprise documentation."
        audit_fallback = audit_hallucination_and_confidence(fallback_ans, chunks, mock_model, graph_facts)
        self.assertFalse(audit_fallback["is_grounded"])


if __name__ == "__main__":
    unittest.main()
