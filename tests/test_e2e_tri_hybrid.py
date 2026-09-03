"""
test_e2e_tri_hybrid.py - End-to-End Live Integration Test for HyRAG Tri-Hybrid Retrieval.
"""

import os
import sys
import unittest
import json
import faiss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings import load_embedding_model
from src.retrieval import build_bm25_index, tri_hybrid_search
from src.graph_store import NetworkXGraphStore
from src.generation import build_grounded_prompt, audit_hallucination_and_confidence


class TestE2ETriHybrid(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.faiss_dir = os.path.join("storage", "faiss_index")
        cls.index_path = os.path.join(cls.faiss_dir, "index.faiss")
        cls.metadata_path = os.path.join(cls.faiss_dir, "chunks_metadata.json")
        cls.graph_path = os.path.join("storage", "graph_store", "knowledge_graph.json")

        assert os.path.exists(cls.index_path), "FAISS index missing! Run build_graph.py"
        assert os.path.exists(cls.metadata_path), "Chunks metadata missing! Run build_graph.py"
        assert os.path.exists(cls.graph_path), "Graph store missing! Run build_graph.py"

        cls.faiss_index = faiss.read_index(cls.index_path)
        with open(cls.metadata_path, "r", encoding="utf-8") as f:
            cls.chunks = json.load(f)

        cls.model = load_embedding_model()
        cls.bm25, _ = build_bm25_index(cls.chunks)
        cls.graph_store = NetworkXGraphStore()
        cls.graph_store.load(cls.graph_path)

    def test_e2e_query_execution(self):
        query = "What is the AWS policy on multi-factor authentication (MFA) for root accounts?"
        
        # 1. Tri-Hybrid Search
        hybrid_chunks, graph_res = tri_hybrid_search(
            query=query,
            model=self.model,
            faiss_index=self.faiss_index,
            bm25_index=self.bm25,
            chunks=self.chunks,
            graph_store=self.graph_store,
            top_k_chunks=3,
            rrf_k=60,
            max_hops=2,
            min_relation_confidence=0.60
        )

        self.assertGreater(len(hybrid_chunks), 0)
        self.assertIn("subgraph", graph_res)
        self.assertGreater(len(graph_res["subgraph"]["nodes"]), 0)

        # 2. Build Grounded Prompt
        prompt = build_grounded_prompt(
            query=query,
            retrieved_chunks=hybrid_chunks,
            graph_facts=graph_res.get("facts_summary", "")
        )

        self.assertIn("VERIFIED KNOWLEDGE GRAPH FACTS", prompt)
        self.assertIn("DOCUMENT CONTEXT", prompt)
        self.assertIn(query, prompt)

        # 3. Audit Grounding
        sample_answer = "AWS Identity and Access Management requires Multi-Factor Authentication (MFA) for all root user accounts [Source: aws_iam_mfa_policy.pdf, Page 1]."
        audit = audit_hallucination_and_confidence(
            answer=sample_answer,
            retrieved_chunks=hybrid_chunks,
            embedding_model=self.model,
            graph_facts=graph_res.get("facts_summary", "")
        )

        self.assertTrue(audit["is_grounded"])
        self.assertEqual(audit["hallucination_risk"], "LOW")
        print(f"\n[E2E TEST PASSED] Audit Result: {audit}")


if __name__ == "__main__":
    unittest.main()
