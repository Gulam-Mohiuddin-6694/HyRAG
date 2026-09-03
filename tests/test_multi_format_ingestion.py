"""
test_multi_format_ingestion.py - Integration Tests for Multi-Format Ingestion and Pipeline Manager.
"""

import os
import sys
import unittest
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion import (
    extract_text_from_txt_or_md,
    extract_text_from_csv,
    extract_text_from_file
)
from src.pipeline_manager import process_new_upload, delete_document_from_system
from src.embeddings import load_embedding_model
from src.graph_store import NetworkXGraphStore


class TestMultiFormatIngestion(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_txt_and_md(self):
        txt_path = os.path.join(self.temp_dir.name, "security_guidelines.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("AWS Identity and Access Management defines standards for cloud permissions.")

        pages = extract_text_from_txt_or_md(txt_path)
        self.assertEqual(len(pages), 1)
        self.assertIn("Identity and Access Management", pages[0]["text"])
        self.assertEqual(pages[0]["metadata"]["file_type"], "TXT")

    def test_extract_csv(self):
        csv_path = os.path.join(self.temp_dir.name, "roles_catalog.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write("RoleName,Department,MFA_Required\nSecurityAdmin,IT,Yes\nDeveloper,Engineering,Yes\n")

        pages = extract_text_from_csv(csv_path)
        self.assertEqual(len(pages), 1)
        self.assertIn("SecurityAdmin", pages[0]["text"])
        self.assertEqual(pages[0]["metadata"]["file_type"], "CSV")

    def test_end_to_end_upload_and_cascading_delete(self):
        test_content = (
            b"AWS Key Management Service (KMS) provides centralized control over cryptographic keys. "
            b"AWS KMS requires Multi-Factor Authentication for root key destruction."
        )
        file_name = "aws_kms_policy.txt"
        
        graph_store = NetworkXGraphStore()
        # Mock embedding model for fast unit testing
        class MockModel:
            def encode(self, texts, **kwargs):
                import numpy as np
                return np.ones((len(texts), 384), dtype=np.float32)

        model = MockModel()

        # Ingest
        res = process_new_upload(
            file_bytes=test_content,
            file_name=file_name,
            uploaded_by="ADMIN001",
            embedding_model=model,
            graph_store=graph_store,
            raw_docs_dir=self.temp_dir.name,
            faiss_storage_dir=self.temp_dir.name,
            graph_storage_path=os.path.join(self.temp_dir.name, "knowledge_graph.json")
        )

        self.assertEqual(res["status"], "PROCESSED")
        self.assertGreater(res["chunk_count"], 0)
        
        # Verify graph facts created
        stats = graph_store.stats()
        self.assertGreater(stats["total_nodes"], 0)

        # Delete document & cascade
        del_res = delete_document_from_system(
            doc_id=res["doc_id"],
            embedding_model=model,
            graph_store=graph_store,
            raw_docs_dir=self.temp_dir.name,
            faiss_storage_dir=self.temp_dir.name,
            graph_storage_path=os.path.join(self.temp_dir.name, "knowledge_graph.json")
        )

        self.assertEqual(del_res["status"], "DELETED")


if __name__ == "__main__":
    unittest.main()
