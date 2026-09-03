"""
test_auth_and_db.py - Unit Tests for HyRAG Authentication & SQLite Database Modules.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.auth import authenticate_user, is_admin, list_all_users, USER_DATABASE
from src.database import (
    log_query,
    get_query_history,
    get_analytics_summary,
    register_document,
    get_all_documents,
    get_document_by_id,
    mark_document_deleted
)


class TestAuthentication(unittest.TestCase):

    def test_all_ten_dummy_accounts_authenticate_successfully(self):
        expected_accounts = [
            ("EMP1001", "HyRAG@1001", "Employee"),
            ("EMP1002", "HyRAG@1002", "Employee"),
            ("EMP1003", "HyRAG@1003", "Employee"),
            ("EMP1004", "HyRAG@1004", "Employee"),
            ("EMP1005", "HyRAG@1005", "Employee"),
            ("EMP1006", "HyRAG@1006", "Employee"),
            ("EMP1007", "HyRAG@1007", "Employee"),
            ("EMP1008", "HyRAG@1008", "Employee"),
            ("EMP1009", "HyRAG@1009", "Employee"),
            ("ADMIN001", "HyRAG@Admin01", "Admin")
        ]

        self.assertEqual(len(USER_DATABASE), 10)

        for uid, pwd, expected_role in expected_accounts:
            user = authenticate_user(uid, pwd)
            self.assertIsNotNone(user, f"Authentication failed for {uid}")
            self.assertEqual(user["user_id"], uid)
            self.assertEqual(user["role"], expected_role)
            self.assertNotIn("password_hash", user)

    def test_invalid_password_and_unknown_user(self):
        # Invalid password
        self.assertIsNone(authenticate_user("EMP1001", "WrongPassword!"))
        # Unknown user
        self.assertIsNone(authenticate_user("UNKNOWN999", "HyRAG@1001"))

    def test_role_enforcement(self):
        self.assertTrue(is_admin("ADMIN001"))
        self.assertFalse(is_admin("EMP1001"))
        self.assertFalse(is_admin("EMP1005"))


class TestDatabasePersistence(unittest.TestCase):

    def test_query_logging_and_retrieval(self):
        test_uid = "EMP1001"
        test_query = "What is the MFA policy for AWS IAM?"
        test_answer = "AWS IAM requires MFA for root accounts [Source: aws.pdf, Page 1]."
        test_sources = [{"file_name": "aws.pdf", "page_number": 1}]

        row_id = log_query(
            user_id=test_uid,
            user_role="Employee",
            query=test_query,
            answer=test_answer,
            retrieved_docs=test_sources,
            confidence_score="92.5%",
            hallucination_risk="LOW",
            graph_facts="• (AWS IAM) -> [REQUIRES] -> (MFA)",
            latency_seconds=0.45
        )
        self.assertGreater(row_id, 0)

        # Retrieve
        history = get_query_history(user_id=test_uid, limit=10)
        self.assertGreater(len(history), 0)
        latest = history[0]
        self.assertEqual(latest["user_id"], test_uid)
        self.assertEqual(latest["query"], test_query)
        self.assertEqual(latest["confidence_score"], "92.5%")

    def test_document_registry_lifecycle(self):
        doc_id = "doc_test_policy_01"
        register_document(
            doc_id=doc_id,
            file_name="test_policy.pdf",
            file_path="/data/test_policy.pdf",
            file_type="PDF",
            file_size_bytes=1024,
            uploaded_by="ADMIN001",
            page_count=2,
            chunk_count=4,
            entity_count=8,
            edge_count=6,
            status="PROCESSED"
        )

        doc = get_document_by_id(doc_id)
        self.assertIsNotNone(doc)
        self.assertEqual(doc["file_name"], "test_policy.pdf")
        self.assertEqual(doc["uploaded_by"], "ADMIN001")

        # Mark deleted
        mark_document_deleted(doc_id)
        all_active = get_all_documents()
        self.assertFalse(any(d["doc_id"] == doc_id for d in all_active))


if __name__ == "__main__":
    unittest.main()
