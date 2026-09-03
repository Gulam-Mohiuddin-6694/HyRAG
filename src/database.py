"""
database.py - Persistent SQLite Audit Logging & Document Registry for HyRAG.

Manages persistent storage for:
1. Complete Query History & Latency / Hallucination Audit Logs
2. Document Registry with Ingestion Metadata and Chunk Counts
"""

import os
import json
import sqlite3
import datetime
import logging
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Generator

logger = logging.getLogger("HyRAG.Database")
DB_PATH = os.path.join("storage", "db", "hyrag.db")


@contextmanager
def get_db_cursor() -> Generator[sqlite3.Cursor, None, None]:
    """
    Context manager yielding a SQLite cursor and guaranteeing connection closure.
    """
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def init_db() -> None:
    """Initializes the database schema if not already present."""
    with get_db_cursor() as cursor:
        # 1. Query History Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_role TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            retrieved_docs TEXT,
            confidence_score TEXT,
            hallucination_risk TEXT,
            graph_facts TEXT,
            latency_seconds REAL
        )
        """)

        # 2. Document Registry Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_registry (
            doc_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size_bytes INTEGER DEFAULT 0,
            uploaded_by TEXT NOT NULL,
            upload_timestamp TEXT NOT NULL,
            page_count INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            entity_count INTEGER DEFAULT 0,
            edge_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'PROCESSED'
        )
        """)
        logger.info(f"Initialized HyRAG database schema at '{DB_PATH}'")


# Initialize on import
init_db()


def log_query(
    user_id: str,
    user_role: str,
    query: str,
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
    confidence_score: str,
    hallucination_risk: str,
    graph_facts: str = "",
    latency_seconds: float = 0.0
) -> int:
    """Logs a query execution record into persistent storage."""
    timestamp = datetime.datetime.now().isoformat()
    docs_json = json.dumps(retrieved_docs, ensure_ascii=False)

    with get_db_cursor() as cursor:
        cursor.execute("""
        INSERT INTO query_history (
            user_id, user_role, timestamp, query, answer,
            retrieved_docs, confidence_score, hallucination_risk,
            graph_facts, latency_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, user_role, timestamp, query, answer,
            docs_json, confidence_score, hallucination_risk,
            graph_facts, round(latency_seconds, 3)
        ))
        return cursor.lastrowid or 0


def get_query_history(
    user_id: Optional[str] = None,
    limit: int = 100,
    search_term: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves query history with optional user filtering and search keywords.
    """
    with get_db_cursor() as cursor:
        query = "SELECT * FROM query_history WHERE 1=1"
        params: List[Any] = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if search_term:
            query += " AND (query LIKE ? OR answer LIKE ?)"
            params.extend([f"%{search_term}%", f"%{search_term}%"])

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for r in rows:
            row_dict = dict(r)
            if row_dict.get("retrieved_docs"):
                try:
                    row_dict["retrieved_docs"] = json.loads(row_dict["retrieved_docs"])
                except Exception:
                    pass
            results.append(row_dict)
        return results


def get_analytics_summary() -> Dict[str, Any]:
    """Computes high-level analytics for the Admin Overview Dashboard."""
    with get_db_cursor() as cursor:
        # Total Queries
        cursor.execute("SELECT COUNT(*) FROM query_history")
        total_queries = cursor.fetchone()[0]

        # Queries Today
        today_str = datetime.date.today().isoformat()
        cursor.execute("SELECT COUNT(*) FROM query_history WHERE timestamp LIKE ?", (f"{today_str}%",))
        queries_today = cursor.fetchone()[0]

        # Queries per Employee
        cursor.execute("""
        SELECT user_id, COUNT(*) as query_count 
        FROM query_history 
        GROUP BY user_id 
        ORDER BY query_count DESC
        """)
        queries_per_employee = [dict(r) for r in cursor.fetchall()]

        # Active Users Count
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM query_history")
        active_users = cursor.fetchone()[0]

        # Average Latency
        cursor.execute("SELECT AVG(latency_seconds) FROM query_history")
        avg_latency = cursor.fetchone()[0] or 0.0

        # Total Registered Documents
        cursor.execute("SELECT COUNT(*) FROM document_registry WHERE status != 'DELETED'")
        total_docs = cursor.fetchone()[0]

        return {
            "total_queries": total_queries,
            "queries_today": queries_today,
            "active_users": active_users,
            "avg_latency": round(avg_latency, 2),
            "total_documents": total_docs,
            "queries_per_employee": queries_per_employee
        }


def register_document(
    doc_id: str,
    file_name: str,
    file_path: str,
    file_type: str,
    file_size_bytes: int,
    uploaded_by: str,
    page_count: int = 1,
    chunk_count: int = 0,
    entity_count: int = 0,
    edge_count: int = 0,
    status: str = "PROCESSED"
) -> None:
    """Inserts or updates a document metadata record in the registry."""
    timestamp = datetime.datetime.now().isoformat()
    with get_db_cursor() as cursor:
        cursor.execute("""
        INSERT OR REPLACE INTO document_registry (
            doc_id, file_name, file_path, file_type, file_size_bytes,
            uploaded_by, upload_timestamp, page_count, chunk_count,
            entity_count, edge_count, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id, file_name, file_path, file_type.upper(), file_size_bytes,
            uploaded_by, timestamp, page_count, chunk_count,
            entity_count, edge_count, status
        ))


def get_all_documents() -> List[Dict[str, Any]]:
    """Retrieves all active documents from the registry."""
    with get_db_cursor() as cursor:
        cursor.execute("SELECT * FROM document_registry WHERE status != 'DELETED' ORDER BY upload_timestamp DESC")
        return [dict(r) for r in cursor.fetchall()]


def get_document_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single document record by its doc_id."""
    with get_db_cursor() as cursor:
        cursor.execute("SELECT * FROM document_registry WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def mark_document_deleted(doc_id: str) -> None:
    """Sets document status to DELETED."""
    with get_db_cursor() as cursor:
        cursor.execute("UPDATE document_registry SET status = 'DELETED' WHERE doc_id = ?", (doc_id,))
