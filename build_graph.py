"""
build_graph.py - End-to-End Indexing & Knowledge Graph Builder for HyRAG.

This script runs ingestion, token chunking, FAISS vector embeddings, and
Knowledge Graph extraction & indexing for all documents in 'data/raw_documents'.
"""

import os
import sys
import json
import logging

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.pdf_generator import create_simple_pdf
from src.ingestion import ingest_directory
from src.chunking import chunk_dataset
from src.embeddings import load_embedding_model, generate_chunk_embeddings, build_faiss_index, save_faiss_index
from src.graph_store import NetworkXGraphStore
from src.graph_extractor import process_and_index_chunks_to_graph
from src.entity_resolution import EntityResolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HyRAG.Builder")


def create_sample_documents_if_needed(docs_dir: str):
    """
    Creates sample enterprise PDF policy documents if the raw_documents directory is empty.
    """
    os.makedirs(docs_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        logger.info(f"No PDFs found in '{docs_dir}'. Generating enterprise sample PDFs...")

        # Document 1: AWS IAM Security & MFA Policy
        doc1_path = os.path.join(docs_dir, "aws_iam_mfa_policy.pdf")
        sections1 = [
            ("Section 1: Multi-Factor Authentication (MFA) Requirements", 
             "AWS Identity and Access Management requires Multi-Factor Authentication (MFA) for all root user accounts and administrative roles. Root user credentials provide unrestricted access to all AWS resources and billing information. Therefore, hardware or virtual MFA devices must be enabled immediately upon account creation."),
            ("Section 2: Password and Access Key Hygiene",
             "IAM users must rotate programmatic access keys every 90 days. Unused access keys must be disabled after 30 days of inactivity. Root account access keys must never be generated or retained for daily operational tasks."),
            ("Section 3: Role-Based Access Control (RBAC)",
             "All human users and automated workloads must leverage IAM Roles with temporary credentials provided by AWS Security Token Service (STS) instead of long-term credentials.")
        ]
        create_simple_pdf(doc1_path, "AWS Identity and Access Management Policy", sections1)

        # Document 2: Amazon Workplace Code of Conduct & Gift Policy
        doc2_path = os.path.join(docs_dir, "amazon_code_of_conduct_gifts.pdf")
        sections2 = [
            ("Section 4: Workplace Gifts and Entertainment",
             "Amazon employees must not accept gifts, favors, meals, or entertainment from suppliers, customers, or partners that could influence, or appear to influence, business decisions. Nominal promotional items under $50 in value may be accepted if infrequent and customary."),
            ("Section 5: Conflict of Interest Policy",
             "Employees must avoid any relationship, activity, or financial interest that creates a conflict of interest between personal affairs and Amazon responsibilities. Any outside employment or commercial board appointment requires prior written approval from Legal and Compliance.")
        ]
        create_simple_pdf(doc2_path, "Amazon Global Code of Business Conduct and Ethics", sections2)

        # Document 3: AWS Well-Architected Framework Security Pillar
        doc3_path = os.path.join(docs_dir, "aws_well_architected_security_pillar.pdf")
        sections3 = [
            ("Overview of the Security Pillar",
             "The Security Pillar of the AWS Well-Architected Framework encompasses the ability to protect data, systems, and assets to take advantage of cloud technologies to improve your security posture."),
            ("Core Security Principles",
             "1. Apply security at all layers (defense in depth). 2. Protect data in transit and at rest using encryption. 3. Enable traceability and automated auditing through AWS CloudTrail and Amazon CloudWatch. 4. Keep people away from data using automated tools and IAM policies.")
        ]
        create_simple_pdf(doc3_path, "AWS Well-Architected Framework: Security Pillar", sections3)

        logger.info(f"Generated 3 enterprise sample PDF documents in '{docs_dir}'.")


def run_pipeline():
    raw_docs_dir = os.path.join("data", "raw_documents")
    faiss_dir = os.path.join("storage", "faiss_index")
    graph_path = os.path.join("storage", "graph_store", "knowledge_graph.json")

    create_sample_documents_if_needed(raw_docs_dir)

    print("==========================================================")
    print("STARTING FULL HyRAG VECTOR + GRAPH PIPELINE BUILD")
    print("==========================================================")

    # 1. Ingestion
    print("\n--- STAGE 1: DOCUMENT INGESTION ---")
    pages = ingest_directory(raw_docs_dir)
    if not pages:
        print("No pages ingested. Ensure PDF files exist in data/raw_documents.")
        return

    # 2. Token-Aware Chunking
    print("\n--- STAGE 2: TOKEN-AWARE CHUNKING ---")
    chunks = chunk_dataset(pages, chunk_size=256, chunk_overlap=32)

    # 3. Dense Vector Embeddings & FAISS Index
    print("\n--- STAGE 3: EMBEDDINGS & FAISS INDEX BUILD ---")
    model = load_embedding_model()
    embeddings = generate_chunk_embeddings(chunks, model, batch_size=64)
    faiss_index = build_faiss_index(embeddings)
    save_faiss_index(faiss_index, chunks, output_dir=faiss_dir)

    # 4. Knowledge Graph Construction
    print("\n--- STAGE 4: KNOWLEDGE GRAPH EXTRACTION & INDEXING ---")
    graph_store = NetworkXGraphStore()
    entity_resolver = EntityResolver(similarity_threshold=0.88, embedding_model=model)
    
    # Process chunks into knowledge graph
    process_and_index_chunks_to_graph(
        chunks=chunks,
        graph_store=graph_store,
        entity_resolver=entity_resolver,
        use_llm=True
    )

    # Save Graph
    graph_store.save(graph_path)

    stats = graph_store.stats()
    print("\n==========================================================")
    print("HyRAG VECTOR & GRAPH PIPELINE SUCCESSFULLY BUILT!")
    print(f"Total Chunks: {len(chunks)}")
    print(f"Total Entities (Nodes): {stats['total_nodes']}")
    print(f"Total Relationships (Edges): {stats['total_edges']}")
    print(f"Vector Index: {faiss_dir}")
    print(f"Graph Store: {graph_path}")
    print("==========================================================")


if __name__ == "__main__":
    run_pipeline()
