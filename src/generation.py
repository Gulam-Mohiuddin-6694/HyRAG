"""
generation.py - Answer Generation, Grounding Verification & Hallucination Audit for HyRAG.

This module formats tri-hybrid retrieval contexts (chunks + knowledge graph facts),
calls LLM APIs (Gemini/Groq), runs a 4-layer hallucination audit, and calculates
verified confidence scores.
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import numpy as np

# Load environment variables from .env
load_dotenv()
logger = logging.getLogger("HyRAG.Generation")


def build_grounded_prompt(
    query: str, 
    retrieved_chunks: List[Dict[str, Any]],
    graph_facts: Optional[str] = None,
    conflict_notice: Optional[str] = None
) -> str:
    """
    Constructs a strictly retrieval-grounded system prompt combining
    document text chunks and validated knowledge graph relational facts.

    Args:
        query (str): User question.
        retrieved_chunks (List[Dict[str, Any]]): Top-K hybrid chunks.
        graph_facts (Optional[str]): Formatted knowledge graph relational facts.
        conflict_notice (Optional[str]): Contradiction/conflict warnings if detected.

    Returns:
        str: Fully formatted prompt ready for LLM inference.
    """
    context_str = ""
    for idx, c in enumerate(retrieved_chunks):
        file_name = c["metadata"].get("file_name", "Unknown Document")
        page_num = c["metadata"].get("page_number", 0)
        chunk_text = c.get("text", "").strip()

        context_str += f"\n[Document #{idx+1} | Source: {file_name}, Page {page_num}]\n"
        context_str += f"{chunk_text}\n"

    graph_section = ""
    if graph_facts and graph_facts.strip() and "No direct relational facts" not in graph_facts:
        graph_section = f"\nVERIFIED KNOWLEDGE GRAPH FACTS:\n{graph_facts.strip()}\n"

    conflict_section = f"\n{conflict_notice.strip()}\n" if conflict_notice and conflict_notice.strip() else ""

    prompt = f"""You are an enterprise AI assistant powered by HyRAG (Hybrid Graph RAG).
Answer the user's question STRICTLY using ONLY the provided verified facts and document context below.

STRICT INSTRUCTIONS:
1. Base your answer ONLY on the facts explicitly mentioned in the CONTEXT and VERIFIED KNOWLEDGE GRAPH FACTS.
2. Do NOT use outside knowledge, prior assumptions, or extrapolate beyond the text.
3. If CONFLICTING FACTS are detected, you MUST explicitly state the discrepancy and cite both sources.
4. If the CONTEXT does not contain sufficient information to answer the question, state:
   "I cannot answer this question based on the provided enterprise documentation."
5. Include inline citations for every factual claim using the format [Source: filename.pdf, Page X].
6. Keep your response concise, factual, structured, and professional.
{graph_section}{conflict_section}
DOCUMENT CONTEXT:
{context_str}

USER QUESTION: {query}

ANSWER WITH CITATIONS:"""
    
    return prompt


def generate_llm_answer(prompt: str) -> str:
    """
    Calls configured LLM API (Groq / Gemini) using credentials from .env.

    Args:
        prompt (str): Grounded prompt string.

    Returns:
        str: Generated LLM response text.
    """
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    # 1. Try Groq if provider is groq or if groq key is valid
    if provider == "groq" or (groq_key and "gsk_" in groq_key):
        if groq_key and groq_key != "your_groq_api_key_here":
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                groq_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
                
                for g_model in groq_models:
                    try:
                        response = client.chat.completions.create(
                            model=g_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1
                        )
                        if response and response.choices:
                            return response.choices[0].message.content.strip()
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"Groq API attempt failed: {e}. Trying Gemini fallback...")

    # 2. Try Google Gemini API
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini API attempt failed: {e}")

    # Fallback if no valid API key is present
    return "⚠️ Please set a valid GROQ_API_KEY or GEMINI_API_KEY in your .env file to enable live LLM generation."


def audit_hallucination_and_confidence(
    answer: str, 
    retrieved_chunks: List[Dict[str, Any]], 
    embedding_model: Any,
    graph_facts: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs a 4-layer Hallucination Audit & Confidence Calculation on the generated answer.

    Layer 1: Rule-Based N-Gram Word Overlap Grounding
    Layer 2: Semantic Embedding Cosine Similarity (Answer vs. Evidence)
    Layer 3: Graph Relational Fact Grounding Check
    Layer 4: Composite Tri-Hybrid Confidence Score

    Returns:
        Dict[str, Any]: Detailed audit report with confidence score percentage.
    """
    if "cannot answer" in answer.lower() or "please set a valid" in answer.lower():
        return {
            "confidence_score": "0.0%",
            "grounding_score": 0.0,
            "semantic_similarity": 0.0,
            "graph_grounding_score": 0.0,
            "hallucination_risk": "HIGH / UNANSWERED",
            "is_grounded": False
        }

    chunks_text = " ".join([c["text"] for c in retrieved_chunks]).lower()
    combined_evidence = f"{chunks_text} {graph_facts.lower() if graph_facts else ''}"
    
    answer_words = re.findall(r'\w+', answer.lower())
    
    # Exclude common stop words
    stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were", "this", "that", "it", "as", "by"}
    content_words = [w for w in answer_words if w not in stop_words and len(w) > 2]
    
    # Layer 1: N-Gram Grounding Overlap
    if not content_words:
        grounding_score = 1.0
    else:
        matched_words = [w for w in content_words if w in combined_evidence]
        grounding_score = len(matched_words) / len(content_words)

    # Layer 2: Semantic Embedding Cosine Similarity
    answer_vector = embedding_model.encode([answer], normalize_embeddings=True, convert_to_numpy=True)
    context_vector = embedding_model.encode([combined_evidence[:2500]], normalize_embeddings=True, convert_to_numpy=True)
    semantic_sim = float(np.dot(answer_vector[0], context_vector[0]))

    # Layer 3: Graph Relational Grounding Check
    graph_grounding_score = 1.0
    if graph_facts and "•" in graph_facts:
        fact_lines = [line.strip() for line in graph_facts.split("\n") if line.strip().startswith("•")]
        if fact_lines:
            matched_facts = 0
            for fact in fact_lines:
                # Extract key entities from path string
                entities = re.findall(r'\((.*?)\)', fact)
                if any(e.lower() in answer.lower() for e in entities if len(e) > 2):
                    matched_facts += 1
            graph_grounding_score = matched_facts / len(fact_lines) if fact_lines else 1.0

    # Layer 4: Composite Confidence Score Calculation
    top_rrf = retrieved_chunks[0].get("rrf_score", 0.01) if retrieved_chunks else 0.0
    rrf_strength = min(top_rrf / 0.033, 1.0)  # Normalize top RRF score against max theoretical ~0.033

    final_confidence = (
        (0.35 * rrf_strength) + 
        (0.25 * grounding_score) + 
        (0.20 * max(semantic_sim, 0.0)) + 
        (0.20 * graph_grounding_score)
    )
    confidence_pct = round(final_confidence * 100, 1)

    risk_level = "LOW" if confidence_pct >= 70.0 else ("MEDIUM" if confidence_pct >= 45.0 else "HIGH")

    return {
        "confidence_score": f"{confidence_pct}%",
        "grounding_score": round(grounding_score, 4),
        "semantic_similarity": round(semantic_sim, 4),
        "graph_grounding_score": round(graph_grounding_score, 4),
        "hallucination_risk": risk_level,
        "is_grounded": confidence_pct >= 60.0
    }
