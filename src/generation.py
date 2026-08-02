"""
generation.py - Answer Generation, Hallucination Verification & Confidence Engine for HyRAG.

This module formats hybrid retrieval contexts, calls free LLM APIs (Gemini/Groq),
runs a 3-layer hallucination audit, and calculates verified confidence scores.
"""

import os
import json
import re
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import numpy as np

# Load environment variables from .env
load_dotenv()


def build_grounded_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Constructs a strictly retrieval-grounded system prompt.

    Args:
        query (str): User question.
        retrieved_chunks (List[Dict[str, Any]]): Top-K hybrid chunks from Phase 7.

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

    prompt = f"""You are an enterprise AI assistant powered by HyRAG.
Answer the user's question STRICTLY using ONLY the provided document context below.

STRICT INSTRUCTIONS:
1. Base your answer ONLY on the facts explicitly mentioned in the CONTEXT.
2. Do NOT use outside knowledge, prior assumptions, or extrapolate beyond the text.
3. If the CONTEXT does not contain sufficient information to answer the question, state:
   "I cannot answer this question based on the provided enterprise documentation."
4. Include inline citations for facts using the format [Source: filename.pdf, Page X].
5. Keep your response concise, factual, and professional.

CONTEXT:
{context_str}

USER QUESTION: {query}

ANSWER WITH CITATIONS:"""
    
    return prompt


def generate_llm_answer(prompt: str) -> str:
    """
    Calls Google Gemini API (or Groq fallback) using free API credentials.

    Args:
        prompt (str): Grounded prompt string.

    Returns:
        str: Generated LLM response text.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    # 1. Try Google Gemini API
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ Gemini API attempt failed ({e}). Trying Groq fallback...")

    # 2. Try Groq API Fallback
    if groq_key and groq_key != "your_groq_api_key_here":
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Groq API attempt failed: {e}")

    # Fallback if no valid API key is present
    return "⚠️ Please set a valid GEMINI_API_KEY or GROQ_API_KEY in your .env file to enable live LLM generation."


def audit_hallucination_and_confidence(
    answer: str, 
    retrieved_chunks: List[Dict[str, Any]], 
    embedding_model: Any
) -> Dict[str, Any]:
    """
    Runs a 3-layer Hallucination Audit & Confidence Calculation on the generated answer.

    Layer 1: Rule-Based N-Gram Grounding Score
    Layer 2: Semantic Embedding Cosine Similarity (Answer vs. Combined Context)
    Layer 3: Confidence Score Calculation

    Returns:
        Dict[str, Any]: Detailed audit report with confidence score percentage.
    """
    if "cannot answer" in answer.lower() or "please set a valid" in answer.lower():
        return {
            "confidence_score": 0.0,
            "grounding_score": 0.0,
            "semantic_similarity": 0.0,
            "hallucination_risk": "HIGH / UNANSWERED",
            "is_grounded": False
        }

    combined_context = " ".join([c["text"] for c in retrieved_chunks]).lower()
    answer_words = re.findall(r'\w+', answer.lower())
    
    # Exclude common stop words
    stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were", "this", "that", "it"}
    content_words = [w for w in answer_words if w not in stop_words and len(w) > 2]
    
    if not content_words:
        grounding_score = 1.0
    else:
        matched_words = [w for w in content_words if w in combined_context]
        grounding_score = len(matched_words) / len(content_words)

    # Layer 2: Semantic Embedding Cosine Similarity
    answer_vector = embedding_model.encode([answer], normalize_embeddings=True, convert_to_numpy=True)
    context_vector = embedding_model.encode([combined_context[:2000]], normalize_embeddings=True, convert_to_numpy=True)
    semantic_sim = float(np.dot(answer_vector[0], context_vector[0]))

    # Layer 3: Fusion Confidence Score Calculation
    top_rrf = retrieved_chunks[0].get("rrf_score", 0.01) if retrieved_chunks else 0.0
    rrf_strength = min(top_rrf / 0.033, 1.0)  # Normalize top RRF score against max theoretical ~0.033

    final_confidence = (0.4 * rrf_strength) + (0.3 * grounding_score) + (0.3 * max(semantic_sim, 0.0))
    confidence_pct = round(final_confidence * 100, 1)

    risk_level = "LOW" if confidence_pct >= 70.0 else ("MEDIUM" if confidence_pct >= 45.0 else "HIGH")

    return {
        "confidence_score": f"{confidence_pct}%",
        "grounding_score": round(grounding_score, 4),
        "semantic_similarity": round(semantic_sim, 4),
        "hallucination_risk": risk_level,
        "is_grounded": confidence_pct >= 60.0
    }


if __name__ == "__main__":
    from src.embeddings import load_embedding_model
    from src.retrieval import build_bm25_index, search_bm25, reciprocal_rank_fusion
    import faiss

    # End-to-End Pipeline Test!
    storage_dir = os.path.join("storage", "faiss_index")
    index_path = os.path.join(storage_dir, "index.faiss")
    metadata_path = os.path.join(storage_dir, "chunks_metadata.json")

    try:
        print("--- PHASE 8 END-TO-END HyRAG PIPELINE TEST ---")
        faiss_index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        model = load_embedding_model()
        bm25, _ = build_bm25_index(chunks)

        # Test Query
        query = "What is Amazon's policy regarding workplace gifts and conflict of interest?"
        print(f"\n❓ USER QUERY: '{query}'")

        # 1. Retrieval (FAISS + BM25 + RRF)
        from src.embeddings import search_faiss_index
        dense_res = search_faiss_index(query, model, faiss_index, chunks, top_k=5)
        sparse_res = search_bm25(query, bm25, chunks, top_k=5)
        hybrid_chunks = reciprocal_rank_fusion(dense_res, sparse_res, k=60, top_k=3)

        # 2. Build Grounded Prompt
        prompt = build_grounded_prompt(query, hybrid_chunks)

        # 3. Generate LLM Answer
        answer = generate_llm_answer(prompt)
        print(f"\n🤖 GENERATED ANSWER:\n{answer}")

        # 4. Run Hallucination & Confidence Audit
        audit = audit_hallucination_and_confidence(answer, hybrid_chunks, model)
        print(f"\n🛡️ HALLUCINATION & CONFIDENCE AUDIT:")
        print(f"   • Confidence Score: {audit['confidence_score']}")
        print(f"   • Grounding Score: {audit['grounding_score']}")
        print(f"   • Semantic Similarity: {audit['semantic_similarity']}")
        print(f"   • Hallucination Risk: {audit['hallucination_risk']}")

    except Exception as e:
        print(f"❌ Error during Phase 8 test: {e}")
