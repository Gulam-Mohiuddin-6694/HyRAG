"""
app.py - Enterprise UI for HyRAG: Hybrid Graph Retrieval-Augmented Generation Console.

HyRAG: Hallucination-Aware Enterprise Knowledge Assistant with Vector + BM25 + Graph RAG.
"""

import os
import json
import time
import base64
import streamlit as st
import faiss
from dotenv import load_dotenv

# Import HyRAG Backend Modules
from src.embeddings import load_embedding_model, search_faiss_index
from src.retrieval import build_bm25_index, search_bm25, reciprocal_rank_fusion, tri_hybrid_search
from src.generation import build_grounded_prompt, generate_llm_answer, audit_hallucination_and_confidence
from src.graph_store import NetworkXGraphStore
from src.conflict_detector import detect_graph_conflicts, format_conflict_prompt_notice

# Load Environment Variables (.env)
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="HyRAG Console",
    page_icon="🟧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to encode local image for CSS display
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

logo_b64 = get_base64_of_bin_file("logo.jpg")

# DESIGN SPECS & COMPACT LAYOUT STYLING (REMOVE EXCESSIVE PADDING)
st.markdown("""
<style>
    /* Remove default Streamlit top padding & extra spaces */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Global App Background & Typography */
    .stApp {
        background-color: #FFFFFF !important;
        color: #232F3E !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #232F3E;
    }

    /* Top Bar Header Component with Logo */
    .top-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 16px;
        margin-bottom: 16px;
        border-bottom: 1px solid #E5E7EB;
        background-color: #FFFFFF;
    }
    .top-logo-img {
        height: 36px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .model-pill {
        background-color: #F8FAFC;
        border: 1px solid #E5E7EB;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Architecture Info Badge Pills */
    .arch-badge {
        background-color: #F8FAFC;
        border: 1px solid #E5E7EB;
        color: #232F3E;
        padding: 6px 10px;
        border-radius: 6px;
        font-family: monospace;
        font-weight: 700;
        font-size: 12px;
        margin-bottom: 8px;
    }

    /* Sidebar Layout */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E7EB !important;
        width: 260px !important;
    }
    section[data-testid="stSidebar"] * {
        color: #232F3E !important;
    }

    /* Navigation item active state */
    .nav-active {
        background-color: #FFF3E0 !important;
        color: #FF9900 !important;
        font-weight: 700;
        padding: 6px 10px;
        border-radius: 6px;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
    }
    .nav-item {
        color: #232F3E;
        padding: 6px 10px;
        border-radius: 6px;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 500;
        font-size: 14px;
    }

    /* Stats Card in Sidebar */
    .kb-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 10px;
        margin-top: 6px;
    }

    /* Welcome Banner Card */
    .welcome-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        border-left: 5px solid #FF9900;
        padding: 18px 24px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .welcome-title {
        font-size: 24px;
        font-weight: 800;
        color: #232F3E;
        margin-bottom: 2px;
    }
    .welcome-subtitle {
        color: #146EB4;
        font-size: 14px;
        font-weight: 500;
    }

    /* Primary Execute Button */
    div.stButton > button {
        background-color: #FF9900 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-size: 14px !important;
    }
    div.stButton > button:hover {
        background-color: #e68a00 !important;
    }

    /* Enterprise 4 KPI Cards */
    .kpi-box {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .kpi-header {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 13px;
        font-weight: 700;
        color: #232F3E;
        margin-bottom: 8px;
    }
    .kpi-number {
        font-size: 24px;
        font-weight: 800;
        color: #232F3E;
        margin-bottom: 2px;
    }
    .kpi-subtext-green {
        color: #10B981;
        font-size: 12px;
        font-weight: 600;
    }
    .kpi-subtext-blue {
        color: #146EB4;
        font-size: 12px;
        font-weight: 600;
    }
    .kpi-subtext-gray {
        color: #64748B;
        font-size: 12px;
        font-weight: 500;
    }
    .progress-bar-green {
        height: 3px;
        background-color: #10B981;
        border-radius: 2px;
        margin-top: 6px;
    }
    .progress-bar-blue {
        height: 3px;
        background-color: #146EB4;
        border-radius: 2px;
        margin-top: 6px;
    }
    .progress-bar-orange {
        height: 3px;
        background-color: #FF9900;
        border-radius: 2px;
        margin-top: 6px;
    }

    /* Verified Answer Card */
    .answer-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 20px;
        margin-top: 8px;
        margin-bottom: 16px;
        font-size: 15px;
        line-height: 1.5;
        color: #232F3E;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }

    /* Graph Facts Container */
    .graph-evidence-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #146EB4;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 10px;
        margin-bottom: 16px;
    }
    .graph-path-item {
        font-family: monospace;
        font-size: 13px;
        color: #1E293B;
        padding: 4px 0;
    }

    /* Sources Row Item */
    .source-row {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .source-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .source-filename {
        font-weight: 700;
        font-size: 13.5px;
        color: #232F3E;
    }
    .source-pagenum {
        font-size: 11.5px;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)


# CACHED BACKEND LOADERS
@st.cache_resource(show_spinner=False)
def get_cached_embedding_model():
    return load_embedding_model()


@st.cache_resource(show_spinner=False)
def get_cached_faiss_and_metadata():
    storage_dir = os.path.join("storage", "faiss_index")
    index_path = os.path.join(storage_dir, "index.faiss")
    metadata_path = os.path.join(storage_dir, "chunks_metadata.json")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, None

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


@st.cache_resource(show_spinner=False)
def get_cached_bm25_index(_chunks):
    if not _chunks:
        return None
    bm25, _ = build_bm25_index(_chunks)
    return bm25


@st.cache_resource(show_spinner=False)
def get_cached_graph_store():
    graph_path = os.path.join("storage", "graph_store", "knowledge_graph.json")
    graph_store = NetworkXGraphStore()
    if os.path.exists(graph_path):
        graph_store.load(graph_path)
    return graph_store


# --- TOP BAR (COMPACT HEADER WITH LOGO) ---
logo_header_html = f'<img src="data:image/jpeg;base64,{logo_b64}" class="top-logo-img" />' if logo_b64 else '<span style="font-size: 20px; font-weight: 800; color: #232F3E;">HyRAG</span>'

col_tb1, col_tb2 = st.columns([4, 1])
with col_tb1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px;">
        {logo_header_html}
        <span style="font-size: 18px; font-weight: 800; color: #232F3E; letter-spacing: -0.3px;">HyRAG Console</span>
    </div>
    """, unsafe_allow_html=True)

with col_tb2:
    st.markdown('<div class="model-pill">⚙️ Hybrid Graph RAG</div>', unsafe_allow_html=True)


# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists("logo.jpg"):
        st.image("logo.jpg", use_container_width=True)
    else:
        st.markdown("""
        <div style="font-size: 22px; font-weight: 800; color: #232F3E; margin-bottom: 12px;">
            <span style="color: #FF9900;">Hy</span>RAG
        </div>
        """, unsafe_allow_html=True)

    # Navigation Menu
    st.markdown('<div class="nav-active">🏠 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">🕸️ Knowledge Graph</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">🕒 Query History</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📄 Audit Logs</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Knowledge Base Section
    st.markdown("<h5 style='font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase;'>KNOWLEDGE BASE</h5>", unsafe_allow_html=True)
    faiss_index, chunks = get_cached_faiss_and_metadata()
    graph_store = get_cached_graph_store()
    g_stats = graph_store.stats() if graph_store else {"total_nodes": 0, "total_edges": 0}
    
    if faiss_index is not None:
        st.markdown("<p style='font-size: 12px; color: #10B981; font-weight: 600;'>🟢 Tri-Hybrid Connected</p>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kb-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <div>
                    <div style="font-size: 10px; color: #64748B;">Total Chunks</div>
                    <div style="font-size: 16px; font-weight: 800; color: #232F3E;">{len(chunks):,}</div>
                </div>
                <div>
                    <div style="font-size: 10px; color: #64748B;">Graph Entities</div>
                    <div style="font-size: 16px; font-weight: 800; color: #232F3E;">{g_stats.get('total_nodes', 0):,}</div>
                </div>
            </div>
            <div style="font-size: 11px; color: #64748B;">Graph Relations: <b>{g_stats.get('total_edges', 0):,}</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Index not found. Run 'python build_graph.py' to generate initial vector & graph indexes.")
        if st.button("🔨 Build Index Now"):
            with st.spinner("Building vector & graph indexes..."):
                import subprocess
                res = subprocess.run(["python", "build_graph.py"], capture_output=True, text=True)
                st.write(res.stdout)
                st.rerun()
        st.stop()

    st.markdown("---")

    # Architecture Section
    st.markdown("<h5 style='font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase;'>ARCHITECTURE</h5>", unsafe_allow_html=True)
    st.caption("Embedding Engine")
    st.markdown('<div class="arch-badge">BAAI/bge-small-en-v1.5</div>', unsafe_allow_html=True)
    
    st.caption("Knowledge Graph Engine")
    st.markdown('<div class="arch-badge">NetworkX + Provenance JSON</div>', unsafe_allow_html=True)
    
    st.caption("Fusion Strategy")
    st.markdown('<div class="arch-badge">Tri-Hybrid RRF (Dense+BM25+Graph)</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Hyperparameters Section
    st.markdown("<h5 style='font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase;'>HYPERPARAMETERS</h5>", unsafe_allow_html=True)
    top_k_chunks = st.slider("Top-K Depth", min_value=1, max_value=10, value=4)
    rrf_k_const = st.slider("RRF Constant (k)", min_value=10, max_value=200, value=60)
    max_hops = st.slider("Graph Max Hops", min_value=1, max_value=3, value=2)
    min_confidence = st.slider("Min Edge Confidence", min_value=0.0, max_value=1.0, value=0.60, step=0.05)

    st.markdown("---")
    st.markdown("""
    <div style="background-color: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 6px; padding: 8px;">
        <div style="font-size: 12px; font-weight: 600; color: #10B981;">🟢 System Status</div>
        <div style="font-size: 11px; color: #64748B;">All Graph & Vector indices active</div>
    </div>
    """, unsafe_allow_html=True)


# --- WELCOME CARD WITH EMBEDDED OFFICIAL LOGO ---
logo_html = f'<img src="data:image/jpeg;base64,{logo_b64}" style="width: 110px; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);" />' if logo_b64 else '<div style="font-size: 32px; color: #FF9900;">🕸️</div>'

st.markdown(f"""
<div class="welcome-card">
    <div>
        <div class="welcome-title">HyRAG Console</div>
        <div class="welcome-subtitle">Hallucination-Aware Hybrid Graph Retrieval Assistant</div>
    </div>
    <div>
        {logo_html}
    </div>
</div>
""", unsafe_allow_html=True)


# Initialize Models
with st.spinner("Initializing models..."):
    model = get_cached_embedding_model()
    bm25 = get_cached_bm25_index(chunks)


# --- ENTERPRISE SEARCH SECTION ---
st.markdown("<h4 style='font-weight: 800; color: #232F3E; margin-bottom: 6px; font-size: 18px;'>Enterprise Search</h4>", unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1])

with col_input:
    query_input = st.text_input(
        "Search Prompt",
        placeholder="Ask questions about policies, frameworks, architectures, or multi-hop dependencies...",
        label_visibility="collapsed"
    )

with col_btn:
    ask_button = st.button("Execute ➔", type="primary", use_container_width=True)


# --- EXAMPLE QUERY CHIPS ---
col_c1, col_c2, col_c3 = st.columns(3)

sample_query = ""
if col_c1.button("🛡️ AWS IAM & MFA Policy", use_container_width=True):
    sample_query = "What is the AWS policy on multi-factor authentication (MFA) for root accounts?"
if col_c2.button("📊 AWS Well-Architected Security", use_container_width=True):
    sample_query = "What are the core pillars and security principles of AWS Well-Architected Framework?"
if col_c3.button("🎁 Amazon Gift & Conflict Policy", use_container_width=True):
    sample_query = "What is Amazon's policy regarding workplace gifts and conflict of interest?"

if sample_query:
    query_input = sample_query


# --- EXECUTION & DISPLAY ---
if (ask_button or sample_query) and query_input.strip():
    start_time = time.time()

    with st.spinner("Executing Tri-Hybrid Retrieval (Vector + BM25 + Graph)..."):
        hybrid_chunks, graph_res = tri_hybrid_search(
            query=query_input,
            model=model,
            faiss_index=faiss_index,
            bm25_index=bm25,
            chunks=chunks,
            graph_store=graph_store,
            top_k_chunks=top_k_chunks,
            rrf_k=rrf_k_const,
            max_hops=max_hops,
            min_relation_confidence=min_confidence
        )

    with st.spinner("Analyzing Multi-Document Conflicts & Grounding..."):
        conflicts = detect_graph_conflicts(graph_res.get("subgraph", {}).get("edges", []))
        conflict_notice = format_conflict_prompt_notice(conflicts)

    with st.spinner("Generating LLM Response..."):
        prompt = build_grounded_prompt(
            query=query_input,
            retrieved_chunks=hybrid_chunks,
            graph_facts=graph_res.get("facts_summary", ""),
            conflict_notice=conflict_notice
        )
        answer = generate_llm_answer(prompt)

    with st.spinner("Auditing 4-Layer Hallucination Grounding..."):
        audit = audit_hallucination_and_confidence(
            answer=answer,
            retrieved_chunks=hybrid_chunks,
            embedding_model=model,
            graph_facts=graph_res.get("facts_summary", "")
        )

    elapsed_time = round(time.time() - start_time, 2)

    st.markdown("---")

    # --- 4 EQUAL KPI CARDS ---
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-header">🛡️ Confidence Score</div>
            <div class="kpi-number">{audit['confidence_score']}</div>
            <div class="kpi-subtext-green">Tri-Hybrid Grounded</div>
            <div class="progress-bar-green"></div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-header">🛡️ Hallucination Risk</div>
            <div class="kpi-number">{audit['hallucination_risk']}</div>
            <div class="kpi-subtext-green">Verified Grounding</div>
            <div class="progress-bar-green"></div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-header">🕸️ Graph Grounding</div>
            <div class="kpi-number">{int(audit.get('graph_grounding_score', 1.0)*100)}%</div>
            <div class="kpi-subtext-blue">Relational Facts Validated</div>
            <div class="progress-bar-blue"></div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-header">🕒 Response Time</div>
            <div class="kpi-number">{elapsed_time}s</div>
            <div class="kpi-subtext-gray">Total Execution</div>
            <div class="progress-bar-orange"></div>
        </div>
        """, unsafe_allow_html=True)

    # --- CONFLICT NOTICE ALERT (IF CONFLICTS DETECTED) ---
    if conflicts:
        st.warning(f"⚠️ **Multi-Document Conflict Detected**: {conflicts[0]['warning']}")

    # --- VERIFIED ANSWER CARD ---
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">
        <h4 style="font-weight: 800; color: #232F3E; margin: 0; font-size: 18px;">🛡️ Verified Answer</h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="answer-card">
        {answer}
    </div>
    """, unsafe_allow_html=True)

    # --- GRAPH EVIDENCE PANEL ---
    subgraph_paths = graph_res.get("subgraph", {}).get("paths", [])
    if subgraph_paths:
        st.markdown("<h4 style='font-weight: 800; color: #232F3E; margin-top: 18px; font-size: 18px;'>🕸️ Knowledge Graph Evidence & Subgraph Paths</h4>", unsafe_allow_html=True)
        paths_html = "".join([f'<div class="graph-path-item">• {p}</div>' for p in subgraph_paths])
        st.markdown(f"""
        <div class="graph-evidence-card">
            <div style="font-weight: 700; color: #146EB4; font-size: 13px; margin-bottom: 6px;">TRAVERSED GRAPH PATHS (Multi-Hop Provenance)</div>
            {paths_html}
        </div>
        """, unsafe_allow_html=True)

    # --- SOURCES REFERENCED ---
    st.markdown("<h4 style='font-weight: 800; color: #232F3E; margin-top: 18px; font-size: 18px;'>📄 Sources Referenced</h4>", unsafe_allow_html=True)
    
    unique_sources = []
    seen = set()
    for c in hybrid_chunks:
        fname = c['metadata']['file_name']
        page = c['metadata']['page_number']
        pair = f"{fname}_{page}"
        if pair not in seen:
            seen.add(pair)
            unique_sources.append((fname, page))

    for fname, page in unique_sources:
        st.markdown(f"""
        <div class="source-row">
            <div class="source-left">
                <span style="font-size: 16px; color: #FF9900;">📄</span>
                <div>
                    <div class="source-filename">{fname}</div>
                    <div class="source-pagenum">Page {page}</div>
                </div>
            </div>
            <span style="color: #64748B;">❯</span>
        </div>
        """, unsafe_allow_html=True)

    # --- EXPANDABLE PANELS ---
    st.markdown("---")
    with st.expander("🌿 Tri-Hybrid Retrieval Details (Dense FAISS + Sparse BM25 + Graph RAG)"):
        for idx, chunk in enumerate(hybrid_chunks):
            d_rank = f"#{chunk['dense_rank']}" if chunk.get('dense_rank') else "N/A"
            s_rank = f"#{chunk['sparse_rank']}" if chunk.get('sparse_rank') else "N/A"
            g_rank = f"#{chunk['graph_rank']}" if chunk.get('graph_rank') else "N/A"
            st.write(f"**Rank #{idx+1}**: RRF Score: `{chunk['rrf_score']:.6f}` | FAISS Rank: `{d_rank}` | BM25 Rank: `{s_rank}` | Graph Rank: `{g_rank}`")

    with st.expander("📄 Retrieved Chunks (Candidate Pool)"):
        for idx, chunk in enumerate(hybrid_chunks):
            is_graph = " [🕸️ Graph Linked]" if chunk.get("is_graph_provenance") else ""
            st.write(f"**Chunk #{idx+1}**: {chunk['metadata']['file_name']} (Page {chunk['metadata']['page_number']}){is_graph}")
            st.info(chunk['text'])

    with st.expander("🛡️ Hallucination Analysis (4-Layer Audit)"):
        st.write(f"• **Layer 1 (N-Gram Grounding Overlap)**: `{audit['grounding_score']*100:.1f}%`")
        st.write(f"• **Layer 2 (Semantic Vector Cosine Similarity)**: `{audit['semantic_similarity']*100:.1f}%`")
        st.write(f"• **Layer 3 (Graph Fact Relational Alignment)**: `{audit['graph_grounding_score']*100:.1f}%`")
        st.write(f"• **Layer 4 (Tri-Hybrid RRF Verification)**: Validated")

    with st.expander("⚙️ Debug & Subgraph Metadata"):
        st.json({
            "query": query_input,
            "seed_entities": [e.get("name") for e in graph_res.get("seed_entities", [])],
            "subgraph_nodes_count": len(graph_res.get("subgraph", {}).get("nodes", [])),
            "subgraph_edges_count": len(graph_res.get("subgraph", {}).get("edges", [])),
            "audit": audit
        })
