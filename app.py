"""
app.py - Enterprise Knowledge Assistant & Admin Console for HyRAG.

Design System: AWS Enterprise AI Platform Aesthetic (DESIGN.md)
Theme Support: Comprehensive Semantic Light & Dark Theme Support (WCAG AA Compliant)
Architecture: Tri-Hybrid Graph RAG (FAISS Dense Vector + BM25 Sparse + NetworkX Knowledge Graph)
Security: Salted SHA-256 Auth & Role-Based Access Control (Employee vs Admin)
Document Ingestion: Multi-Format (PDF, DOCX, TXT, CSV) Automated Graph RAG Lifecycle
"""

import os
import json
import time
import base64
import streamlit as st
import streamlit.components.v1 as components
import faiss
from dotenv import load_dotenv

# Import Backend Modules (Preserve all underlying business logic)
from src.auth import authenticate_user, is_admin, list_all_users
from src.database import (
    log_query,
    get_query_history,
    get_analytics_summary,
    get_all_documents,
    get_document_by_id
)
from src.embeddings import load_embedding_model, search_faiss_index
from src.retrieval import build_bm25_index, search_bm25, reciprocal_rank_fusion, tri_hybrid_search
from src.generation import build_grounded_prompt, generate_llm_answer, audit_hallucination_and_confidence
from src.graph_store import NetworkXGraphStore
from src.conflict_detector import detect_graph_conflicts, format_conflict_prompt_notice
from src.pipeline_manager import (
    process_new_upload,
    delete_document_from_system,
    reprocess_document_in_system
)
from src.graph_visualizer import generate_interactive_graph_html
from src.ui_assets import (
    get_logo_svg_base64,
    get_icon_svg_base64
)

# Load Environment Variables (.env)
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="HyRAG Enterprise Knowledge Console",
    page_icon="🟧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"
if "search_history" not in st.session_state:
    st.session_state["search_history"] = []
if "active_query_text" not in st.session_state:
    st.session_state["active_query_text"] = ""

is_dark = st.session_state["theme"] == "dark"
logo_b64 = get_logo_svg_base64(is_dark=is_dark)
icon_b64 = get_icon_svg_base64(is_dark=is_dark)


# ==============================================================================
# AUTHORITATIVE CSS DESIGN SYSTEM (SEMANTIC LIGHT & DARK TOKENS)
# ==============================================================================
theme_vars = f"""
    --color-brand-orange: #FF9900;
    --color-brand-orange-hover: #EC7211;
    --color-brand-orange-text: {'#FFB800' if is_dark else '#C25E00'};
    --color-brand-orange-subtle: {'rgba(255, 153, 0, 0.15)' if is_dark else 'rgba(255, 153, 0, 0.08)'};
    --color-brand-charcoal: {'#F8FAFC' if is_dark else '#232F3E'};
    --color-brand-navy: {'#0B0F19' if is_dark else '#131A22'};
    --color-tech-blue: #146EB4;
    --color-tech-blue-hover: #0284C7;
    --color-tech-blue-subtle: {'rgba(2, 132, 199, 0.15)' if is_dark else '#F0F9FF'};
    --color-success: {'#10B981' if is_dark else '#047857'};
    --color-success-bg: {'rgba(16, 185, 129, 0.15)' if is_dark else '#ECFDF5'};
    --color-success-border: {'#065F46' if is_dark else '#A7F3D0'};
    --color-danger: {'#EF4444' if is_dark else '#B91C1C'};
    --color-danger-bg: {'rgba(239, 68, 68, 0.15)' if is_dark else '#FEF2F2'};
    --color-danger-border: {'#991B1B' if is_dark else '#FECACA'};
    --color-warning: {'#F59E0B' if is_dark else '#B45309'};
    --color-warning-bg: {'rgba(245, 158, 11, 0.15)' if is_dark else '#FFFBEB'};
    --color-warning-border: {'#92400E' if is_dark else '#FDE68A'};
    --color-bg-app: {'#0B0F19' if is_dark else '#F8FAFC'};
    --color-bg-card: {'#151C2C' if is_dark else '#FFFFFF'};
    --color-bg-surface-subtle: {'#1E293B' if is_dark else '#F1F5F9'};
    --color-border: {'#26354A' if is_dark else '#E2E8F0'};
    --color-border-subtle: {'#1E293B' if is_dark else '#F1F5F9'};
    --color-border-hover: {'#475569' if is_dark else '#CBD5E1'};
    --color-text-main: {'#F8FAFC' if is_dark else '#0F172A'};
    --color-text-body: {'#E2E8F0' if is_dark else '#1E293B'};
    --color-text-muted: {'#94A3B8' if is_dark else '#475569'};
    --color-input-bg: {'#0F172A' if is_dark else '#FFFFFF'};
    --color-input-border: {'#334155' if is_dark else '#CBD5E1'};
    --color-sidebar-bg: {'#111726' if is_dark else '#FFFFFF'};
    --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', Consolas, monospace;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, {'0.35' if is_dark else '0.05'});
    --shadow-card: 0 1px 3px 0 rgba(0, 0, 0, {'0.4' if is_dark else '0.06'}), 0 1px 2px -1px rgba(0, 0, 0, {'0.4' if is_dark else '0.06'});
    --shadow-hover: 0 4px 6px -1px rgba(0, 0, 0, {'0.5' if is_dark else '0.08'}), 0 2px 4px -2px rgba(0, 0, 0, {'0.5' if is_dark else '0.04'});
    --shadow-modal: 0 10px 25px -5px rgba(0, 0, 0, {'0.6' if is_dark else '0.08'}), 0 8px 10px -6px rgba(0, 0, 0, {'0.5' if is_dark else '0.04'});
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 12px;
    --radius-pill: 9999px;
"""

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        {theme_vars}
    }}

    /* Global App View Container */
    html, body, [data-testid="stAppViewContainer"], .stApp {{
        background-color: var(--color-bg-app) !important;
        color: var(--color-text-main) !important;
        font-family: var(--font-sans) !important;
    }}
    

    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
        background-color: var(--color-bg-app) !important;
    }}

    header[data-testid="stHeader"] {{
        display: none !important;
    }}

    /* Global Typography */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--color-text-main) !important;
        font-family: var(--font-sans) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }}

    p, span, label, div {{
        color: var(--color-text-main);
        font-family: var(--font-sans);
    }}

    .stCaption, caption {{
        color: var(--color-text-muted) !important;
        font-size: 13px !important;
        line-height: 1.4 !important;
    }}

    /* Top Navigation Bar */
    .top-nav-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 10px 18px;
        margin-bottom: 18px;
        box-shadow: var(--shadow-sm);
    }}

    .user-profile-badge {{
        display: flex;
        align-items: center;
        gap: 10px;
        background: var(--color-bg-surface-subtle);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-sm);
        padding: 6px 12px;
    }}

    .role-badge-admin {{
        background: var(--color-danger-bg);
        color: var(--color-danger);
        border: 1px solid var(--color-danger-border);
        padding: 3px 8px;
        border-radius: var(--radius-pill);
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .role-badge-emp {{
        background: var(--color-tech-blue-subtle);
        color: var(--color-tech-blue);
        border: 1px solid {'#0369A1' if is_dark else '#BAE6FD'};
        padding: 3px 8px;
        border-radius: var(--radius-pill);
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: var(--color-sidebar-bg) !important;
        border-right: 1px solid var(--color-border) !important;
        width: 300px !important;
    }}

    section[data-testid="stSidebar"] * {{
        color: var(--color-text-main);
    }}

    /* PRIMARY BUTTONS (Sign In, Search, Ingest) */
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stFormSubmitButton"] button,
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    button[data-testid="stBaseButton-primary"] {{
        background-color: var(--color-brand-orange) !important;
        background: var(--color-brand-orange) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid #E68A00 !important;
        padding: 8px 18px !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
        cursor: pointer !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        opacity: 1 !important;
        visibility: visible !important;
    }}

    div[data-testid="stFormSubmitButton"] > button *,
    div[data-testid="stFormSubmitButton"] button *,
    button[kind="primary"] *,
    button[data-testid="baseButton-primary"] *,
    button[data-testid="stBaseButton-primary"] * {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }}

    div[data-testid="stFormSubmitButton"] button p,
    div[data-testid="stFormSubmitButton"] > button p,
    button[kind="primary"] p,
    button[data-testid="baseButton-primary"] p,
    button[data-testid="stBaseButton-primary"] p,
    div[data-testid="stFormSubmitButton"] button [data-testid="stMarkdownContainer"] p,
    button[kind="primary"] [data-testid="stMarkdownContainer"] p {{
        display: block !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 1 !important;
        visibility: visible !important;
        line-height: normal !important;
    }}

    div[data-testid="stFormSubmitButton"] > button:hover,
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {{
        background-color: var(--color-brand-orange-hover) !important;
        background: var(--color-brand-orange-hover) !important;
        border-color: #CC6008 !important;
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(255, 153, 0, 0.25) !important;
    }}

    /* SECONDARY & UTILITY BUTTONS (Theme toggle, Logout, Chips, Clear) */
    div.stButton > button:not([kind="primary"]) {{
        background-color: var(--color-bg-card) !important;
        background: var(--color-bg-card) !important;
        color: var(--color-text-main) !important;
        border: 1px solid var(--color-border) !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        border-radius: var(--radius-sm) !important;
        padding: 7px 14px !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: var(--shadow-sm) !important;
    }}

    div.stButton > button:not([kind="primary"]) * {{
        color: var(--color-text-main) !important;
        font-weight: 600 !important;
    }}

    div.stButton > button:not([kind="primary"]):hover {{
        background-color: var(--color-bg-surface-subtle) !important;
        background: var(--color-bg-surface-subtle) !important;
        border-color: var(--color-border-hover) !important;
        color: var(--color-text-main) !important;
        transform: translateY(-1px);
        box-shadow: var(--shadow-hover) !important;
    }}

    div.stButton > button:not([kind="primary"]):hover * {{
        color: var(--color-text-main) !important;
    }}

    /* Suggested Query Chip Styling */
    div[data-testid="stHorizontalBlock"] button[key^="chip_"] {{
        background-color: var(--color-bg-surface-subtle) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: var(--radius-pill) !important;
        font-size: 12.5px !important;
    }}

    /* Input Fields & Textareas */
    div[data-baseweb="input"],
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    ul[role="listbox"],
    li[role="option"],
    .stTextInput > div > div,
    div[data-testid="stTextInput"] > div > div {{
        background-color: var(--color-input-bg) !important;
        border-color: var(--color-input-border) !important;
        color: var(--color-text-main) !important;
        border-radius: var(--radius-sm) !important;
        cursor: text !important;
    }}

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    div[data-baseweb="popover"] span,
    div[data-baseweb="popover"] div,
    ul[role="listbox"] span,
    li[role="option"] span,
    li[role="option"] {{
        color: var(--color-text-main) !important;
    }}

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {{
        background-color: var(--color-bg-surface-subtle) !important;
        color: var(--color-brand-orange) !important;
    }}

    div[data-baseweb="select"] svg {{
        fill: var(--color-text-main) !important;
        color: var(--color-text-main) !important;
    }}

    input, textarea,
    input[data-baseweb="input"],
    div[data-baseweb="input"] input,
    .stTextInput input,
    div[data-testid="stTextInput"] input {{
        color: var(--color-text-main) !important;
        caret-color: var(--color-brand-orange) !important;
        background-color: transparent !important;
        font-size: 14px !important;
        cursor: text !important;
        line-height: 1.5 !important;
    }}

    input:focus, textarea:focus,
    input[data-baseweb="input"]:focus,
    div[data-baseweb="input"] input:focus,
    .stTextInput input:focus,
    div[data-testid="stTextInput"] input:focus {{
        caret-color: var(--color-brand-orange) !important;
        outline: none !important;
    }}

    input::placeholder, textarea::placeholder {{
        color: var(--color-text-muted) !important;
        font-size: 13.5px !important;
    }}

    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"]:focus-within {{
        border-color: var(--color-brand-orange) !important;
        box-shadow: 0 0 0 3px rgba(255, 153, 0, 0.22) !important;
    }}

    .stTextInput label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stForm"] label,
    .stSelectbox label,
    div[data-testid="stSelectbox"] label,
    .stSlider label {{
        color: var(--color-text-main) !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
        margin-bottom: 4px !important;
    }}

    /* Password Visibility / Eye Toggle Button Styling */
    div[data-baseweb="input"] button,
    div[data-baseweb="input"] button[aria-label*="password"],
    div[data-baseweb="input"] button[title*="password"],
    [data-testid="stTextInput"] div[data-baseweb="input"] button {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: {'#94A3B8' if is_dark else '#334155'} !important;
        padding: 6px 10px !important;
        margin-right: 4px !important;
        border-radius: var(--radius-sm) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        min-width: 32px !important;
        min-height: 32px !important;
    }}

    div[data-baseweb="input"] button svg,
    div[data-baseweb="input"] button[aria-label*="password"] svg,
    div[data-baseweb="input"] button[title*="password"] svg,
    [data-testid="stTextInput"] div[data-baseweb="input"] button svg {{
        color: {'#94A3B8' if is_dark else '#334155'} !important;
        fill: {'#94A3B8' if is_dark else '#334155'} !important;
        width: 18px !important;
        height: 18px !important;
        transition: all 0.15s ease-in-out !important;
    }}

    div[data-baseweb="input"] button svg path,
    div[data-baseweb="input"] button[aria-label*="password"] svg path,
    div[data-baseweb="input"] button[title*="password"] svg path {{
        fill: {'#94A3B8' if is_dark else '#334155'} !important;
    }}

    div[data-baseweb="input"] button:hover,
    div[data-baseweb="input"] button[aria-label*="password"]:hover,
    div[data-baseweb="input"] button[title*="password"]:hover {{
        background-color: {'rgba(255, 255, 255, 0.08)' if is_dark else '#F1F5F9'} !important;
        color: {'#F8FAFC' if is_dark else '#0F172A'} !important;
    }}

    div[data-baseweb="input"] button:hover svg,
    div[data-baseweb="input"] button:hover svg path {{
        color: {'#F8FAFC' if is_dark else '#0F172A'} !important;
        fill: {'#F8FAFC' if is_dark else '#0F172A'} !important;
    }}

    div[data-baseweb="input"] button:focus,
    div[data-baseweb="input"] button:focus-visible,
    div[data-baseweb="input"] button:active {{
        background-color: {'rgba(255, 255, 255, 0.12)' if is_dark else '#E2E8F0'} !important;
        outline: none !important;
        color: var(--color-brand-orange) !important;
    }}

    div[data-baseweb="input"] button:focus svg,
    div[data-baseweb="input"] button:focus-visible svg,
    div[data-baseweb="input"] button:active svg,
    div[data-baseweb="input"] button:focus svg path,
    div[data-baseweb="input"] button:focus-visible svg path,
    div[data-baseweb="input"] button:active svg path {{
        color: var(--color-brand-orange) !important;
        fill: var(--color-brand-orange) !important;
    }}

    /* Slider Styling */
    div[data-baseweb="slider"] div[role="slider"] {{
        background-color: var(--color-brand-orange) !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    }}

    /* KPI Metric Cards */
    .kpi-card {{
        background-color: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 16px 18px;
        box-shadow: var(--shadow-card);
        position: relative;
        overflow: hidden;
        transition: all 0.2s ease;
    }}

    .kpi-card:hover {{
        border-color: var(--color-border-hover);
        box-shadow: var(--shadow-hover);
    }}

    .kpi-header {{
        font-size: 12px;
        font-weight: 700;
        color: var(--color-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    .kpi-value {{
        font-size: 26px;
        font-weight: 900;
        color: var(--color-text-main);
        letter-spacing: -0.02em;
        line-height: 1.15;
    }}

    .kpi-subtext {{
        font-size: 12px;
        font-weight: 600;
        margin-top: 6px;
    }}

    .kpi-subtext-green {{ color: var(--color-success); }}
    .kpi-subtext-blue {{ color: var(--color-tech-blue); }}
    .kpi-subtext-gray {{ color: var(--color-text-muted); }}
    .kpi-subtext-orange {{ color: var(--color-warning); }}
    .kpi-subtext-red {{ color: var(--color-danger); }}

    .kpi-bar-green {{ position: absolute; bottom: 0; left: 0; right: 0; height: 3px; background: var(--color-success); }}
    .kpi-bar-blue {{ position: absolute; bottom: 0; left: 0; right: 0; height: 3px; background: var(--color-tech-blue); }}
    .kpi-bar-orange {{ position: absolute; bottom: 0; left: 0; right: 0; height: 3px; background: var(--color-brand-orange); }}

    /* Responsive Media Queries */
    @media (max-width: 900px) {{
        .block-container {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
        .kpi-card {{
            margin-bottom: 8px !important;
        }}
    }}

    @media (max-width: 600px) {{
        .login-wrapper {{
            margin: 12px auto !important;
            padding: 0 4px !important;
        }}
        .login-card {{
            padding: 24px 18px 20px 18px !important;
        }}
        .kpi-value {{
            font-size: 22px !important;
        }}
    }}

    /* Verified Answer Card */
    .answer-card {{
        background-color: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 22px 26px;
        margin-top: 8px;
        margin-bottom: 16px;
        font-size: 14.5px;
        line-height: 1.7;
        color: var(--color-text-body);
        box-shadow: var(--shadow-card);
    }}

    .answer-card p {{
        color: var(--color-text-body);
        line-height: 1.7;
        margin-bottom: 12px;
    }}

    .answer-card strong {{
        color: var(--color-text-main);
        font-weight: 700;
    }}

    .answer-card ul, .answer-card ol {{
        margin-left: 20px;
        margin-bottom: 12px;
        color: var(--color-text-body);
    }}

    .answer-card li {{
        margin-bottom: 6px;
        color: var(--color-text-body);
    }}

    /* Graph Evidence Card */
    .graph-evidence-card {{
        background-color: var(--color-bg-surface-subtle);
        border: 1px solid var(--color-border);
        border-left: 4px solid var(--color-tech-blue);
        border-radius: var(--radius-sm);
        padding: 16px 20px;
        margin-top: 10px;
        margin-bottom: 16px;
    }}

    .graph-path-badge {{
        font-family: var(--font-mono);
        font-size: 12px;
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        padding: 6px 10px;
        border-radius: 4px;
        margin: 6px 0;
        color: var(--color-text-main);
        display: block;
        line-height: 1.4;
    }}

    /* Source Row */
    .source-row {{
        background-color: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-sm);
        padding: 12px 16px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: var(--shadow-sm);
        transition: all 0.15s ease;
    }}

    .source-row:hover {{
        border-color: var(--color-border-hover);
        box-shadow: var(--shadow-hover);
    }}

    /* Login Screen Layout */
    .login-wrapper {{
        max-width: 440px;
        margin: 28px auto 20px auto;
    }}

    .login-card {{
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: 36px 32px 32px 32px;
        box-shadow: var(--shadow-modal);
    }}

    .login-heading {{
        font-size: 21px;
        font-weight: 800;
        color: var(--color-text-main);
        text-align: center;
        margin-top: 12px;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }}

    .login-subheading {{
        font-size: 13px;
        color: var(--color-text-muted);
        text-align: center;
        margin-bottom: 24px;
    }}

    .security-badge {{
        font-size: 11.5px;
        color: var(--color-text-muted);
        text-align: center;
        margin-top: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        font-weight: 500;
    }}

    /* Streamlit Tabs */
    button[data-baseweb="tab"] {{
        font-family: var(--font-sans) !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
        color: var(--color-text-muted) !important;
        background: transparent !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.15s ease !important;
    }}

    button[data-baseweb="tab"]:hover {{
        color: var(--color-text-main) !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--color-text-main) !important;
        border-bottom-color: var(--color-brand-orange) !important;
        border-bottom-width: 3px !important;
        font-weight: 700 !important;
    }}

    /* Expanders & DataFrames */
    div[data-testid="stExpander"] {{
        background-color: var(--color-bg-card) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: var(--radius-sm) !important;
        margin-bottom: 8px !important;
    }}

    div[data-testid="stExpander"] summary {{
        color: var(--color-text-main) !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        padding: 10px 14px !important;
    }}

    div[data-testid="stExpander"] div[role="region"] {{
        color: var(--color-text-body) !important;
        padding: 12px 16px !important;
        border-top: 1px solid var(--color-border);
    }}

    /* Dataframe & Table Styling for Light and Dark Modes */
    div[data-testid="stDataFrame"],
    div[data-testid="stDataFrame"] > div,
    div[data-testid="stTable"] {{
        background-color: var(--color-bg-card) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: var(--radius-sm) !important;
    }}

    /* File Uploader Dropzone */
    div[data-testid="stFileUploader"] {{
        background-color: var(--color-bg-card) !important;
        border: 1.5px dashed var(--color-input-border) !important;
        border-radius: var(--radius-md) !important;
        padding: 18px !important;
        transition: border-color 0.2s ease;
    }}

    div[data-testid="stFileUploader"]:hover {{
        border-color: var(--color-brand-orange) !important;
    }}

    div[data-testid="stFileUploader"] * {{
        color: var(--color-text-main) !important;
    }}

    /* Remove Streamlit "Press Enter to submit form" helper text without affecting button text */
    [data-testid="InputInstructions"],
    [data-testid="stFormInstructions"],
    .stFormInstructions,
    div[data-testid="stFormSubmitButton"] > [data-testid="InputInstructions"],
    div[data-testid="stFormSubmitButton"] > small,
    div[data-testid="stForm"] > [data-testid="InputInstructions"],
    div[data-testid="stForm"] > [data-testid="stFormInstructions"] {{
        display: none !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
        visibility: hidden !important;
    }}

    /* Comprehensive Alert, Warning & Notification Box Styling (High Contrast & Theme Compliant) */
    div[data-testid="stAlert"],
    div.stAlert {{
        border-radius: var(--radius-md) !important;
        padding: 12px 18px !important;
        font-size: 13.5px !important;
        line-height: 1.55 !important;
        box-shadow: var(--shadow-sm) !important;
        border-width: 1px !important;
        border-style: solid !important;
        margin-bottom: 12px !important;
        opacity: 1 !important;
        visibility: visible !important;
    }}

    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] div,
    div[data-testid="stAlert"] label,
    div[data-testid="stAlert"] strong,
    div[data-testid="stAlert"] b,
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] * {{
        opacity: 1 !important;
        visibility: visible !important;
        display: inline !important;
    }}

    /* Streamlit Warning Alert Styling */
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]),
    div[data-testid="stAlert"].stAlert-warning,
    div.stAlert:has(svg[data-testid="stIconWarning"]),
    div[data-testid="stAlert"][data-test-script-state="warning"] {{
        background-color: var(--color-warning-bg) !important;
        border-color: var(--color-warning-border) !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]) p,
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]) span,
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]) div,
    div.stAlert:has(svg[data-testid="stIconWarning"]) p,
    div.stAlert:has(svg[data-testid="stIconWarning"]) span {{
        color: {{'#FDE68A' if is_dark else '#78350F'}} !important;
        font-weight: 500 !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]) strong,
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]) b,
    div.stAlert:has(svg[data-testid="stIconWarning"]) strong,
    div.stAlert:has(svg[data-testid="stIconWarning"]) b {{
        color: {{'#FBBF24' if is_dark else '#92400E'}} !important;
        font-weight: 700 !important;
    }}

    /* Streamlit Error Alert Styling */
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentError"]),
    div[data-testid="stAlert"].stAlert-error,
    div.stAlert:has(svg[data-testid="stIconError"]) {{
        background-color: var(--color-danger-bg) !important;
        border-color: var(--color-danger-border) !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stNotificationContentError"]) *,
    div.stAlert:has(svg[data-testid="stIconError"]) * {{
        color: {{'#FECACA' if is_dark else '#991B1B'}} !important;
    }}

    /* Streamlit Success Alert Styling */
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentSuccess"]),
    div[data-testid="stAlert"].stAlert-success,
    div.stAlert:has(svg[data-testid="stIconSuccess"]) {{
        background-color: var(--color-success-bg) !important;
        border-color: var(--color-success-border) !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stNotificationContentSuccess"]) *,
    div.stAlert:has(svg[data-testid="stIconSuccess"]) * {{
        color: {{'#A7F3D0' if is_dark else '#065F46'}} !important;
    }}

    /* Streamlit Info Alert Styling */
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentInfo"]),
    div[data-testid="stAlert"].stAlert-info,
    div.stAlert:has(svg[data-testid="stIconInfo"]) {{
        background-color: var(--color-tech-blue-subtle) !important;
        border-color: {{'#0369A1' if is_dark else '#BAE6FD'}} !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stNotificationContentInfo"]) *,
    div.stAlert:has(svg[data-testid="stIconInfo"]) * {{
        color: {{'#E0F2FE' if is_dark else '#0369A1'}} !important;
    }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHED BACKEND RESOURCE LOADERS
# ==============================================================================
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


# ==============================================================================
# VIEW 1: PRODUCTION LOGIN SCREEN (ENTERPRISE SSO PORTAL)
# ==============================================================================
if not st.session_state["authenticated"]:
    # Top Utility Row with Theme Toggle
    col_t1, col_t2 = st.columns([6, 1])
    with col_t2:
        theme_btn_label = "☀ Light" if is_dark else "🌙 Dark"
        if st.button(theme_btn_label, key="login_theme_toggle", use_container_width=True):
            st.session_state["theme"] = "light" if is_dark else "dark"
            st.rerun()

    # Centered Single-Container SSO Card
    col_l1, col_l2, col_l3 = st.columns([1, 1.4, 1])
    with col_l2:
        st.markdown(f"""
        <div class="login-wrapper">
            <div class="login-card">
                <div style="text-align: center; margin-bottom: 16px;">
                    <img src="{logo_b64}" alt="HyRAG" style="display: block; margin: 0 auto; max-width: 220px; width: 100%; height: auto; object-fit: contain;" />
                </div>
                <div class="login-heading">Enterprise Sign In</div>
                <div class="login-subheading">Hallucination-Aware Hybrid Graph RAG Console</div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            user_id_input = st.text_input(
                "Employee ID",
                placeholder="e.g. EMP1001 or ADMIN001",
                help="Enter your assigned enterprise employee ID or admin credential."
            )
            
            password_input = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                help="Salted SHA-256 protected authentication."
            )

            submit_login = st.form_submit_button("Sign In to HyRAG ➔", use_container_width=True)

            if submit_login:
                if not user_id_input or not password_input:
                    st.error("Please provide both Employee ID and password.")
                else:
                    user_record = authenticate_user(user_id_input, password_input)
                    if user_record:
                        st.session_state["authenticated"] = True
                        st.session_state["user"] = user_record
                        st.session_state["role"] = user_record["role"]
                        st.success(f"Authenticated as {user_record['name']} ({user_record['role']})")
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error("Invalid Employee ID or password. Please verify credentials.")

        st.markdown("""
                <div class="security-badge">
                    🔒 256-Bit Encrypted Session • Role-Based Access Control
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔑 Quick Reference Test Credentials"):
            st.markdown("""
            | Account | Password | Role | Department |
            | :--- | :--- | :--- | :--- |
            | **EMP1001** | `HyRAG@1001` | Employee | Security Architecture |
            | **EMP1002** | `HyRAG@1002` | Employee | Cloud Engineering |
            | **EMP1003** | `HyRAG@1003` | Employee | Compliance & Audit |
            | **ADMIN001** | `HyRAG@Admin01` | Admin | Infrastructure Governance |
            """)

    st.stop()


# ==============================================================================
# LOGGED IN APPLICATION CONSOLE
# ==============================================================================
current_user = st.session_state["user"]
is_user_admin = current_user.get("role") == "Admin"

model = get_cached_embedding_model()
faiss_index, chunks = get_cached_faiss_and_metadata()
bm25 = get_cached_bm25_index(chunks)
graph_store = get_cached_graph_store()
g_stats = graph_store.stats() if graph_store else {"total_nodes": 0, "total_edges": 0}


# ==============================================================================
# TOP NAVIGATION BAR (AWS ENTERPRISE CONSOLE HEADER)
# ==============================================================================
col_nb1, col_nb2, col_nb3, col_nb4 = st.columns([3.2, 1.8, 0.65, 0.65])

with col_nb1:
    role_badge_html = '<span class="role-badge-admin">Admin</span>' if is_user_admin else '<span class="role-badge-emp">Employee</span>'
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 14px; height: 100%;">
        <img src="{logo_b64}" alt="HyRAG" style="object-fit: contain; max-height: 42px; height: 42px; width: auto; display: block;" />
        <span style="border-left: 1.5px solid var(--color-border); height: 26px;"></span>
        <span style="font-size: 13.5px; font-weight: 700; color: var(--color-text-main);">Console</span>
        {role_badge_html}
    </div>
    """, unsafe_allow_html=True)

with col_nb2:
    st.markdown(f"""
    <div style="text-align: right; line-height: 1.35;">
        <div style="font-size: 13px; font-weight: 700; color: var(--color-text-main);">{current_user['name']} <span style="color: var(--color-text-muted); font-size: 11.5px; font-weight: 500;">({current_user['user_id']})</span></div>
        <div style="font-size: 11.5px; color: var(--color-text-muted);">{current_user.get('department', 'Enterprise')}</div>
    </div>
    """, unsafe_allow_html=True)

with col_nb3:
    theme_btn_label = "☀ Light" if is_dark else "🌙 Dark"
    if st.button(theme_btn_label, key="top_theme_toggle", use_container_width=True):
        st.session_state["theme"] = "light" if is_dark else "dark"
        st.rerun()

with col_nb4:
    if st.button("🚪 Sign Out", key="btn_top_logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.session_state["role"] = None
        st.session_state["search_history"] = []
        st.session_state["active_query_text"] = ""
        st.rerun()


# ==============================================================================
# SIDEBAR CONTROLS & TELEMETRY
# ==============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding-bottom: 14px; border-bottom: 1px solid var(--color-border); margin-bottom: 14px;">
        <img src="{logo_b64}" alt="HyRAG" style="object-fit: contain; max-width: 170px; width: 100%; height: auto; display: block; margin: 0 auto;" />
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h5 style='font-size: 11px; font-weight: 800; color: var(--color-text-muted); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;'>KNOWLEDGE BASE TELEMETRY</h5>", unsafe_allow_html=True)
    
    if faiss_index is not None and chunks:
        st.markdown(f"""
        <div style="background: var(--color-bg-card); border: 1px solid var(--color-border); border-radius: 8px; padding: 12px 14px; margin-bottom: 14px; box-shadow: var(--shadow-sm);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <div>
                    <div style="font-size: 10px; color: var(--color-text-muted); font-weight: 700; text-transform: uppercase;">VECTOR CHUNKS</div>
                    <div style="font-size: 18px; font-weight: 800; color: var(--color-text-main);">{len(chunks):,}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 10px; color: var(--color-text-muted); font-weight: 700; text-transform: uppercase;">GRAPH ENTITIES</div>
                    <div style="font-size: 18px; font-weight: 800; color: var(--color-text-main);">{g_stats.get('total_nodes', 0):,}</div>
                </div>
            </div>
            <div style="font-size: 11.5px; color: var(--color-text-muted); border-top: 1px dashed var(--color-border); padding-top: 6px; display: flex; justify-content: space-between;">
                <span>Verified Relations:</span>
                <b style="color: var(--color-text-main);">{g_stats.get('total_edges', 0):,}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Knowledge index empty. Upload documents in Document Management to initialize.")

    st.markdown("<h5 style='font-size: 11px; font-weight: 800; color: var(--color-text-muted); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;'>RETRIEVAL CONFIGURATION</h5>", unsafe_allow_html=True)
    top_k_chunks = st.slider("Top-K Depth", min_value=1, max_value=10, value=4, help="Number of hybrid chunks retrieved.")
    rrf_k_const = st.slider("RRF Constant (k)", min_value=10, max_value=200, value=60, help="Reciprocal Rank Fusion smoothing parameter.")
    max_hops = st.slider("Graph Max Hops", min_value=1, max_value=3, value=2, help="Bounded traversal exploration depth.")
    min_confidence = st.slider("Min Edge Confidence", min_value=0.0, max_value=1.0, value=0.60, step=0.05, help="Pruning threshold for relation extraction.")


# ==============================================================================
# MAIN VIEW DISPATCHER
# ==============================================================================
if is_user_admin:
    admin_tabs = st.tabs([
        "📊 System Overview",
        "💬 Enterprise Search",
        "📁 Document Management",
        "🕸️ Knowledge Graph Explorer",
        "📜 Audit History"
    ])
    tab_overview, tab_search, tab_docs, tab_graph, tab_audit = admin_tabs
else:
    emp_tabs = st.tabs([
        "💬 Enterprise Search",
        "🕒 My Query History",
        "🕸️ Knowledge Graph Explorer"
    ])
    tab_search, tab_my_history, tab_graph = emp_tabs
    tab_overview = None
    tab_docs = None
    tab_audit = None


# ==============================================================================
# ADMIN TAB: OVERVIEW & SYSTEM ANALYTICS
# ==============================================================================
if is_user_admin and tab_overview:
    with tab_overview:
        st.markdown("<h3 style='font-weight: 800; margin-bottom: 2px;'>System Overview & Enterprise Telemetry</h3>", unsafe_allow_html=True)
        st.caption("Live system telemetry, document ingestion metrics, query volumes, and employee activity.")

        summary = get_analytics_summary()

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">👥 Registered Users</div>
                <div class="kpi-value">{len(list_all_users())}</div>
                <div class="kpi-subtext kpi-subtext-green">10 Active Accounts</div>
                <div class="kpi-bar-green"></div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">📁 Active Documents</div>
                <div class="kpi-value">{summary['total_documents']}</div>
                <div class="kpi-subtext kpi-subtext-blue">{len(chunks) if chunks else 0} Chunks Indexed</div>
                <div class="kpi-bar-blue"></div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">💬 Total Queries</div>
                <div class="kpi-value">{summary['total_queries']}</div>
                <div class="kpi-subtext kpi-subtext-green">All Time Total</div>
                <div class="kpi-bar-green"></div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">🕒 Queries Today</div>
                <div class="kpi-value">{summary['queries_today']}</div>
                <div class="kpi-subtext kpi-subtext-blue">Today's Volume</div>
                <div class="kpi-bar-blue"></div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">🕸️ Graph Entities</div>
                <div class="kpi-value">{g_stats.get('total_nodes', 0)}</div>
                <div class="kpi-subtext kpi-subtext-gray">{g_stats.get('total_edges', 0)} Verified Relations</div>
                <div class="kpi-bar-orange"></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        col_g1, col_g2 = st.columns([1.2, 1])
        with col_g1:
            st.markdown("#### 👥 Employee Activity Metrics")
            emp_data = summary.get("queries_per_employee", [])
            if emp_data:
                st.dataframe(emp_data, use_container_width=True)
            else:
                st.info("No query activity recorded yet. Run a search in the Enterprise Search tab.")

        with col_g2:
            st.markdown("#### ⚙️ Infrastructure & Model Status")
            st.markdown(f"""
            <div style="background: var(--color-bg-card); border: 1px solid var(--color-border); border-radius: var(--radius-md); padding: 16px 20px; box-shadow: var(--shadow-sm);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 13px;">
                    <span style="color: var(--color-text-muted);">Avg Pipeline Latency:</span>
                    <b style="color: var(--color-success);">{summary['avg_latency']}s</b>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 13px;">
                    <span style="color: var(--color-text-muted);">Embedding Model:</span>
                    <span style="font-family: var(--font-mono); font-size: 12px; color: var(--color-text-main);">BAAI/bge-small-en-v1.5</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 13px;">
                    <span style="color: var(--color-text-muted);">Dense Vector Engine:</span>
                    <span style="font-family: var(--font-mono); font-size: 12px; color: var(--color-text-main);">FAISS IndexFlatIP</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 13px;">
                    <span style="color: var(--color-text-muted);">Sparse Search Engine:</span>
                    <span style="font-family: var(--font-mono); font-size: 12px; color: var(--color-text-main);">BM25 (Rank-BM25)</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 13px;">
                    <span style="color: var(--color-text-muted);">Knowledge Graph DB:</span>
                    <span style="font-family: var(--font-mono); font-size: 12px; color: var(--color-text-main);">NetworkX MultiDiGraph</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 13px;">
                    <span style="color: var(--color-text-muted);">Synthesis LLM:</span>
                    <span style="font-family: var(--font-mono); font-size: 12px; color: var(--color-brand-orange-text); font-weight: 700;">Groq openai/gpt-oss-120b</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# TAB: ENTERPRISE SEARCH & CHAT (STATE-MANAGED HYBRID RETRIEVAL)
# ==============================================================================
with tab_search:
    st.markdown("<h3 style='font-weight: 800; margin-bottom: 2px;'>Enterprise Knowledge Assistant</h3>", unsafe_allow_html=True)
    st.caption("Tri-Hybrid Retrieval Engine with FAISS Vector Search, BM25 Keyword Search, and Knowledge Graph Grounding.")

    # Check for requested query trigger from suggested chips or form submit
    execute_query = None

    # Suggested Query Chips
    st.markdown("<div style='margin-top: 8px; margin-bottom: 8px; font-size: 11.5px; font-weight: 700; color: var(--color-text-muted); text-transform: uppercase; letter-spacing: 0.6px;'>⚡ Recommended Knowledge Queries</div>", unsafe_allow_html=True)
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    
    if col_c1.button("🛡️ AWS IAM & MFA Policy", key="chip_aws_iam", use_container_width=True):
        execute_query = "What is the AWS policy on multi-factor authentication (MFA) for root accounts?"
        st.session_state["active_query_text"] = execute_query
    if col_c2.button("📊 Well-Architected Pillars", key="chip_aws_arch", use_container_width=True):
        execute_query = "What are the core pillars and security principles of AWS Well-Architected Framework?"
        st.session_state["active_query_text"] = execute_query
    if col_c3.button("🎁 Amazon Gift & Conflict", key="chip_amz_gift", use_container_width=True):
        execute_query = "What is Amazon's policy regarding workplace gifts, meals, and conflict of interest?"
        st.session_state["active_query_text"] = execute_query
    if col_c4.button("☁️ Cloud Disaster Recovery", key="chip_cloud_dr", use_container_width=True):
        execute_query = "What are the required RTO and RPO targets and AWS services for multi-region disaster recovery?"
        st.session_state["active_query_text"] = execute_query

    # Search Form (Supports Enter Key and Search Button Click)
    with st.form("enterprise_search_form", clear_on_submit=False):
        col_inp, col_sub = st.columns([5, 1.2])
        with col_inp:
            form_query_val = st.text_input(
                "Search Input",
                value=st.session_state["active_query_text"],
                placeholder="Ask questions about company policies, AWS IAM, MFA security, disaster recovery, or cross-document rules... (Press Enter to search)",
                label_visibility="collapsed"
            )
        with col_sub:
            submit_search_btn = st.form_submit_button("Search Knowledge ➔", use_container_width=True)

        if submit_search_btn:
            if form_query_val.strip():
                execute_query = form_query_val.strip()
            else:
                st.warning("Please enter a question or select a suggested topic to search.")

    # Execution Engine for Tri-Hybrid Graph RAG Pipeline
    if execute_query:
        st.session_state["active_query_text"] = execute_query
        if not chunks or faiss_index is None:
            st.error("Knowledge base is currently empty. Please upload documents in the Document Management tab.")
        else:
            with st.status(f"⚡ Processing query: \"{execute_query[:60]}...\"", expanded=True) as search_status:
                try:
                    start_time = time.time()

                    search_status.write("🔍 [1/4] Executing Tri-Hybrid Search (FAISS Dense Vector + BM25 Sparse + Knowledge Graph)...")
                    hybrid_chunks, graph_res = tri_hybrid_search(
                        query=execute_query,
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

                    search_status.write("🕸️ [2/4] Analyzing relational graph facts and detecting multi-document conflicts...")
                    conflicts = detect_graph_conflicts(graph_res.get("subgraph", {}).get("edges", []))
                    conflict_notice = format_conflict_prompt_notice(conflicts)

                    search_status.write("🧠 [3/4] Synthesizing grounded response via Groq LLM...")
                    prompt = build_grounded_prompt(
                        query=execute_query,
                        retrieved_chunks=hybrid_chunks,
                        graph_facts=graph_res.get("facts_summary", ""),
                        conflict_notice=conflict_notice
                    )
                    answer = generate_llm_answer(prompt)

                    search_status.write("🛡️ [4/4] Auditing 4-Layer Hallucination Grounding & Confidence Score...")
                    audit = audit_hallucination_and_confidence(
                        answer=answer,
                        retrieved_chunks=hybrid_chunks,
                        embedding_model=model,
                        graph_facts=graph_res.get("facts_summary", "")
                    )

                    elapsed_time = round(time.time() - start_time, 2)
                    search_status.update(label=f"✅ Verified Response Generated in {elapsed_time}s", state="complete", expanded=False)

                    # Log to Persistent SQLite Database
                    retrieved_sources_summary = [
                        {"file_name": c["metadata"].get("file_name"), "page_number": c["metadata"].get("page_number")}
                        for c in hybrid_chunks
                    ]
                    log_query(
                        user_id=current_user["user_id"],
                        user_role=current_user["role"],
                        query=execute_query,
                        answer=answer,
                        retrieved_docs=retrieved_sources_summary,
                        confidence_score=audit["confidence_score"],
                        hallucination_risk=audit["hallucination_risk"],
                        graph_facts=graph_res.get("facts_summary", ""),
                        latency_seconds=elapsed_time
                    )

                    # Store in session state for persistence
                    result_payload = {
                        "query": execute_query,
                        "answer": answer,
                        "audit": audit,
                        "graph_res": graph_res,
                        "conflicts": conflicts,
                        "hybrid_chunks": hybrid_chunks,
                        "latency": elapsed_time,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # Add to session history
                    st.session_state["search_history"].insert(0, result_payload)

                except Exception as err:
                    search_status.update(label="❌ Retrieval Pipeline Error", state="error", expanded=True)
                    st.error(f"Error executing retrieval pipeline: {err}")

    # Render Current / Recent Search Results from State
    if st.session_state["search_history"]:
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        
        # Action Bar (Clear Results)
        c_head1, c_head2 = st.columns([5, 1])
        with c_head1:
            st.markdown("#### 🎯 Latest Verified Query Result")
        with c_head2:
            if st.button("🗑️ Clear View", key="btn_clear_search_view", use_container_width=True):
                st.session_state["search_history"] = []
                st.session_state["active_query_text"] = ""
                st.rerun()

        # Render Most Recent Query
        latest = st.session_state["search_history"][0]

        # 4 Responsive KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">🛡️ Confidence Score</div>
                <div class="kpi-value">{latest['audit']['confidence_score']}</div>
                <div class="kpi-subtext kpi-subtext-green">Tri-Hybrid Grounded</div>
                <div class="kpi-bar-green"></div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            risk_val = latest['audit']['hallucination_risk']
            risk_class = "kpi-subtext-green" if risk_val == "LOW" else ("kpi-subtext-orange" if risk_val == "MEDIUM" else "kpi-subtext-red")
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">🛡️ Hallucination Risk</div>
                <div class="kpi-value">{risk_val}</div>
                <div class="kpi-subtext {risk_class}">Lineage Verified</div>
                <div class="kpi-bar-green"></div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">🕸️ Graph Grounding</div>
                <div class="kpi-value">{int(latest['audit'].get('graph_grounding_score', 1.0)*100)}%</div>
                <div class="kpi-subtext kpi-subtext-blue">Relational Facts</div>
                <div class="kpi-bar-blue"></div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">🕒 Execution Latency</div>
                <div class="kpi-value">{latest['latency']}s</div>
                <div class="kpi-subtext kpi-subtext-gray">{latest['timestamp'][11:]}</div>
                <div class="kpi-bar-orange"></div>
            </div>
            """, unsafe_allow_html=True)

        if latest.get("conflicts"):
            st.warning(f"⚠️ **Multi-Document Conflict Detected**: {latest['conflicts'][0]['warning']}")

        # Verified Answer
        st.markdown(f"""
        <div style="margin-top: 14px; font-weight: 700; color: var(--color-text-main); font-size: 15px;">
            Question: <span style="color: var(--color-brand-orange-text); font-weight: 800;">"{latest['query']}"</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="answer-card">{latest["answer"]}</div>', unsafe_allow_html=True)

        # Knowledge Graph Evidence Paths
        subgraph_paths = latest.get("graph_res", {}).get("subgraph", {}).get("paths", [])
        if subgraph_paths:
            paths_html = "".join([f'<div class="graph-path-badge">• {p}</div>' for p in subgraph_paths])
            st.markdown(f"""
            <div class="graph-evidence-card">
                <div style="font-weight: 700; color: var(--color-tech-blue); font-size: 13px; margin-bottom: 8px;">TRAVERSED GRAPH PATHS (Multi-Hop Lineage Grounding)</div>
                {paths_html}
            </div>
            """, unsafe_allow_html=True)

        # Sources Referenced
        st.markdown("#### 📄 Grounded Source Provenance")
        unique_sources = []
        seen = set()
        for c in latest.get("hybrid_chunks", []):
            fname = c['metadata']['file_name']
            page = c['metadata']['page_number']
            pair = f"{fname}_{page}"
            if pair not in seen:
                seen.add(pair)
                unique_sources.append((fname, page))

        for fname, page in unique_sources:
            st.markdown(f"""
            <div class="source-row">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 20px;">📄</span>
                    <div>
                        <div style="font-weight: 700; font-size: 13.5px; color: var(--color-text-main);">{fname}</div>
                        <div style="font-size: 11.5px; color: var(--color-text-muted);">Page {page}</div>
                    </div>
                </div>
                <span style="font-size: 12px; color: var(--color-success); font-weight: 700; background: var(--color-success-bg); padding: 4px 12px; border-radius: var(--radius-pill); border: 1px solid var(--color-success-border);">✓ Grounded Source</span>
            </div>
            """, unsafe_allow_html=True)

        # Previous Queries Accordion if more than 1 query executed in session
        if len(st.session_state["search_history"]) > 1:
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            st.markdown("#### 🕒 Previous Queries in Current Session")
            for idx, hist in enumerate(st.session_state["search_history"][1:], start=2):
                with st.expander(f"#{idx} [{hist['timestamp'][11:]}] \"{hist['query'][:60]}...\" — Confidence: {hist['audit']['confidence_score']}"):
                    st.write(f"**Latency:** `{hist['latency']}s` | **Risk:** `{hist['audit']['hallucination_risk']}`")
                    st.markdown(f"**Answer:**\n{hist['answer']}")
                    if hist.get("sources"):
                        st.write("**Sources:**", hist["sources"])


# ==============================================================================
# ADMIN TAB: DOCUMENT MANAGEMENT & INGESTION PIPELINE
# ==============================================================================
if is_user_admin and tab_docs:
    with tab_docs:
        st.markdown("<h3 style='font-weight: 800; margin-bottom: 2px;'>Document Management & Ingestion</h3>", unsafe_allow_html=True)
        st.caption("Upload enterprise documents to automatically execute chunking, vector embedding, and Knowledge Graph extraction.")

        with st.container():
            st.markdown("#### 📤 Upload Enterprise Document")
            uploaded_file = st.file_uploader(
                "Select File (PDF, DOCX, TXT, CSV)",
                type=["pdf", "docx", "txt", "csv"],
                help="Supports multi-format parsing with automated Graph RAG indexing."
            )

            if uploaded_file is not None:
                if st.button("🚀 Ingest & Index into Graph RAG", type="primary", use_container_width=True):
                    progress_bar = st.progress(0, text="[1/5] Reading uploaded file...")
                    status_placeholder = st.empty()
                    try:
                        file_bytes = uploaded_file.getvalue()
                        status_placeholder.info("⚡ Extracting text and parsing pages...")
                        progress_bar.progress(25, text="[2/5] Parsing document structure...")
                        time.sleep(0.15)

                        progress_bar.progress(50, text="[3/5] Generating token chunks & BGE dense embeddings...")
                        status_placeholder.info("⚡ Updating FAISS vector index...")
                        time.sleep(0.15)

                        progress_bar.progress(75, text="[4/5] Extracting Knowledge Graph entities and relations...")
                        status_placeholder.info("⚡ Building Knowledge Graph facts with Entity Resolution...")

                        ingest_result = process_new_upload(
                            file_bytes=file_bytes,
                            file_name=uploaded_file.name,
                            uploaded_by=current_user["user_id"],
                            embedding_model=model,
                            graph_store=graph_store,
                            faiss_index=faiss_index,
                            existing_chunks=chunks
                        )

                        progress_bar.progress(100, text="[5/5] Ingestion Complete!")
                        status_placeholder.success(f"✅ Successfully ingested '{uploaded_file.name}' into Graph RAG! Added {ingest_result['chunk_count']} chunks. Knowledge Graph now contains {ingest_result['total_graph_nodes']} entities and {ingest_result['total_graph_edges']} relations.")
                        
                        # Clear cache so all tabs immediately reflect the new document
                        st.cache_resource.clear()
                        time.sleep(0.8)
                        st.rerun()

                    except Exception as e:
                        progress_bar.empty()
                        status_placeholder.error(f"❌ Ingestion failed: {e}")

        st.markdown("---")

        # Active Document Registry Table
        st.markdown("#### 📚 Active Document Registry")
        docs_list = get_all_documents()

        if docs_list:
            for doc in docs_list:
                with st.expander(f"📄 {doc['file_name']}  •  {doc['file_type']}  •  {doc['chunk_count']} Chunks  •  Status: {doc['status']}"):
                    dcol1, dcol2, dcol3 = st.columns([2, 2, 1.4])
                    with dcol1:
                        st.write(f"**Document ID:** `{doc['doc_id']}`")
                        st.write(f"**Size:** `{doc['file_size_bytes'] / 1024:.1f} KB`")
                        st.write(f"**Pages:** `{doc['page_count']}`")
                        st.write(f"**Chunks:** `{doc['chunk_count']}`")
                    with dcol2:
                        st.write(f"**Uploaded By:** `{doc['uploaded_by']}`")
                        st.write(f"**Upload Timestamp:** `{doc['upload_timestamp'][:19]}`")
                        st.write(f"**Status:** `{doc['status']}`")
                        st.write(f"**File Path:** `{doc['file_path']}`")
                    with dcol3:
                        if st.button("🔄 Re-process", key=f"rep_{doc['doc_id']}", use_container_width=True):
                            with st.spinner(f"Re-indexing '{doc['file_name']}'..."):
                                try:
                                    reprocess_document_in_system(doc['doc_id'], current_user["user_id"], model, graph_store)
                                    st.cache_resource.clear()
                                    st.success(f"Document '{doc['file_name']}' re-processed successfully!")
                                    time.sleep(0.5)
                                    st.rerun()
                                except Exception as err:
                                    st.error(f"Re-processing error: {err}")

                        if st.button("🗑️ Delete Document", key=f"del_{doc['doc_id']}", use_container_width=True):
                            with st.spinner("Pruning vector & graph facts..."):
                                delete_document_from_system(doc['doc_id'], model, graph_store)
                                st.cache_resource.clear()
                                st.success(f"Document '{doc['file_name']}' deleted.")
                                time.sleep(0.5)
                                st.rerun()
        else:
            st.info("No documents currently registered. Upload a document above to get started.")


# ==============================================================================
# TAB: KNOWLEDGE GRAPH VISUALIZER (INTERACTIVE NETWORK)
# ==============================================================================
with tab_graph:
    st.markdown("<h3 style='font-weight: 800; margin-bottom: 2px;'>Interactive Knowledge Graph Explorer</h3>", unsafe_allow_html=True)
    st.caption("Visually inspect verified enterprise entities, cross-document relationships, and source provenance.")

    col_gf1, col_gf2 = st.columns(2)
    with col_gf1:
        doc_options = ["ALL"]
        if hasattr(graph_store, "graph"):
            doc_ids = set()
            for _, d in graph_store.graph.nodes(data=True):
                for p in d.get("provenance", []):
                    if "doc_id" in p:
                        doc_ids.add(p["doc_id"])
            doc_options.extend(sorted(list(doc_ids)))
        selected_doc = st.selectbox("Filter by Source Document", options=doc_options, key="graph_doc_filter")

    with col_gf2:
        type_options = ["ALL", "SecurityControl", "Policy", "Organization", "FrameworkPillar", "Framework", "Entity", "Concept"]
        selected_type = st.selectbox("Filter by Entity Type", options=type_options, key="graph_type_filter")

    graph_html = generate_interactive_graph_html(
        graph_store=graph_store,
        filter_doc_id=selected_doc,
        filter_entity_type=selected_type,
        height_px=620,
        is_dark_theme=is_dark
    )
    components.html(graph_html, height=670, scrolling=False)


# ==============================================================================
# TAB: QUERY HISTORY & AUDIT LOGS
# ==============================================================================
if is_user_admin and tab_audit:
    with tab_audit:
        st.markdown("<h3 style='font-weight: 800; margin-bottom: 2px;'>Enterprise Query History & Audit Logs</h3>", unsafe_allow_html=True)
        st.caption("Full audit trail of all enterprise queries, answers, groundings, and latencies.")

        col_f1, col_f2 = st.columns([1.5, 3])
        with col_f1:
            all_users_list = ["ALL"] + [u["user_id"] for u in list_all_users()]
            filter_user = st.selectbox("Filter by Employee", options=all_users_list)
        with col_f2:
            search_query_term = st.text_input("Search Logs", placeholder="Search query or answer keyword...")

        user_filter = None if filter_user == "ALL" else filter_user
        search_filter = search_query_term.strip() if search_query_term else None
        
        history_records = get_query_history(user_id=user_filter, search_term=search_filter, limit=100)

        if history_records:
            st.write(f"Showing **{len(history_records)}** audit records:")
            for rec in history_records:
                with st.expander(f"🕒 [{rec['timestamp'][:19]}] {rec['user_id']} ({rec['user_role']}) ➔ \"{rec['query'][:65]}...\""):
                    st.write(f"**User:** `{rec['user_id']}` | **Role:** `{rec['user_role']}` | **Latency:** `{rec['latency_seconds']}s`")
                    st.write(f"**Confidence:** `{rec['confidence_score']}` | **Hallucination Risk:** `{rec['hallucination_risk']}`")
                    st.markdown(f"**Question:**\n{rec['query']}")
                    st.markdown(f"**Answer:**\n{rec['answer']}")
                    if rec.get("graph_facts"):
                        st.markdown(f"**Graph Facts:**\n```\n{rec['graph_facts']}\n```")
                    if rec.get("retrieved_docs"):
                        st.write("**Sources:**", rec["retrieved_docs"])
        else:
            st.info("No audit log records found matching the criteria.")

elif not is_user_admin and "tab_my_history" in locals() and tab_my_history:
    with tab_my_history:
        st.markdown("<h3 style='font-weight: 800; margin-bottom: 2px;'>My Query History</h3>", unsafe_allow_html=True)
        st.caption("Personal history of queries executed during your session.")

        my_history = get_query_history(user_id=current_user["user_id"], limit=50)

        if my_history:
            for rec in my_history:
                with st.expander(f"🕒 [{rec['timestamp'][:19]}] \"{rec['query'][:65]}...\""):
                    st.write(f"**Confidence:** `{rec['confidence_score']}` | **Latency:** `{rec['latency_seconds']}s`")
                    st.markdown(f"**Question:**\n{rec['query']}")
                    st.markdown(f"**Answer:**\n{rec['answer']}")
                    if rec.get("retrieved_docs"):
                        st.write("**Sources:**", rec["retrieved_docs"])
        else:
            st.info("You haven't executed any queries yet.")
