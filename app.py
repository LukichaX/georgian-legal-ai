"""
app.py
=============================================================================
Georgian Legal AI — Enterprise Legal Research & Conversational RAG Interface
Minimalist, high-precision technical UI (No gradients, no emojis).
=============================================================================
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

import dotenv
import streamlit as st

# უზრუნველყოფს მოდულებისა და .env-ის პოვნას როგორც root, ისე src/ საქაღალდიდან
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "src" else _THIS_DIR
SRC_DIR = PROJECT_ROOT / "src"

for _p in (PROJECT_ROOT, SRC_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

dotenv.load_dotenv(PROJECT_ROOT / ".env")
dotenv.load_dotenv(SRC_DIR / ".env")

from legal_agent import LegalAgentPipeline

# =============================================================================
# 1. გვერდის კონფიგურაცია და მკაცრი ინდუსტრიული სტილი (CSS)
# =============================================================================
st.set_page_config(
    page_title="Georgian Legal AI | Statutory Research System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Header styling */
    .system-header {
        border-bottom: 1px solid rgba(128, 128, 128, 0.22);
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .system-title {
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.3;
    }
    .system-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #71717a;
        margin-top: 0.35rem;
    }

    /* Telemetry Bar under assistant messages */
    .telemetry-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.75rem;
        margin-bottom: 0.35rem;
        padding-top: 0.65rem;
        border-top: 1px solid rgba(128, 128, 128, 0.15);
    }
    .telemetry-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 0.2rem 0.55rem;
        border-radius: 3px;
        border: 1px solid rgba(128, 128, 128, 0.28);
        background-color: rgba(128, 128, 128, 0.06);
        letter-spacing: 0.01em;
    }
    .pill-active {
        border-color: rgba(59, 130, 246, 0.45);
        background-color: rgba(59, 130, 246, 0.08);
    }
    .pill-verified {
        border-color: rgba(16, 185, 129, 0.45);
        background-color: rgba(16, 185, 129, 0.08);
    }
    .pill-warn {
        border-color: rgba(239, 68, 68, 0.45);
        background-color: rgba(239, 68, 68, 0.08);
    }

    /* Legal Source Cards */
    .source-card {
        border: 1px solid rgba(128, 128, 128, 0.24);
        border-radius: 4px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        background-color: rgba(128, 128, 128, 0.03);
    }
    .source-card-cited {
        border-left: 3px solid #10b981;
    }
    .source-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        color: #71717a;
        margin-bottom: 0.35rem;
    }
    .source-title {
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .source-breadcrumb {
        font-size: 0.78rem;
        color: #71717a;
    }

    /* Sidebar refinements */
    .sidebar-section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #71717a;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }
    .memory-item {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        padding: 0.35rem 0.5rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 3px;
        margin-bottom: 0.3rem;
        background-color: rgba(128, 128, 128, 0.04);
    }

    /* Button alignment */
    div.stButton > button {
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 500;
        text-align: left;
        justify-content: flex-start;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CODE_LABELS: Dict[str | None, str] = {
    "constitution": "საქართველოს კონსტიტუცია",
    "criminal": "სისხლის სამართლის კოდექსი",
    "criminal_proc": "სისხლის სამართლის საპროცესო კოდექსი",
    "civil": "სამოქალაქო კოდექსი",
    "civil_proc": "სამოქალაქო საპროცესო კოდექსი",
    "labor": "შრომის კოდექსი",
    "gen_admin": "ზოგადი ადმინისტრაციული კოდექსი",
    "admin_proc": "ადმინისტრაციული საპროცესო კოდექსი",
    "admin_offenses": "ადმინისტრაციულ სამართალდარღვევათა კოდექსი",
    "tax": "საგადასახადო კოდექსი",
    "customs": "საბაჟო კოდექსი",
    "election": "საარჩევნო კოდექსი",
    "imprisonment": "პატიმრობის კოდექსი",
    "juvenile": "არასრულწლოვანთა მართლმსაჯულების კოდექსი",
    "child_rights": "ბავშვის უფლებათა კოდექსი",
    "spatial_building": "სივრცითი და სამშენებლო კოდექსი",
    "product_safety": "პროდუქტის უსაფრთხოების კოდექსი",
    "forest": "ტყის კოდექსი",
    "env_assessment": "გარემოსდაცვითი შეფასების კოდექსი",
    "air": "საჰაერო კოდექსი",
    "maritime": "საზღვაო კოდექსი",
    None: "ყველა კოდექსი (21 აქტი)",
}

INTENT_LABELS: Dict[str, str] = {
    "direct_chat": "DIRECT_DIALOGUE",
    "follow_up_existing": "MEMORY_CONTEXT_REUSE",
    "exact_lookup": "TIER_0_SQL_LOOKUP",
    "hybrid_search": "TIER_1_2_HYBRID_RRF",
    "out_of_scope": "OUT_OF_SCOPE_GUARD",
}


# =============================================================================
# 2. აგენტის ინიციალიზაცია (Singleton GPU Resource)
# =============================================================================
@st.cache_resource(show_spinner="მიმდინარეობს საძიებო და მარშრუტიზაციის მოდელების ჩატვირთვა GPU მეხსიერებაში...")
def get_legal_agent() -> LegalAgentPipeline:
    return LegalAgentPipeline()


# =============================================================================
# 3. სესიების მართვა
# =============================================================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"sess-{uuid.uuid4().hex[:6]}"

if "sessions" not in st.session_state:
    st.session_state.sessions: Dict[str, List[Dict[str, Any]]] = {
        st.session_state.thread_id: []
    }

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


def create_new_session() -> None:
    new_id = f"sess-{uuid.uuid4().hex[:6]}"
    st.session_state.thread_id = new_id
    st.session_state.sessions[new_id] = []


# =============================================================================
# 4. ტექნიკური ტელემეტრიისა და წყაროების ინსპექტორი
# =============================================================================
def render_turn_telemetry(meta: Dict[str, Any]) -> None:
    decision = meta.get("router_decision") or {}
    timings = meta.get("timings") or {}
    articles = meta.get("retrieved_articles") or []
    citation = meta.get("citation_status") or {}

    intent = decision.get("intent", "hybrid_search")
    code_slug = decision.get("code_slug")
    used_search = intent in ("exact_lookup", "hybrid_search")
    total_sec = timings.get("total_seconds", 0.0)
    cited_list = citation.get("cited_markers") or []
    guard_ok = citation.get("is_valid", True)

    retrieval_mode_str = "DB_RETRIEVAL" if used_search else "ZERO_SEARCH_MEMORY"
    mode_class = "pill-active" if used_search else ""

    if intent in ("direct_chat", "out_of_scope"):
        cite_str = "CITATIONS: N/A"
        cite_class = ""
    elif guard_ok and cited_list:
        cite_str = f"CITATIONS: {', '.join(cited_list)} VERIFIED"
        cite_class = "pill-verified"
    else:
        cite_str = "CITATIONS: UNVERIFIED"
        cite_class = "pill-warn"

    code_upper = (code_slug or "GLOBAL").upper()

    st.markdown(
        f"""
        <div class="telemetry-bar">
            <span class="telemetry-pill {mode_class}">MODE: {retrieval_mode_str}</span>
            <span class="telemetry-pill">PIPELINE: {INTENT_LABELS.get(intent, intent.upper())}</span>
            <span class="telemetry-pill">DOMAIN: {code_upper}</span>
            <span class="telemetry-pill {cite_class}">{cite_str}</span>
            <span class="telemetry-pill">LATENCY: {total_sec:.2f}s</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    expander_label = f"ტექნიკური ტელემეტრია და ნორმატიული წყაროები ({len(articles)} მუხლი)"
    with st.expander(expander_label, expanded=False):
        tab_sources, tab_trace, tab_audit = st.tabs([
            f"ნორმატიული წყაროები ({len(articles)})",
            "მარშრუტიზაცია და დროები",
            "ციტირების აუდიტი",
        ])

        with tab_sources:
            if not articles:
                st.caption("მიმდინარე სვლაზე საძიებო ბაზიდან ახალი მუხლების ამოღება არ მოთხოვნილა.")
            else:
                for idx, art in enumerate(articles, 1):
                    marker = f"E{idx}"
                    is_cited = marker in cited_list
                    card_cls = "source-card source-card-cited" if is_cited else "source-card"
                    status_txt = "CITED IN ANSWER" if is_cited else "RETRIEVED CANDIDATE"
                    score_val = art.get("rerank_score", art.get("score", 0.0))

                    st.markdown(
                        f"""
                        <div class="{card_cls}">
                            <div class="source-meta-row">
                                <span>[{marker}] ID: {art.get('node_id', '')} | {art.get('retrieval_tier', '')} | SCORE: {score_val:.4f}</span>
                                <span>{status_txt}</span>
                            </div>
                            <div class="source-title">{art.get('title', '')}</div>
                            <div class="source-breadcrumb">{art.get('breadcrumb', '')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander(f"[{marker}] ნორმატიული ტექსტის სრული ვერსია ({art.get('node_id', '')})"):
                        st.text(art.get("body_text", ""))

        with tab_trace:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Intent Classification", intent)
            c2.metric("Target Code", code_slug or "global (21 codes)")
            c3.metric("Explicit Article", decision.get("article_no") or "None")
            c4.metric("Database Query", "Executed" if used_search else "Bypassed (0 ms)")

            st.markdown("**ფორმულირებული საძიებო შეკითხვა (Standalone Search Query):**")
            st.code(decision.get("search_query", ""), language="text")

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("1. Router Node", f"{timings.get('router_seconds', 0.0) * 1000:.0f} ms")
            t2.metric("2. Retrieval Node", f"{timings.get('retrieval_seconds', 0.0):.2f} s")
            t3.metric("3. Generator Node", f"{timings.get('generation_seconds', 0.0):.2f} s")
            t4.metric("4. Citation Guard", f"{timings.get('guard_seconds', 0.0) * 1000:.1f} ms")

        with tab_audit:
            st.json(citation)


# =============================================================================
# 5. Sidebar: სესიების მართვა, აქტიური კონტექსტი და სატესტო სცენარები
# =============================================================================
agent = get_legal_agent()
current_thread = st.session_state.thread_id
session_state = agent.get_session_state(current_thread)
active_articles = session_state.get("active_articles") or []
active_code = session_state.get("active_code_slug")

with st.sidebar:
    st.markdown("### GEORGIAN LEGAL AI")
    st.caption("Statutory RAG System | v4.0 Production")

    if st.button("ახალი სესიის დაწყება", use_container_width=True, type="primary"):
        create_new_session()
        st.rerun()

    all_threads = list(st.session_state.sessions.keys())
    if len(all_threads) > 1:
        selected_thread = st.selectbox(
            "აქტიური სესიები",
            options=all_threads,
            index=all_threads.index(current_thread),
        )
        if selected_thread != current_thread:
            st.session_state.thread_id = selected_thread
            st.rerun()

    st.markdown('<div class="sidebar-section-label">სესიის კონტექსტური მეხსიერება</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="memory-item">ACTIVE CODE: {(active_code or "NONE").upper()}<br>'
        f'<span style="color:#71717a;">{CODE_LABELS.get(active_code, "ყველა კოდექსი")}</span></div>',
        unsafe_allow_html=True,
    )

    if active_articles:
        for i, a in enumerate(active_articles, 1):
            nid = a.get("node_id", "")
            title_short = a.get("title", "")[:42]
            st.markdown(
                f'<div class="memory-item">[E{i}] {nid}<br>'
                f'<span style="color:#71717a;">{title_short}...</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("მეხსიერებაში აქტიური მუხლები არ არის.")

    st.markdown('<div class="sidebar-section-label">სავერიფიკაციო სცენარები</div>', unsafe_allow_html=True)

    verification_scenarios = [
        (
            "01 / Tier 0 SQL: სსკ მუხლი 108",
            "რა წერია სისხლის სამართლის კოდექსის 108-ე მუხლში?",
        ),
        (
            "02 / Memory Reuse: მუხლის განმარტება",
            "შეგიძლია ეს მუხლი უფრო მარტივი ენით ამიხსნა და მოკლე მაგალითი მომიყვანო?",
        ),
        (
            "03 / Contextual Rewrite: დამამძიმებელი გარემოება",
            "და თუ იგივე დანაშაული დამამძიმებელ გარემოებებში, მაგალითად ჯგუფურად ან ანგარებით მოხდა, მაშინ რა სასჯელია?",
        ),
        (
            "04 / Colloquial Bridge: სქესობრივი თანხმობის ასაკი",
            "საქართველოში რომელი წლიდან არის დაშვებული სექსი?",
        ),
        (
            "05 / Homonym Guard: კაზუსი 'მუხლი 39-ჯერ'",
            "დამკითხეს სასტიკად, მცემდნენ და მაიძულებდნენ მუხლი 39-ჯერ მიმერტყა იატაკზე აღიარების მისაღებად. რა დანაშაულია?",
        ),
        (
            "06 / Tax Code: საშემოსავლო გადასახადი",
            "რამდენი პროცენტია ფიზიკური პირის საშემოსავლო გადასახადის განაკვეთი საქართველოში?",
        ),
    ]

    for btn_label, prompt_text in verification_scenarios:
        if st.button(btn_label, use_container_width=True):
            st.session_state.pending_prompt = prompt_text
            st.rerun()

    st.markdown('<div class="sidebar-section-label">სისტემური პარამეტრები</div>', unsafe_allow_html=True)
    st.caption(
        "CORPUS: 21 Codes (5,624 Articles)\n\n"
        "INDEX: SQLite FTS5 Trigram + ChromaDB\n\n"
        "EMBEDDER: Qwen3-8B (GGUF Q8_0)\n\n"
        "RERANKER: Qwen3-0.6B Cross-Encoder\n\n"
        "ROUTER: Qwen3.5-4B Local GBNF"
    )


# =============================================================================
# 6. მთავარი სამუშაო არეალი (Main Workspace)
# =============================================================================
st.markdown(
    f"""
    <div class="system-header">
        <div class="system-title">საქართველოს კანონმდებლობის ანალიტიკური სისტემა</div>
        <div class="system-subtitle">
            CORPUS: 21 STATUTORY CODES (5,624 ARTICLES) &nbsp;|&nbsp;
            ARCHITECTURE: MULTI-TIER HYBRID RAG &nbsp;|&nbsp;
            SESSION: {current_thread.upper()}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

messages = st.session_state.sessions.get(current_thread, [])

if not messages:
    st.markdown(
        """
        <div class="source-card">
            <div class="source-meta-row"><span>SYSTEM READY</span><span>21 CODES INDEXED</span></div>
            <div class="source-title">სამართლებრივი ძიებისა და კონვერსაციული ანალიზის რეჟიმი</div>
            <div class="source-breadcrumb">
                დასვით შეკითხვა საქართველოს კონსტიტუციისა და 20 მოქმედი კოდექსის გარშემო,
                მოითხოვეთ კონკრეტული მუხლი ნომრით ან აღწერეთ ფაქტობრივი გარემოება (კაზუსი).
                ყველა სამართლებრივი დასკვნა მოწმდება დეტერმინისტული [E1]–[E4] ციტირების ფილტრით.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            render_turn_telemetry(msg["meta"])

user_input = st.chat_input("შეიყვანეთ სამართლებრივი შეკითხვა, მუხლის ნომერი ან კაზუსი...")
if st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_input:
    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("მიმდინარეობს კონტექსტის მარშრუტიზაცია, ძიება და ციტირების ვერიფიკაცია..."):
            result = agent.run(user_input, thread_id=current_thread)

        answer_text = result.get("answer", "")
        st.markdown(answer_text)

        meta_data = {
            "router_decision": result.get("router_decision"),
            "retrieved_articles": result.get("retrieved_articles"),
            "citation_status": result.get("citation_status"),
            "timings": result.get("timings"),
        }
        render_turn_telemetry(meta_data)

    messages.append({
        "role": "assistant",
        "content": answer_text,
        "meta": meta_data,
    })
    st.rerun()