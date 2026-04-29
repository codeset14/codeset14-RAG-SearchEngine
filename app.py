from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from src.agents.query_analyzer import QueryAnalyzerAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.retriever_agent import RetrieverAgent
from src.agents.tool_decision_agent import ToolDecisionAgent
from src.agents.validator_agent import ValidatorAgent
from src.config.settings import load_settings
from src.core.logging import setup_logging
from src.embeddings.encoder import EmbeddingEncoder
from src.llm.groq_client import GroqClient
from src.memory.conversation import ConversationMemory
from src.orchestration.agentic_rag import AgenticRAGOrchestrator
from src.orchestration.indexing import IndexingService
from src.retrieval.faiss_store import FaissStore
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.reranker import CrossEncoderReranker


@st.cache_resource(show_spinner=False)
def build_runtime():
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    setup_logging(logging.INFO)
    settings = load_settings()
    if not settings.groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in .env.")
    if not settings.groq_api_key.startswith("gsk_"):
        raise ValueError("Invalid GROQ_API_KEY format. A Groq key should start with 'gsk_'.")

    encoder = EmbeddingEncoder(settings.embedding_model)
    faiss_store = FaissStore(settings.index_dir)
    faiss_store.load()
    keyword = KeywordRetriever()
    if faiss_store.chunks:
        keyword.fit(faiss_store.chunks)
    reranker = CrossEncoderReranker(settings.reranker_model)
    llm = GroqClient(settings.groq_api_key, settings.groq_model)
    memory = ConversationMemory()

    indexing = IndexingService(
        encoder=encoder,
        faiss_store=faiss_store,
        keyword_retriever=keyword,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    orchestrator = AgenticRAGOrchestrator(
        settings=settings,
        analyzer=QueryAnalyzerAgent(llm),
        retriever=RetrieverAgent(encoder, faiss_store, keyword, reranker),
        reasoner=ReasoningAgent(llm),
        validator=ValidatorAgent(llm),
        decider=ToolDecisionAgent(),
        memory=memory,
    )
    return settings, indexing, orchestrator


def _uploads_to_payload(files) -> List[Tuple[str, bytes]]:
    return [(f.name, f.getvalue()) for f in files]


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background: radial-gradient(circle at top right, #2e1b5b 0%, #0b1220 42%, #070b14 100%);
            color: #e5e7eb;
        }
        .top-header {
            position: sticky;
            top: 0;
            z-index: 30;
            backdrop-filter: blur(12px);
            background: rgba(17, 25, 40, 0.55);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            padding: 16px 20px;
            margin-bottom: 14px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.25);
        }
        .title-gradient {
            background: linear-gradient(90deg, #60a5fa, #a78bfa 45%, #22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
        }
        .subtitle {
            color: #cbd5e1;
            font-size: 0.9rem;
            margin-top: 4px;
        }
        .status-badge {
            display: inline-block;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.2);
            padding: 6px 10px;
            border-radius: 999px;
            margin-left: 8px;
            font-size: 0.82rem;
        }
        .glass-card {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(148,163,184,0.2);
            border-radius: 16px;
            padding: 14px;
            box-shadow: 0 12px 25px rgba(0,0,0,0.25);
        }
        .answer-card {
            animation: fadeIn 0.45s ease-in-out;
            background: rgba(15, 23, 42, 0.62);
            border: 1px solid rgba(96,165,250,0.35);
            border-radius: 16px;
            padding: 14px;
            margin-bottom: 8px;
        }
        .source-card {
            border: 1px solid rgba(148,163,184,0.2);
            border-radius: 12px;
            padding: 10px;
            margin: 8px 0;
            background: rgba(2, 6, 23, 0.45);
        }
        .tag {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 0.72rem;
            margin-right: 6px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.16);
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(8px);}
            to {opacity: 1; transform: translateY(0);}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(model_name: str, status_ok: bool) -> None:
    status = "🟢 Active" if status_ok else "🔴 Error"
    st.markdown(
        f"""
        <div class="top-header">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
                <div>
                    <h1 class="title-gradient">Agentic RAG</h1>
                    <div class="subtitle">Multi-Agent Retrieval • Reasoning • Validation</div>
                </div>
                <div>
                    <span class="status-badge">Model: {model_name}</span>
                    <span class="status-badge">{status}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(settings, indexing_service, orchestrator) -> None:
    with st.sidebar:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📁 Upload & Index")
        uploads = st.file_uploader(
            "Drag and drop your files",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True,
        )
        st.subheader("⚙️ Controls")
        top_k = st.slider("Top-K", 3, 20, int(settings.top_k))
        max_iterations = st.slider("Max iterations", 1, 6, int(settings.max_iterations))
        settings.top_k = top_k
        settings.max_iterations = max_iterations

        if st.button("🚀 Index Documents", use_container_width=True, type="primary"):
            if not uploads:
                st.warning("Upload at least one file before indexing.")
            else:
                payload = _uploads_to_payload(uploads)
                with st.spinner("Building index..."):
                    chunks = indexing_service.build_from_uploads(payload)
                st.success(f"Indexed {len(chunks)} chunks.")

        if st.button("🧹 Clear Memory", use_container_width=True):
            st.session_state.messages = []
            orchestrator.memory.clear()
            st.toast("Conversation memory cleared.")

        st.markdown('</div>', unsafe_allow_html=True)


def render_chat(orchestrator) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = None

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role_label = "You" if msg["role"] == "user" else "Assistant"
        bubble_bg = "linear-gradient(120deg,#2563eb,#7c3aed)" if msg["role"] == "user" else "rgba(30,41,59,0.75)"
        align = "flex-end" if msg["role"] == "user" else "flex-start"
        st.markdown(
            f"""
            <div style="display:flex; justify-content:{align}; margin:8px 0;">
                <div style="max-width:82%; background:{bubble_bg}; border:1px solid rgba(255,255,255,0.12);
                            padding:10px 12px; border-radius:14px; box-shadow:0 6px 18px rgba(0,0,0,0.22);">
                    <div style="font-size:0.75rem; opacity:0.85; margin-bottom:4px;">{role_label}</div>
                    <div>{msg["content"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    user_query = st.chat_input("Ask anything from your documents...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        step_placeholder = st.empty()
        try:
            with step_placeholder.container():
                st.info("Analyzing query...")
            with step_placeholder.container():
                st.info("Retrieving documents...")
            with step_placeholder.container():
                st.info("Reasoning...")
            with step_placeholder.container():
                st.info("Validating answer...")

            response = asyncio.run(orchestrator.run(user_query))
            step_placeholder.empty()
            st.session_state.messages.append({"role": "assistant", "content": response.answer})
            st.session_state.last_trace = response.trace
            st.session_state.last_sources = response.citations
            st.session_state.last_confidence = response.confidence
            st.rerun()
        except Exception as exc:
            st.toast("Groq request failed. Check key/network and retry.", icon="⚠️")
            if st.button("Retry last query", use_container_width=True):
                st.rerun()
            st.error(f"Details: {exc}")

    if st.session_state.get("messages") and st.session_state["messages"][-1]["role"] == "assistant":
        answer = st.session_state["messages"][-1]["content"]
        st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
        confidence = float(st.session_state.get("last_confidence", 0.0))
        st.progress(min(max(confidence, 0.0), 1.0), text=f"Confidence Score: {confidence:.2f}")

        with st.expander("Sources", expanded=True):
            sources = st.session_state.get("last_sources", [])
            if not sources:
                st.info("No citations available for this response.")
            else:
                for c in sources[:8]:
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <b>{c.chunk.source}</b> • page {c.chunk.page} • score {c.combined_score:.2f}<br/>
                            <span style="color:#cbd5e1;">{c.chunk.text[:240]}{"..." if len(c.chunk.text) > 240 else ""}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_debug_panel() -> None:
    open_debug = st.toggle("Open Debug Drawer", value=st.session_state.get("debug_open", False))
    st.session_state.debug_open = open_debug
    if not open_debug:
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Debug Drawer")
    st.markdown('<span class="tag">🧠 reasoning</span><span class="tag">🔍 retrieval</span><span class="tag">✅ validation</span>', unsafe_allow_html=True)

    trace = st.session_state.get("last_trace")
    if not trace:
        st.info("Run a query to view debug traces.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    tabs = st.tabs(["Analyzer", "Retrieval", "Reasoning", "Validator"])
    with tabs[0]:
        analyzer = trace["analyzer"] if isinstance(trace, dict) else (trace.analyzer.__dict__ if trace.analyzer else {})
        st.json(analyzer)
    with tabs[1]:
        if isinstance(trace, dict):
            st.json(trace.get("retrieval", []))
        else:
            serial = [{"iteration": s.iteration, "query_used": s.query_used, "notes": s.notes} for s in trace.retrieval_steps]
            st.json(serial)
    with tabs[2]:
        if isinstance(trace, dict):
            st.json(trace.get("reasoning", {}))
        else:
            st.json(trace.reasoning.__dict__ if trace.reasoning else {})
    with tabs[3]:
        if isinstance(trace, dict):
            st.json(trace.get("validator", {}))
        else:
            st.json(trace.validation.__dict__ if trace.validation else {})
    st.markdown("</div>", unsafe_allow_html=True)


def render_config_error(exc: Exception) -> None:
    st.markdown(
        f"""
        <div style="max-width:680px; margin:50px auto; text-align:center; padding:24px; border-radius:18px;
                    border:1px solid rgba(248,113,113,0.35); background:rgba(30,10,20,0.5);">
            <h2 style="margin:0 0 10px 0;">⚠️ Configuration Issue</h2>
            <p style="color:#fecaca;">{exc}</p>
            <p style="color:#cbd5e1;">Create or fix your <code>.env</code> and add a valid Groq API key.</p>
            <pre style="text-align:left; background:rgba(2,6,23,0.7); padding:10px; border-radius:10px;">
GROQ_API_KEY=gsk_your_real_key_here
GROQ_MODEL=llama-3.3-70b-versatile
            </pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Agentic RAG with Groq", layout="wide")
    inject_theme()

    try:
        settings, indexing_service, orchestrator = build_runtime()
    except Exception as exc:
        render_header("llama-3.3-70b-versatile", status_ok=False)
        render_config_error(exc)
        st.stop()

    render_header(settings.groq_model, status_ok=True)
    render_sidebar(settings, indexing_service, orchestrator)

    chat_col, debug_col = st.columns([3, 1.4], gap="large")
    with chat_col:
        render_chat(orchestrator)
    with debug_col:
        render_debug_panel()


if __name__ == "__main__":
    main()
