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


def main() -> None:
    st.set_page_config(page_title="Agentic RAG with Groq", layout="wide")
    st.title("Agentic RAG Web App (Streamlit + Groq)")
    st.caption("Multi-agent retrieval, reasoning, validation, and grounded answering.")

    try:
        settings, indexing_service, orchestrator = build_runtime()
    except Exception as exc:
        st.error(
            "Configuration error while starting app.\n\n"
            f"Details: {exc}\n\n"
            "Fix `.env` with a valid Groq key and restart Streamlit."
        )
        st.code(
            "GROQ_API_KEY=gsk_your_real_key_here\n"
            "GROQ_MODEL=llama-3.3-70b-versatile",
            language="bash",
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = None

    with st.sidebar:
        st.subheader("Document Indexing")
        st.write("Upload PDF, TXT, CSV files.")
        uploads = st.file_uploader(
            "Knowledge files",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True,
        )
        if st.button("Index Documents", use_container_width=True):
            if not uploads:
                st.warning("Upload at least one file.")
            else:
                payload = _uploads_to_payload(uploads)
                with st.spinner("Indexing documents..."):
                    chunks = indexing_service.build_from_uploads(payload)
                st.success(f"Index built with {len(chunks)} chunks.")
        st.divider()
        st.write(f"Model: `{settings.groq_model}`")
        st.write(f"Top-K: `{settings.top_k}`")
        st.write(f"Max iterations: `{settings.max_iterations}`")
        if st.button("Clear Chat Memory", use_container_width=True):
            st.session_state.messages = []
            orchestrator.memory.clear()
            st.success("Conversation memory cleared.")

    col_chat, col_debug = st.columns([2.2, 1.2], gap="large")
    with col_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("Ask a question over indexed documents...")
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Running agentic loop..."):
                    try:
                        response = asyncio.run(orchestrator.run(user_query))
                    except Exception as exc:
                        st.error(
                            "Groq request failed.\n\n"
                            f"Details: {exc}\n\n"
                            "This is usually an invalid/revoked API key. Update `GROQ_API_KEY` in `.env`, "
                            "then fully restart Streamlit."
                        )
                        return
                st.markdown(response.answer)
                st.progress(min(max(response.confidence, 0.0), 1.0), text=f"Confidence: {response.confidence:.2f}")
                with st.expander("Sources / Citations", expanded=True):
                    for idx, c in enumerate(response.citations[:8], start=1):
                        st.markdown(
                            f"**[{idx}]** `{c.chunk.source}` (p.{c.chunk.page}) "
                            f"- combined `{c.combined_score:.3f}`"
                        )
                        st.caption(c.chunk.text[:320] + ("..." if len(c.chunk.text) > 320 else ""))
                st.session_state.messages.append({"role": "assistant", "content": response.answer})
                st.session_state.last_trace = response.trace

    with col_debug:
        st.subheader("Debug Panel")
        trace = st.session_state.last_trace
        if not trace:
            st.info("Run a query to view agent decisions and traces.")
        else:
            with st.expander("Analyzer", expanded=True):
                st.json(trace.analyzer.__dict__ if trace.analyzer else {})
            with st.expander("Decisions", expanded=True):
                st.json([d.__dict__ for d in trace.decisions])
            with st.expander("Retrieval Steps", expanded=False):
                serial = []
                for step in trace.retrieval_steps:
                    serial.append(
                        {
                            "iteration": step.iteration,
                            "query_used": step.query_used,
                            "notes": step.notes,
                            "results": [
                                {
                                    "chunk_id": c.chunk.chunk_id,
                                    "source": c.chunk.source,
                                    "semantic": c.semantic_score,
                                    "keyword": c.keyword_score,
                                    "rerank": c.rerank_score,
                                    "combined": c.combined_score,
                                }
                                for c in step.chunks
                            ],
                        }
                    )
                st.json(serial)
            with st.expander("Reasoning Trace", expanded=False):
                st.write(trace.reasoning.reasoning_trace if trace.reasoning else "None")
                if trace.reasoning:
                    st.json(
                        {
                            "claims": trace.reasoning.claims,
                            "claim_support": trace.reasoning.claim_support,
                        }
                    )
            with st.expander("Validator", expanded=False):
                st.json(trace.validation.__dict__ if trace.validation else {})


if __name__ == "__main__":
    main()
