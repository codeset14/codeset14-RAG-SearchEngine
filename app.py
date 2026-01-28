import streamlit as st
from rag_files.loader import LocalDocumentLoader
from rag_files.chunker import DocumentChunker
from rag_files.indexer import EmbeddingIndexer
from rag_files.retriever import Retriever
from rag_files.reranker import Reranker
from rag_files.generator import RAGGenerator

st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
st.title("📚 Local RAG Chatbot")

@st.cache_resource
def load_backend():
    indexer = EmbeddingIndexer(index_path="vector_store")
    indexer.load()
    return Retriever(indexer), Reranker(), RAGGenerator("phi3:mini")

retriever, reranker, generator = load_backend()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask from your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        retrieved = retriever.retrieve(query, top_k=10)
        best = reranker.rerank(query, retrieved, top_k=4)
        answer = generator.generate(query, best)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})