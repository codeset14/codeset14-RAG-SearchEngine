# RAG-SearchEngine
A fully local Retrieval-Augmented Generation (RAG) system that indexes private documents using FAISS and Transformer embeddings, performs semantic search with reranking, and generates grounded answers with a local LLM (Ollama + Llama/Mistral), served via a Streamlit chat interface.

This project implements an end-to-end local Retrieval-Augmented Generation (RAG) pipeline that enables semantic question answering over private files (PDFs, DOCX, source code, and scanned documents via OCR).

The system performs document ingestion, intelligent chunking, dense embedding using Sentence Transformers, vector indexing with FAISS, semantic reranking using a Cross-Encoder, and answer synthesis using a locally hosted LLM via Ollama.

A Streamlit-based chat interface allows users to interactively query their personal knowledge base with full privacy, offline capability, and source-grounded responses.
>>>>>>> 5db0035 (Initial commit: Local RAG Assistant)
