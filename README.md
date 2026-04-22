# Agentic RAG Web App (Streamlit + Groq)

Production-ready prototype of a multi-agent Retrieval-Augmented Generation system using:
- Streamlit frontend
- Groq LLM (`llama-3.3-70b-versatile`)
- FAISS vector search
- Hybrid retrieval (semantic + keyword)
- Re-ranking + validation loop to minimize hallucinations

## Features
- Upload and index `PDF`, `TXT`, `CSV` files
- Chunking + embedding pipeline
- Iterative retrieval refinement for complex questions
- Multi-agent workflow:
  - Query Analyzer Agent
  - Retriever Agent
  - Reasoning Agent
  - Validator Agent
  - Tool Decision Agent
- Confidence scoring and source citations
- Conversation memory
- Debug panel with traces:
  - analyzer output
  - retrieval iterations
  - decisions and validator feedback

## Folder Structure
```text
RAG/
├─ app.py
├─ requirements.txt
├─ .env.example
├─ README.md
└─ src/
   ├─ __init__.py
   ├─ config/
   │  └─ settings.py
   ├─ core/
   │  ├─ logging.py
   │  └─ models.py
   ├─ llm/
   │  └─ groq_client.py
   ├─ ingestion/
   │  ├─ loaders.py
   │  └─ chunker.py
   ├─ embeddings/
   │  └─ encoder.py
   ├─ retrieval/
   │  ├─ faiss_store.py
   │  ├─ keyword.py
   │  └─ reranker.py
   ├─ memory/
   │  └─ conversation.py
   ├─ agents/
   │  ├─ query_analyzer.py
   │  ├─ retriever_agent.py
   │  ├─ reasoning_agent.py
   │  ├─ validator_agent.py
   │  └─ tool_decision_agent.py
   ├─ orchestration/
   │  ├─ indexing.py
   │  └─ agentic_rag.py
   └─ utils/
      ├─ cache.py
      ├─ prompts.py
      ├─ scoring.py
      └─ text.py
```

## Setup
1. Create a virtual environment:
   - Windows PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Configure environment:
   - Copy `.env.example` to `.env`
   - Set `GROQ_API_KEY`

## Run
```powershell
streamlit run app.py
```

## Example Usage
1. Open the Streamlit app.
2. Upload one or more `PDF`/`TXT`/`CSV` files from the sidebar.
3. Click **Index Documents**.
4. Ask:
   - simple query: `What is the warranty period?`
   - multi-hop query: `Compare policy A and policy B, then summarize key risks.`
5. Inspect:
   - source citations under each answer
   - confidence bar
   - debug panel for retrieval steps and validation decisions

## Notes on Hallucination Control
- Answers are generated only from retrieved evidence.
- Validator checks claim support and can trigger re-retrieval.
- Control loop retries until confidence threshold is met or max iterations is reached.
