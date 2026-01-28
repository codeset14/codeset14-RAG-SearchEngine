from rag_files.loader import LocalDocumentLoader
from rag_files.chunker import DocumentChunker
from rag_files.indexer import EmbeddingIndexer
from rag_files.retriever import Retriever
from rag_files.reranker import Reranker
from rag_files.generator import RAGGenerator


from huggingface_hub import login 



hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN not found in environment")

def build_index(data_path="data", index_path="vector_store"):
    print("📂 Loading documents...")
    loader = LocalDocumentLoader(data_path)
    documents = loader.load_all()

    print("✂️ Chunking documents...")
    chunker = DocumentChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.chunk_documents(documents)

    print("🔢 Creating embeddings + FAISS index...")
    indexer = EmbeddingIndexer(index_path=index_path)
    indexer.build_index(chunks)
    indexer.save()

    print("✅ Vector store built and saved.")


def load_rag(index_path="vector_store"):
    print("📥 Loading FAISS index...")
    indexer = EmbeddingIndexer(index_path=index_path)
    indexer.load()

    retriever = Retriever(indexer)
    reranker = Reranker()
    generator = RAGGenerator(model_name="llama3.1")

    return retriever, reranker, generator


def chat_loop():
    retriever, reranker, generator = load_rag()

    print("\n🤖 Local RAG Chatbot Ready (type 'exit' to quit)\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        print("\n🔍 Retrieving relevant chunks...")
        retrieved = retriever.retrieve(query, top_k=10)

        print("🎯 Reranking for semantic relevance...")
        best_chunks = reranker.rerank(query, retrieved, top_k=4)

        print("🧠 Generating answer with LLM...\n")
        answer = generator.generate(query, best_chunks)

        print("Assistant:\n", answer)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    """
    First time run:
        build_index("your_data_folder")

    Then comment build_index and run chat.
    """

    # Run once to build FAISS index
    #build_index(data_path=r"C:\Users\Viraj Sawant\OneDrive\Desktop\6 Months Goal")

    # Run chatbot
    chat_loop()