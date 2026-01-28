import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingIndexer:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        index_path: str = "vector_store"
    ):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.index = None
        self.chunks = None

    def build_index(self, chunks: List):
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self.chunks = chunks

    def save(self):
        os.makedirs(self.index_path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(self.index_path, "faiss.index"))

        with open(os.path.join(self.index_path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self):
        self.index = faiss.read_index(os.path.join(self.index_path, "faiss.index"))

        with open(os.path.join(self.index_path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

    def search(self, query: str, top_k: int = 5):
        query_vec = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results