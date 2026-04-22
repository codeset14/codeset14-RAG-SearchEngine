from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from src.core.models import DocumentChunk


class FaissStore:
    def __init__(self, index_dir: str) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "vectors.faiss"
        self.meta_path = self.index_dir / "chunks.json"
        self.index: faiss.Index | None = None
        self.chunks: List[DocumentChunk] = []

    def build(self, chunks: List[DocumentChunk], vectors: List[List[float]]) -> None:
        if not chunks or not vectors:
            self.index = None
            self.chunks = []
            return
        mat = np.array(vectors, dtype="float32")
        dim = mat.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(mat)
        self.index = index
        self.chunks = chunks

    def save(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        serializable = [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "page": c.page,
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        self.meta_path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        self.index = faiss.read_index(str(self.index_path))
        raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.chunks = [DocumentChunk(**row) for row in raw]
        return True

    def search(self, query_vector: List[float], top_k: int) -> List[Tuple[DocumentChunk, float]]:
        if self.index is None or not self.chunks:
            return []
        q = np.array([query_vector], dtype="float32")
        scores, ids = self.index.search(q, top_k)
        results: List[Tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))
        return results
