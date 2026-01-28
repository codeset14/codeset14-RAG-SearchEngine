from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Cross-encoder for semantic reranking.
        This model jointly encodes (query, chunk) and gives a relevance score.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, top_k=4):
        """
        Args:
            query (str): user question
            chunks (List[Chunk]): retrieved chunks from FAISS
            top_k (int): number of best chunks to keep

        Returns:
            List[Chunk]: top_k most relevant chunks after reranking
        """
        # Prepare query-chunk pairs
        pairs = [(query, c.text) for c in chunks]

        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Sort by score (high → low)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

        # Return top_k chunks only
        return [chunk for chunk, _ in ranked[:top_k]]