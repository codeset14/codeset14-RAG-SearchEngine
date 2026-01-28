class Retriever:
    def __init__(self, indexer):
        self.indexer = indexer

    def retrieve(self, query: str, top_k: int = 5):
        return self.indexer.search(query, top_k=top_k)