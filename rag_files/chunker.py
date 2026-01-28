from typing import List
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int

class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_documents(self, documents: List) -> List[Chunk]:
        all_chunks = []
        chunk_counter = 0

        for doc in documents:
            splits = self.splitter.split_text(doc.text)

            for split in splits:
                all_chunks.append(
                    Chunk(
                        text=split,
                        source=doc.source,
                        chunk_id=chunk_counter
                    )
                )
                chunk_counter += 1

        return all_chunks