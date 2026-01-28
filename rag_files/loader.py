import os
from typing import List
from dataclasses import dataclass


import fitz # PyMuPDF
import pytesseract
from PIL import Image
from docx import Document

@dataclass
class DocumentChunk:
    text: str
    source: str

class LocalDocumentLoader:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.supported_code_ext = {".py", ".js", ".java", ".cpp", ".c", ".cs", ".go", ".rs", ".sql", ".html", ".css"}
        self.supported_text_ext = {".txt", ".md"}
        self.supported_docx_ext = {".docx"}
        self.supported_pdf_ext = {".pdf"}

    def load_all(self) -> List[DocumentChunk]:
        documents = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                if ext in self.supported_code_ext or ext in self.supported_text_ext:
                    documents.append(self._load_text_file(path))

                elif ext in self.supported_docx_ext:
                    documents.append(self._load_docx(path))

                elif ext in self.supported_pdf_ext:
                    documents.extend(self._load_pdf(path))

        return documents

    def _load_text_file(self, path: str) -> DocumentChunk:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return DocumentChunk(text=text, source=path)

    def _load_docx(self, path: str) -> DocumentChunk:
        doc = Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return DocumentChunk(text=text, source=path)

    def _load_pdf(self, path: str) -> List[DocumentChunk]:
        chunks = []
        pdf = fitz.open(path)

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text().strip()

            # If text exists → normal PDF
            if text:
                chunks.append(DocumentChunk(
                    text=text,
                    source=f"{path} | page {page_num+1}"
                ))
            # Else → scanned image PDF (OCR)
            else:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img)

                chunks.append(DocumentChunk(
                    text=ocr_text,
                    source=f"{path} | page {page_num+1} (OCR)"
                ))

        return chunks