from typing import List
from .models import Page, Document, Chunk

class PdfParser:
    def parse(self, file_path: str) -> List[Page]:
        raise NotImplementedError("PDF parsing not implemented")

class Chunker:
    def chunk(self, pages: List[Page]) -> List[Chunk]:
        raise NotImplementedError("Text chunking not implemented")
    
class Embedder:
    def get_embedding(self, text: str) -> List[float]:
        raise NotImplementedError("Embedding service not implemented")

class DocumentRepository:
    def save_document(self, document: Document):
        raise NotImplementedError("Document repository not implemented")

class VectorRepository:
    def save(self, chunk: Chunk, embedding: List[float]):
        raise NotImplementedError("Vector repository not implemented")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Chunk]:
        raise NotImplementedError("Vector search not implemented")