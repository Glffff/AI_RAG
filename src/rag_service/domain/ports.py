from abc import ABC, abstractmethod
from typing import List
from .models import Page, Document, Chunk, Question

class PdfParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> List[Page]:
        raise NotImplementedError("PDF parsing not implemented")

class Chunker(ABC):
    @abstractmethod
    def chunk(self, document_id: str, pages: List[Page]) -> List[Chunk]:
        raise NotImplementedError("Text chunking not implemented")
    
class Embedder(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        raise NotImplementedError("Embedding service not implemented")

class DocumentRepository(ABC):
    @abstractmethod
    def save_document(self, document: Document):
        raise NotImplementedError("Document repository not implemented")

class VectorRepository(ABC):
    @abstractmethod
    def save(self, chunk: Chunk, embedding: List[float]):
        raise NotImplementedError("Vector repository not implemented")
    
    @abstractmethod
    def search(self, query_embedding: List[float]) -> List[Chunk]:
        raise NotImplementedError("Vector search not implemented")

class LLMService(ABC):
    @abstractmethod
    def generate_answer(self, question: Question, context_chunks: List[Chunk]) -> str:
        raise NotImplementedError("LLM service not implemented")