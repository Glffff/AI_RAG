from abc import ABC, abstractmethod
from typing import List
from .models import Page, Document, Chunk, Question


class PdfParser(ABC):
    """Abstract base class for PDF parsing implementations.
    
    Defines the interface for extracting pages from PDF documents.
    """
    
    @abstractmethod
    def parse(self, file_path: str) -> List[Page]:
        """Extract pages from a PDF file.
        
        Args:
            file_path: Path to the PDF file to parse.
        
        Returns:
            List of extracted pages.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("PDF parsing not implemented")


class Chunker(ABC):
    """Abstract base class for text chunking implementations.
    
    Defines the interface for splitting document pages into smaller, manageable chunks.
    """
    
    @abstractmethod
    def chunk(self, filename: str, pages: List[Page]) -> List[Chunk]:
        """Split pages into chunks for embedding and retrieval.
        
        Args:
            filename: Name of the source document.
            pages: List of pages to chunk.
        
        Returns:
            List of text chunks extracted from pages.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Text chunking not implemented")


class Embedder(ABC):
    """Abstract base class for embedding service implementations.
    
    Defines the interface for converting text to vector embeddings.
    """
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Generate a vector embedding for the given text.
        
        Args:
            text: Text to embed.
        
        Returns:
            Vector embedding of fixed dimension.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Embedding service not implemented")


class DocumentRepository(ABC):
    """Abstract base class for document storage implementations.
    
    Defines the interface for persisting and retrieving documents.
    """
    
    @abstractmethod
    def save_document(self, document: Document) -> None:
        """Save a document to the repository.
        
        Args:
            document: The document to save.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Document repository not implemented")


class VectorRepository(ABC):
    """Abstract base class for vector database implementations.
    
    Defines the interface for storing and searching vector embeddings.
    """
    
    @abstractmethod
    def save(self, chunk: Chunk, embedding: List[float]) -> None:
        """Save a chunk with its embedding vector.
        
        Args:
            chunk: The text chunk to store.
            embedding: The vector embedding of the chunk.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Vector repository not implemented")
    
    @abstractmethod
    def search(self, query_embedding: List[float]) -> List[Chunk]:
        """Search for chunks similar to the query embedding.
        
        Args:
            query_embedding: Query vector to search with.
        
        Returns:
            List of most similar chunks, ordered by relevance.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Vector search not implemented")


class LLMService(ABC):
    """Abstract base class for large language model service implementations.
    
    Defines the interface for generating answers based on context chunks.
    """
    
    @abstractmethod
    def generate_answer(self, question: Question, context_chunks: List[Chunk]) -> str:
        """Generate an answer to a question using provided context chunks.
        
        Args:
            question: The user's question.
            context_chunks: Relevant context chunks for the question.
        
        Returns:
            The generated answer text.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("LLM service not implemented")
