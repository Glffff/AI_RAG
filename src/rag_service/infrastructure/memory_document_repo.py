"""In-memory document repository implementation."""

from src.rag_service.domain.models import Document
from src.rag_service.domain.ports import DocumentRepository


class MemoryDocumentRepository(DocumentRepository):
    """In-memory storage for PDF documents.
    
    Stores documents in a Python dictionary. Data is lost when the application
    restarts. Suitable for development and testing only.
    """

    def __init__(self) -> None:
        """Initialize the in-memory document store."""
        self.documents: dict[str, Document] = {}

    def save_document(self, document: Document) -> None:
        """Save a document to the in-memory store.
        
        Args:
            document: The document to save.
        """
        self.documents[document.id] = document

    def get_document(self, document_id: str) -> Document:
        """Retrieve a document by its ID.
        
        Args:
            document_id: The unique identifier of the document.
        
        Returns:
            The requested document.
        
        Raises:
            KeyError: If document with the given ID does not exist.
        """
        return self.documents[document_id]
