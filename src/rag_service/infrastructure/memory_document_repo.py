from rag_service.domain.models import Document
from rag_service.domain.ports import DocumentRepository

class MemoryDocumentRepository(DocumentRepository):
    def __init__(self):
        self.documents = {}
    
    def save_document(self, document: Document) -> None:
        self.documents[document.id] = document
    
    def get_document(self, document_id: str) -> Document:
        return self.documents[document_id]