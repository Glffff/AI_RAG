import uuid
from rag_service.domain.ports import Chunker
from rag_service.domain.models import Page, Chunk

class SimpleChunker(Chunker):
    def chunk(self, document_id: str, pages: list[Page]) -> list[Chunk]:
        chunks = []
        for page in pages:
            chunks.append(
                Chunk(
                    id = str(uuid.uuid4()),  
                    document_id=document_id,
                    page_number=page.number, 
                    text=page.content,))
        return chunks