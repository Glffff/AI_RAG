import uuid
from src.rag_service.domain.ports import Chunker
from src.rag_service.domain.models import Page, Chunk

class SimpleChunker(Chunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
    
    def chunk(self, document_id: str, pages: list[Page]) -> list[Chunk]:
        chunks = []
        for page in pages:
            page_chunks = self._split_text(page.content, page.number, document_id)
            chunks.extend(page_chunks)
        return chunks
    
    def _split_text(self, text: str, page_number: int, document_id: str) -> list[Chunk]:
        chunks = []
        
        if len(text) <= self._chunk_size:
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    page_number=page_number,
                    text=text,
                )
            )
            return chunks
        
        # Calculate step size (distance between chunk starts)
        step_size = self._chunk_size - self._chunk_overlap
        
        start = 0
        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    page_number=page_number,
                    text=chunk_text,
                )
            )
            
            # Move to next chunk start position
            start += step_size
            
            # Stop if we've reached the end
            if end == len(text):
                break
        
        return chunks