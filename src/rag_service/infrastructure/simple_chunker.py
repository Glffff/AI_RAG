"""Simple text chunking implementation."""

import uuid

from src.rag_service.domain.models import Chunk, Page
from src.rag_service.domain.ports import Chunker


class SimpleChunker(Chunker):
    """Text chunker using a sliding window approach.
    
    Splits document pages into overlapping chunks of fixed size, useful for
    embedding and retrieval operations.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Initialize the chunker with size and overlap parameters.
        
        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of overlapping characters between adjacent chunks.
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, filename: str, pages: list[Page]) -> list[Chunk]:
        """Split pages into chunks.
        
        Args:
            filename: Name of the source document.
            pages: List of pages to chunk.
        
        Returns:
            List of text chunks extracted from all pages.
        """
        chunks = []
        for page in pages:
            page_chunks = self._split_text(page.content, page.number, filename)
            chunks.extend(page_chunks)
        return chunks

    def _split_text(
        self, text: str, page_number: int, filename: str
    ) -> list[Chunk]:
        """Split a single text into overlapping chunks using sliding window.
        
        Args:
            text: Text content to split.
            page_number: Page number for reference.
            filename: Source filename for reference.
        
        Returns:
            List of chunks created from the text.
        """
        chunks = []

        if len(text) <= self._chunk_size:
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    filename=filename,
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
                    filename=filename,
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