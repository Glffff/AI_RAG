"""PDF ingestion use case implementation."""

from src.rag_service.domain.models import Document
from src.rag_service.domain.ports import (
    Chunker,
    DocumentRepository,
    Embedder,
    PdfParser,
    VectorRepository,
)


class IngestPdfUseCase:
    """Use case for processing and storing PDF documents.
    
    Orchestrates the document ingestion workflow:
    1. Parse PDF to extract pages
    2. Split content into chunks
    3. Generate embeddings for each chunk
    4. Store document and vectors for later retrieval
    """

    def __init__(
        self,
        pdf_parser: PdfParser,
        chunker: Chunker,
        embedder: Embedder,
        document_repo: DocumentRepository,
        vector_repo: VectorRepository,
    ) -> None:
        """Initialize the use case with required services.
        
        Args:
            pdf_parser: Service to extract pages from PDF files.
            chunker: Service to split pages into text chunks.
            embedder: Service to generate vector embeddings.
            document_repo: Repository to store document metadata.
            vector_repo: Repository to store chunk embeddings and vectors.
        """
        self.pdf_parser = pdf_parser
        self.chunker = chunker
        self.embedder = embedder
        self.document_repo = document_repo
        self.vector_repo = vector_repo

    def execute(self, pdf_path: str) -> str:
        """Process a PDF file and store its content for retrieval.
        
        Args:
            pdf_path: Path to the PDF file to process.
        
        Returns:
            Document ID for later reference.
        """
        pages_text = self.pdf_parser.parse(pdf_path)

        document = Document.from_file(pdf_path, pages_text)
        chunks = self.chunker.chunk(document.filename, pages_text)
        document.add_chunks(chunks)

        self.document_repo.save_document(document)

        for chunk in chunks:
            embedding = self.embedder.get_embedding(chunk.text)
            self.vector_repo.save(chunk, embedding)

        return document.id
