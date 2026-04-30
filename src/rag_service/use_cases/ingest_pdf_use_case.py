from src.rag_service.domain.ports import PdfParser, Chunker, Embedder, DocumentRepository, VectorRepository
from src.rag_service.domain.models import Document

class IngestPdfUseCase:
    def __init__(self, 
                 pdf_parser: PdfParser, 
                 chunker: Chunker,
                 embedder: Embedder,
                 document_repo: DocumentRepository,
                 vector_repo: VectorRepository,
                 ):
        self.pdf_parser = pdf_parser
        self.chunker = chunker
        self.embedder = embedder
        self.document_repo = document_repo
        self.vector_repo = vector_repo
    
    def execute(self, pdf_path: str) -> str:
        pages_text = self.pdf_parser.parse(pdf_path)
        
        document = Document.from_file(pdf_path, pages_text)
        chunks = self.chunker.chunk(document.filename, pages_text)
        document.add_chunks(chunks)
        
        self.document_repo.save_document(document)

        for chunk in chunks:
            embedding = self.embedder.get_embedding(chunk.text)
            self.vector_repo.save(chunk, embedding)

        return document.id