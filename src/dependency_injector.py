from src.rag_service.use_cases.ask_question_use_case import AskQuestionUseCase
from src.rag_service.use_cases.ingest_pdf_use_case import IngestPdfUseCase

from src.rag_service.infrastructure.embedding import SentenceTransformerEmbeddingService
from src.rag_service.infrastructure.vector_repo import MemoryVectorRepository, QdrantVectorRepository
from src.rag_service.infrastructure.llm_service import OllamaLLMService
from src.rag_service.infrastructure.pdf_parser import PymupdfParser
from src.rag_service.infrastructure.simple_chunker import SimpleChunker
from src.rag_service.infrastructure.memory_document_repo import MemoryDocumentRepository  

from src.config import settings

class DependencyInjector:
    def __init__(self):
        self._services = {}
        self._services["vector_repo"] = QdrantVectorRepository(
            top_k=settings.top_k_chunks,
            collection_name=settings.collection_name,
            vector_size=settings.qdrant_vector_size,
            qdrant_url=settings.qdrant_url
        )
        self._services["embedding_service"] = SentenceTransformerEmbeddingService(
            model_name=settings.embedding_model
        )
        self._services["llm_service"] = OllamaLLMService(
            model_name=settings.llm_model, 
            base_url=settings.llm_base_url
        )
        #self._services["vector_repo"] = MemoryVectorRepository(top_k=settings.top_k_chunks)
        
    def create_ask_question_use_case(self) -> AskQuestionUseCase:
        return AskQuestionUseCase(
            self._services["embedding_service"], 
            self._services["vector_repo"], 
            self._services["llm_service"]
        )
    
    def create_ingest_pdf_use_case(self):
        return IngestPdfUseCase(
            PymupdfParser(), 
            SimpleChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap), 
            self._services["embedding_service"], 
            MemoryDocumentRepository(),
            self._services["vector_repo"]
        )