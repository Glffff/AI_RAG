"""Question answering use case implementation."""

from src.rag_service.domain.models import Answer, Question
from src.rag_service.domain.ports import Embedder, LLMService, VectorRepository


class AskQuestionUseCase:
    """Use case for answering questions based on retrieved documents.
    
    Orchestrates the retrieval and answer generation workflow:
    1. Convert question to embedding
    2. Search for relevant chunks in vector store
    3. Generate answer using LLM with context
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_repo: VectorRepository,
        llm_service: LLMService,
    ) -> None:
        """Initialize the use case with required services.
        
        Args:
            embedder: Service to convert text to embeddings.
            vector_repo: Vector database for similarity search.
            llm_service: LLM service for answer generation.
        """
        self.embedder = embedder
        self.vector_repo = vector_repo
        self.llm_service = llm_service

    def execute(self, question: Question) -> Answer:
        """Process a question and generate an answer with citations.
        
        Args:
            question: The user's question.
        
        Returns:
            Answer containing generated text and source chunks.
        """
        question_embedding = self.embedder.get_embedding(question.text)
        related_chunks = self.vector_repo.search(question_embedding)
        answer_text = self.llm_service.generate_answer(question, related_chunks)
        return Answer(text=answer_text, used_chunks=related_chunks)
