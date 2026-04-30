from src.rag_service.domain.models import Question, Answer
from src.rag_service.domain.ports import Embedder, VectorRepository, LLMService

class AskQuestionUseCase:
    def __init__(self,
                 embedder: Embedder,
                 vector_repo: VectorRepository,
                 llm_service: LLMService,
                 ):
        self.embedder = embedder
        self.vector_repo = vector_repo
        self.llm_service = llm_service
    
    def execute(self, question: Question) -> Answer:
        question_embedding = self.embedder.get_embedding(question.text)
        related_chunks = self.vector_repo.search(question_embedding)
        answer_text = self.llm_service.generate_answer(question, related_chunks)
        return Answer(text=answer_text, used_chunks=related_chunks)