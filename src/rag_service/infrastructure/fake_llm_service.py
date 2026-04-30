from rag_service.domain.ports import LLMService
from rag_service.domain.models import Question, Chunk
from typing import List

class FakeLLMService(LLMService):
    def generate_answer(self, question: Question, context_chunks: List[Chunk]) -> str:
        context = "\n".join(chunk.text for chunk in context_chunks)
        return (
            f"Fake answer to: '{question.text}'\n\n"        
            f"Based on context:\n{context}"
        )