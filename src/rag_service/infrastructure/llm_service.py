from rag_service.domain.ports import LLMService
from rag_service.domain.models import Question, Chunk
from typing import List
import ollama

class OllamaLLMService(LLMService):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_answer(self, question: Question, context_chunks: List[Chunk]) -> str:
        context = "\n".join(f"[Page {chunk.page_number}: {chunk.text}]" for chunk in context_chunks)
        
        prompt = f"""Please answer the following question based on the provided context:\n\n, say no if you don't know the answer.\n
                     Context: {context}
                     Question: {question.text}
                  """
        
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response["message"]["content"]