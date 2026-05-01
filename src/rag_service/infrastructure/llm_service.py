"""Ollama-based LLM service implementation."""

from typing import List

import ollama

from src.rag_service.domain.models import Chunk, Question
from src.rag_service.domain.ports import LLMService


class OllamaLLMService(LLMService):
    """LLM service using Ollama running locally or remotely.
    
    Generates answers to questions based on provided context chunks
    using a specified Ollama model.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the LLM service with a specific Ollama model.
        
        Args:
            model_name: Name of the Ollama model to use (e.g., 'mistral', 'llama3.1:8b').
        """
        self._model_name = model_name

    def generate_answer(self, question: Question, context_chunks: List[Chunk]) -> str:
        """Generate an answer to a question using provided context.
        
        Args:
            question: The user's question.
            context_chunks: Relevant context chunks to base the answer on.
        
        Returns:
            Generated answer text from the LLM.
        """
        context = "\n".join(
            f"[Page {chunk.page_number}: {chunk.text}]" for chunk in context_chunks
        )

        prompt = f"""Please answer the following question based on the provided context:

                Context: {context}
                Question: {question.text}

                If you don't know the answer based on the context, say so.
                """

        response = ollama.chat(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
