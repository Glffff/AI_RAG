"""Sentence Transformer based embedding service implementation."""

from src.rag_service.domain.ports import Embedder
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingService(Embedder):
    """Embedding service using Sentence Transformers library.
    
    Provides vector embeddings for text using pre-trained sentence transformer models.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the embedding service with a specific model.
        
        Args:
            model_name: Name of the sentence transformer model (e.g., 'all-MiniLM-L6-v2').
        """
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> list[float]:
        """Generate a vector embedding for the given text.
        
        Args:
            text: Text to encode into a vector embedding.
        
        Returns:
            Vector embedding as a list of floats.
        """
        return self.model.encode(text).tolist()

