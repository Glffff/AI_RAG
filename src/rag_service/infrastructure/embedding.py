from src.rag_service.domain.ports import Embedder
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddingService(Embedder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()
