import numpy as np
from rag_service.domain.ports import VectorRepository, Chunk

class MemoryVectorRepository(VectorRepository):
    def __init__(self, top_k: int = 3):
        self._vectors = []
        self._top_k = top_k
    
    def save(self, chunk: Chunk, embedding: list[float]) -> None:
        self._vectors.append({
            "chunk": chunk,
            "embedding": embedding
        })
    
    def search(self, query_embedding: list[float]) -> list[Chunk]:
        query = np.array(query_embedding)
        score_item = []
        for item in self._vectors:
            score = self._get_cosine_similarity(query, np.array(item["embedding"]))
            score_item.append((score, item["chunk"]))
        score_item.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in score_item[:self._top_k]]
    
    def _get_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            return 0.0
        return np.dot(a, b) / denominator