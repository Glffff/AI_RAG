from rag_service.domain.ports import VectorRepository, Chunk

class MemoryVectorRepository(VectorRepository):
    def __init__(self):
        self.vectors = []
    
    def save(self, chunk: Chunk, embedding: list[float]) -> None:
        self.vectors.append({
            "chunk": chunk,
            "embedding": embedding
        })
    
    def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        # For simplicity, return the first top_k chunks (no real similarity search)
        return [vector["chunk"] for vector in self.vectors[:top_k]]