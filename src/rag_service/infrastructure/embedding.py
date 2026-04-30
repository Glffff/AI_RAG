from rag_service.domain.ports import Embedder

class FakeEmbedder(Embedder):
    def get_embedding(self, text: str) -> list[float]:
        return [
            0.1,
            0.2,
            0.3
        ]