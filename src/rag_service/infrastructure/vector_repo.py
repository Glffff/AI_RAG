import numpy as np
from src.rag_service.domain.ports import VectorRepository, Chunk

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

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid 

class QdrantVectorRepository(VectorRepository):
    def __init__(self, top_k: int, collection_name: str, vector_size: int, qdrant_url: str):
        self._top_k = top_k
        self._client = QdrantClient(url=qdrant_url)
        self._collection_name = collection_name

        try: 
            self._client.get_collection(collection_name)
        except Exception:
            self._client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
    
    def save(self, chunk: Chunk, embedding: list[float]) -> None:
        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "page_number": chunk.page_number,
                        "text": chunk.text
                    }
                )
            ]
        )
    
    def search(self, query_embedding: list[float]) -> list[Chunk]:
        search_result = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=self._top_k,
            with_payload=True
        )
        chunks = []
        for result in search_result:
            payload = result.payload
            chunk = Chunk(
                id=payload["chunk_id"],
                document_id=payload["document_id"],
                page_number=payload["page_number"],
                text=payload["text"]
            )
            chunks.append(chunk)
        return chunks