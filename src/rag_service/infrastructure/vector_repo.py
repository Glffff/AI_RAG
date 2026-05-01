"""Vector storage and search implementations."""

import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.rag_service.domain.models import Chunk
from src.rag_service.domain.ports import VectorRepository


class MemoryVectorRepository(VectorRepository):
    """In-memory vector storage using NumPy for similarity search.
    
    Stores embeddings and chunks in memory with cosine similarity search.
    Data is lost when the application restarts. Suitable for development and testing only.
    """

    def __init__(self, top_k: int = 3) -> None:
        """Initialize the in-memory vector store.
        
        Args:
            top_k: Number of top similar chunks to return in search results.
        """
        self._vectors: list[dict] = []
        self._top_k = top_k

    def save(self, chunk: Chunk, embedding: list[float]) -> None:
        """Save a chunk with its embedding vector.
        
        Args:
            chunk: The text chunk to store.
            embedding: The vector embedding of the chunk.
        """
        self._vectors.append({"chunk": chunk, "embedding": embedding})

    def search(self, query_embedding: list[float]) -> list[Chunk]:
        """Search for chunks similar to the query embedding.
        
        Args:
            query_embedding: Query vector to search with.
        
        Returns:
            List of top_k most similar chunks ordered by similarity.
        """
        query = np.array(query_embedding)
        score_item = []
        for item in self._vectors:
            score = self._get_cosine_similarity(
                query, np.array(item["embedding"])
            )
            score_item.append((score, item["chunk"]))
        score_item.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in score_item[: self._top_k]]

    def _get_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            a: First vector.
            b: Second vector.
        
        Returns:
            Cosine similarity score between -1 and 1.
        """
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            return 0.0
        return np.dot(a, b) / denominator


class QdrantVectorRepository(VectorRepository):
    """Vector storage using Qdrant database.
    
    Persists embeddings in Qdrant for scalable and efficient similarity search.
    Suitable for production environments with large-scale data.
    """

    def __init__(
        self, top_k: int, collection_name: str, vector_size: int, qdrant_url: str
    ) -> None:
        """Initialize connection to Qdrant and create collection if needed.
        
        Args:
            top_k: Number of top similar chunks to return in search results.
            collection_name: Name of the Qdrant collection to use.
            vector_size: Dimension of the embedding vectors.
            qdrant_url: URL of the Qdrant server (e.g., 'http://localhost:6333').
        """
        self._top_k = top_k
        self._client = QdrantClient(url=qdrant_url)
        self._collection_name = collection_name

        try:
            self._client.get_collection(collection_name)
        except Exception:
            self._client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def save(self, chunk: Chunk, embedding: list[float]) -> None:
        """Save a chunk with its embedding vector to Qdrant.
        
        Args:
            chunk: The text chunk to store.
            embedding: The vector embedding of the chunk.
        """
        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.id,
                        "document_id": chunk.filename,
                        "page_number": chunk.page_number,
                        "text": chunk.text,
                        "score": chunk.score,
                    },
                )
            ],
        )

    def search(self, query_embedding: list[float]) -> list[Chunk]:
        """Search for chunks similar to the query embedding in Qdrant.
        
        Args:
            query_embedding: Query vector to search with.
        
        Returns:
            List of top_k most similar chunks with scores.
        """
        search_result = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=self._top_k,
            with_payload=True,
        )
        chunks = []
        for point in search_result.points:
            payload = point.payload

            if payload is None:
                continue

            chunk = Chunk(
                id=payload["chunk_id"],
                filename=payload["document_id"],
                page_number=payload["page_number"],
                text=payload["text"],
                score=point.score,
            )
            chunks.append(chunk)
        return chunks
