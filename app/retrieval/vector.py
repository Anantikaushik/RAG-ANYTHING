from __future__ import annotations

from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.models import RetrievalResponse, RetrievalResult
from app.retrieval.vector_store import InMemoryVectorStore


class VectorRetriever:
    """Retrieves content using embedding similarity."""

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResponse:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_vector = self.embedding_provider.embed(query)

        matches = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
        )

        results = [
            RetrievalResult(
                item_id=record.item_id,
                score=score,
                source="vector",
                content_type=record.metadata.get("content_type"),
                text=record.text,
                metadata=record.metadata,
            )
            for record, score in matches
        ]

        return RetrievalResponse(
            query=query,
            results=results,
        )