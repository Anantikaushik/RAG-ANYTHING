from __future__ import annotations

from app.retrieval.embeddings.base import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Real semantic embedding provider using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Text must not be empty.")

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return vector.tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Texts must not contain empty values.")

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return vectors.tolist()

    @property
    def dimensions(self) -> int:
        return self.model.get_sentence_embedding_dimension()