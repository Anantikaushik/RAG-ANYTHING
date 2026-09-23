from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for converting text into embedding vectors."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]