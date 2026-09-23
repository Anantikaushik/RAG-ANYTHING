from __future__ import annotations

import hashlib
import math

from app.retrieval.embeddings.base import EmbeddingProvider


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """
    Lightweight deterministic embeddings for tests and local development.

    The same text always produces the same vector.
    """

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero.")

        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Text must not be empty.")

        values: list[float] = []

        for index in range(self.dimensions):
            digest = hashlib.sha256(
                f"{index}:{text}".encode("utf-8")
            ).digest()

            value = int.from_bytes(digest[:8], "big") / (2**64)
            values.append((value * 2.0) - 1.0)

        magnitude = math.sqrt(sum(value * value for value in values))

        if magnitude == 0:
            return [0.0] * self.dimensions

        return [value / magnitude for value in values]