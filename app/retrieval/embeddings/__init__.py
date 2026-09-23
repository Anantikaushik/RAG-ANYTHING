from __future__ import annotations

from typing import TYPE_CHECKING

from app.retrieval.embeddings.base import EmbeddingProvider
from app.retrieval.embeddings.deterministic import DeterministicEmbeddingProvider

if TYPE_CHECKING:
    from app.retrieval.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddingProvider,
    )

__all__ = [
    "EmbeddingProvider",
    "DeterministicEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
]


def __getattr__(name: str):
    if name == "SentenceTransformerEmbeddingProvider":
        from app.retrieval.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingProvider,
        )

        return SentenceTransformerEmbeddingProvider

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
