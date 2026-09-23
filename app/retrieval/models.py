from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """
    A single candidate returned by a retrieval strategy.

    The result keeps the original graph/content identity so later
    retrieval stages can merge, rank, and deduplicate candidates.
    """

    item_id: str
    score: float = 0.0
    source: str
    content_type: str | None = None
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    """Collection of ranked retrieval candidates."""

    query: str
    results: list[RetrievalResult] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.results)