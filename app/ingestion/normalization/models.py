from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NormalizedContent(BaseModel):
    """Stable internal representation of a document content element."""

    element_id: str
    document_id: str

    type: str
    page_idx: int | None = None
    position: int

    text: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    parent_id: str | None = None
    previous_id: str | None = None
    next_id: str | None = None


class NormalizedDocument(BaseModel):
    """Normalized document ready for downstream processing."""

    document_id: str
    filename: str
    elements: list[NormalizedContent] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def element_count(self) -> int:
        return len(self.elements)