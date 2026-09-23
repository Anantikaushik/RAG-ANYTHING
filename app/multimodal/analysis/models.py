from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MultimodalAnalysis(BaseModel):
    """Semantic analysis produced for a multimodal content item."""

    content_id: str
    content_type: str

    summary: str | None = None

    extracted_text: str | None = None

    entities: list[str] = Field(default_factory=list)

    relationships: list[dict[str, Any]] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )