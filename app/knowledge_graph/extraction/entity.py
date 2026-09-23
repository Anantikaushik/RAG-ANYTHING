from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """A semantic entity extracted from document content."""

    entity_id: str
    name: str
    entity_type: str
    source_content_id: str
    properties: dict = Field(default_factory=dict)