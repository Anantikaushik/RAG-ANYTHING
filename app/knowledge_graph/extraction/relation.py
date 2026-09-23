from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedRelation(BaseModel):
    """A semantic relationship extracted from document content."""

    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation: str
    properties: dict = Field(default_factory=dict)