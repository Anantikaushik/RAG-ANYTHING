from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class GraphEdge(BaseModel):
    """A directed relationship between two graph nodes."""

    edge_id: str = Field(default_factory=lambda: str(uuid4()))

    source_id: str
    target_id: str

    relation: str

    properties: dict[str, Any] = Field(
        default_factory=dict
    )