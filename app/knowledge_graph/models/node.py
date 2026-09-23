from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


NodeType = Literal[
    "document",
    "page",
    "section",
    "content",
    "entity",
]


class GraphNode(BaseModel):
    """A node in the knowledge graph."""

    node_id: str
    node_type: NodeType

    label: str | None = None

    properties: dict[str, Any] = Field(
        default_factory=dict
    )

    @property
    def is_document(self) -> bool:
        return self.node_type == "document"

    @property
    def is_content(self) -> bool:
        return self.node_type == "content"

    @property
    def is_entity(self) -> bool:
        return self.node_type == "entity"