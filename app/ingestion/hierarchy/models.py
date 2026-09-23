from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PageNode(BaseModel):
    """Represents a document page."""

    page_id: str
    document_id: str
    page_idx: int

    element_ids: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class SectionNode(BaseModel):
    """Represents a logical section."""

    section_id: str
    document_id: str

    title: str | None = None

    element_ids: list[str] = Field(default_factory=list)
    page_ids: list[str] = Field(default_factory=list)

    parent_id: str | None = None


class DocumentHierarchy(BaseModel):
    """Hierarchical representation of a normalized document."""

    document_id: str

    page_nodes: list[PageNode] = Field(default_factory=list)
    section_nodes: list[SectionNode] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.page_nodes)

    @property
    def section_count(self) -> int:
        return len(self.section_nodes)