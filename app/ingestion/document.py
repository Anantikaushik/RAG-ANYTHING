from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from app.ingestion.content import ContentItem


class Document(BaseModel):
    """
    Internal representation of an ingested document.
    """

    document_id: str

    source_path: str

    filename: str

    content: list[ContentItem] = Field(
        default_factory=list,
    )

    metadata: dict = Field(
        default_factory=dict,
    )

    @property
    def path(self) -> Path:
        return Path(self.source_path)

    @property
    def page_count(self) -> int:
        pages = {
            item.page_idx
            for item in self.content
            if item.page_idx is not None
        }

        return len(pages)

    @property
    def content_count(self) -> int:
        return len(self.content)

    def add_content(
        self,
        item: ContentItem,
    ) -> None:
        self.content.append(item)
        