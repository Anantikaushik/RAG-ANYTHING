from __future__ import annotations

from typing import Iterable

from app.ingestion.content import ContentItem


class ContentList:
    """Container for structured multimodal document content."""

    def __init__(self, items: Iterable[ContentItem] | None = None):
        self.items = list(items or [])

    def add(self, item: ContentItem) -> None:
        self.items.append(item)

    def to_list(self) -> list[dict]:
        return [
            item.model_dump(exclude_none=True)
            for item in self.items
        ]

    def by_type(self, content_type: str) -> list[ContentItem]:
        return [
            item
            for item in self.items
            if item.type == content_type
        ]

    @property
    def text_items(self) -> list[ContentItem]:
        return self.by_type("text")

    @property
    def multimodal_items(self) -> list[ContentItem]:
        return [
            item
            for item in self.items
            if item.type != "text"
        ]

    @classmethod
    def from_list(cls, items: list[dict]) -> "ContentList":
        return cls(
            ContentItem.model_validate(item)
            for item in items
        )

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)