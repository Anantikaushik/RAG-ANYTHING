from __future__ import annotations

from abc import ABC, abstractmethod

from app.ingestion.content import ContentItem


class ContentExtractor(ABC):
    """Base interface for multimodal content extractors."""

    @property
    @abstractmethod
    def content_type(self) -> str:
        """Return the content type handled by this extractor."""
        raise NotImplementedError

    def supports(self, item: ContentItem) -> bool:
        """Return whether this extractor can process the item."""
        return item.type == self.content_type

    @abstractmethod
    def extract(self, item: ContentItem) -> ContentItem:
        """Process and return the content item."""
        raise NotImplementedError