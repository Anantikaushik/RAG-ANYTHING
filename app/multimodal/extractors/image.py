from __future__ import annotations

from app.ingestion.content import ContentItem
from app.multimodal.extractors.base import ContentExtractor


class ImageExtractor(ContentExtractor):
    @property
    def content_type(self) -> str:
        return "image"

    def extract(self, item: ContentItem) -> ContentItem:
        if not self.supports(item):
            raise ValueError(
                f"ImageExtractor cannot process content type: {item.type}"
            )

        return item