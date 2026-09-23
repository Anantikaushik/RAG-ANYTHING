from __future__ import annotations

from app.ingestion.content import ContentItem
from app.multimodal.extractors.base import ContentExtractor


class TableExtractor(ContentExtractor):
    @property
    def content_type(self) -> str:
        return "table"

    def extract(self, item: ContentItem) -> ContentItem:
        return item