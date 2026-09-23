from __future__ import annotations

from app.ingestion.content import ContentItem
from app.multimodal.extractors.base import ContentExtractor


class EquationExtractor(ContentExtractor):
    @property
    def content_type(self) -> str:
        return "equation"

    def extract(self, item: ContentItem) -> ContentItem:
        return item