from __future__ import annotations

from app.ingestion.content import ContentItem
from app.ingestion.content_list import ContentList
from app.ingestion.ocr.base import OCRProvider
from app.multimodal.extractors.base import ContentExtractor
from app.multimodal.extractors.equation import EquationExtractor
from app.multimodal.extractors.image import ImageExtractor
from app.multimodal.extractors.table import TableExtractor


class MultimodalProcessor:
    """Route document content through the appropriate extractor."""

    def __init__(
        self,
        extractors: list[ContentExtractor] | None = None,
        ocr_provider: OCRProvider | None = None,
    ) -> None:
        self._extractors = list(
            extractors
            or [
                ImageExtractor(),
                TableExtractor(),
                EquationExtractor(),
            ]
        )
        self._ocr_provider = ocr_provider

    def register_extractor(
        self,
        extractor: ContentExtractor,
    ) -> None:
        self._extractors.append(extractor)

    def process(self, content: ContentList) -> list[dict]:
        results: list[dict] = []

        for item in content:
            processed = self._process_item(item)
            results.append(self._to_result(processed))

        return results

    def process_document(self, document) -> list[dict]:
        """Process normalized document elements for pipeline compatibility."""
        return [
            self._normalized_result(element)
            for element in document.elements
            if element.type != "text"
        ]

    def _process_item(self, item: ContentItem) -> ContentItem:
        if item.type == "text":
            return item

        for extractor in self._extractors:
            if extractor.supports(item):
                return extractor.extract(item)

        raise ValueError(
            f"No multimodal extractor registered for content type: "
            f"{item.type}"
        )

    def extract_image_text(self, image_path: str) -> str:
        """Run OCR on an image when an OCR provider is configured."""
        if self._ocr_provider is None:
            return ""

        return self._ocr_provider.extract_text(image_path)

    @staticmethod
    def _normalized_result(element) -> dict:
        result = {
            "type": element.type,
            "page_idx": element.page_idx,
        }

        if element.payload:
            result.update(element.payload)

        return result

    @staticmethod
    def _to_result(item: ContentItem) -> dict:
        result = {
            "type": item.type,
            "page_idx": item.page_idx,
        }

        if item.type == "text":
            result["text"] = item.text
        elif item.type == "image":
            result["path"] = item.img_path
        elif item.type == "table":
            result["body"] = item.table_body
        elif item.type == "equation":
            result["latex"] = item.latex

        result.update(item.metadata)

        return result