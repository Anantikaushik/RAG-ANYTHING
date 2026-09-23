from __future__ import annotations

from app.ingestion.normalization.models import NormalizedContent
from app.multimodal.analysis.base import MultimodalAnalyzer
from app.multimodal.analysis.models import MultimodalAnalysis


class DeterministicImageAnalyzer(MultimodalAnalyzer):
    """Baseline image analyzer without an external model."""

    @property
    def supported_type(self) -> str:
        return "image"

    def analyze(
        self,
        content: NormalizedContent,
    ) -> MultimodalAnalysis:
        return MultimodalAnalysis(
            content_id=content.element_id,
            content_type="image",
            summary="Image content",
            metadata={
                "img_path": content.payload.get("img_path"),
            },
        )


class DeterministicTableAnalyzer(MultimodalAnalyzer):
    """Baseline table analyzer without an external model."""

    @property
    def supported_type(self) -> str:
        return "table"

    def analyze(
        self,
        content: NormalizedContent,
    ) -> MultimodalAnalysis:
        table_body = content.payload.get("table_body")

        return MultimodalAnalysis(
            content_id=content.element_id,
            content_type="table",
            summary="Table content",
            extracted_text=table_body,
        )


class DeterministicEquationAnalyzer(MultimodalAnalyzer):
    """Baseline equation analyzer without an external model."""

    @property
    def supported_type(self) -> str:
        return "equation"

    def analyze(
        self,
        content: NormalizedContent,
    ) -> MultimodalAnalysis:
        latex = content.payload.get("latex")

        return MultimodalAnalysis(
            content_id=content.element_id,
            content_type="equation",
            summary="Mathematical equation",
            extracted_text=latex,
        )