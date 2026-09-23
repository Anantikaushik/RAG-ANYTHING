from __future__ import annotations

from abc import ABC, abstractmethod

from app.ingestion.normalization.models import NormalizedContent
from app.multimodal.analysis.models import MultimodalAnalysis


class MultimodalAnalyzer(ABC):
    """Contract for semantic analysis of multimodal content."""

    @property
    @abstractmethod
    def supported_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def analyze(
        self,
        content: NormalizedContent,
    ) -> MultimodalAnalysis:
        raise NotImplementedError

    def supports(
        self,
        content: NormalizedContent,
    ) -> bool:
        return content.type == self.supported_type