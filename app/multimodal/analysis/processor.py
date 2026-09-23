from __future__ import annotations

from collections.abc import Iterable

from app.multimodal.analysis.base import MultimodalAnalyzer
from app.multimodal.analysis.deterministic import (
    DeterministicEquationAnalyzer,
    DeterministicImageAnalyzer,
    DeterministicTableAnalyzer,
)
from app.multimodal.analysis.models import MultimodalAnalysis


class MultimodalAnalysisProcessor:
    """
    Coordinates multimodal analyzers and caches their results.

    The processor is intentionally independent of any specific
    VLM provider. Analyzers can be injected at construction time
    or registered later.

    Analysis results can be reused by:
    - semantic graph processing
    - multimodal graph processing
    - vector indexing
    """

    def __init__(
        self,
        analyzers: Iterable[MultimodalAnalyzer] | None = None,
    ) -> None:

        if analyzers is None:
            analyzers = [
                DeterministicImageAnalyzer(),
                DeterministicTableAnalyzer(),
                DeterministicEquationAnalyzer(),
            ]

        self._analyzers: list[MultimodalAnalyzer] = list(
            analyzers
        )

        self._cache: dict[str, MultimodalAnalysis] = {}

    @property
    def analyzers(self) -> tuple[MultimodalAnalyzer, ...]:
        """Return the currently registered analyzers."""
        return tuple(self._analyzers)

    def register(
        self,
        analyzer: MultimodalAnalyzer,
        *,
        prepend: bool = False,
    ) -> None:
        """
        Register an analyzer.

        By default the analyzer is appended.

        Use prepend=True when the new analyzer should have
        priority over existing analyzers for the same content type.
        """

        if prepend:
            self._analyzers.insert(0, analyzer)
        else:
            self._analyzers.append(analyzer)

    def analyze(self, content) -> MultimodalAnalysis:
        """
        Analyze one content element and cache the result.
        """

        content_id = getattr(
            content,
            "element_id",
            None,
        )

        if (
            content_id is not None
            and content_id in self._cache
        ):
            return self._cache[content_id]

        analyzer = self._find_analyzer(content)

        if analyzer is None:
            raise ValueError(
                "No multimodal analyzer registered for "
                f"content type: {getattr(content, 'type', None)}"
            )

        result = analyzer.analyze(content)

        if content_id is not None:
            self._cache[content_id] = result

        return result

    def analyze_document(
        self,
        document,
    ) -> list[MultimodalAnalysis]:
        """Analyze every multimodal element in a document."""

        results: list[MultimodalAnalysis] = []

        for element in document.elements:
            if element.type == "text":
                continue

            results.append(
                self.analyze(element)
            )

        return results

    def materialize(
        self,
        document,
    ) -> list[MultimodalAnalysis]:
        """
        Analyze multimodal content and write the analysis back
        into the normalized document.

        Extracted multimodal text becomes searchable by downstream
        vector and graph components.
        """

        results: list[MultimodalAnalysis] = []

        for element in document.elements:
            if element.type == "text":
                continue

            analysis = self.analyze(element)

            results.append(analysis)

            if element.payload is None:
                element.payload = {}

            analysis_payload = analysis.model_dump(
                exclude_none=True
            )
            analysis_payload.update(analysis.metadata)
            element.payload["analysis"] = analysis_payload

            if analysis.extracted_text:
                element.payload["extracted_text"] = (
                    analysis.extracted_text
                )

                if not element.text:
                    element.text = (
                        analysis.extracted_text
                    )

        return results

    def get_cached(
        self,
        content_id: str,
    ) -> MultimodalAnalysis | None:
        """Return cached analysis without triggering analysis."""

        return self._cache.get(content_id)

    def clear_cache(self) -> None:
        """Clear all cached analysis results."""

        self._cache.clear()

    def _find_analyzer(
        self,
        content,
    ) -> MultimodalAnalyzer | None:
        """Find the first registered analyzer supporting the content."""

        for analyzer in self._analyzers:
            if analyzer.supports(content):
                return analyzer

        return None