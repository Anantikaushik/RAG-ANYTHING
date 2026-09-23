from __future__ import annotations

from app.ingestion.normalization.models import NormalizedDocument
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.models.node import GraphNode
from app.multimodal.analysis.models import MultimodalAnalysis
from app.multimodal.analysis.processor import (
    MultimodalAnalysisProcessor,
)


class MultimodalGraphProcessor:
    """Add multimodal analysis results to the knowledge graph."""

    def __init__(
        self,
        graph: GraphManager,
        analysis_processor: MultimodalAnalysisProcessor | None = None,
    ) -> None:
        self.graph = graph
        self.analysis_processor = (
            analysis_processor
            or MultimodalAnalysisProcessor()
        )

    def process(
        self,
        document: NormalizedDocument,
    ) -> GraphManager:
        for element in document.elements:
            if element.type == "text":
                continue

            # Reuse an existing analysis if materialize() or another
            # downstream component has already analyzed this element.
            analysis = self.analysis_processor.get_cached(
                element.element_id
            )

            if analysis is None:
                analysis = self.analysis_processor.analyze(
                    element
                )

            self._add_analysis(analysis)

        return self.graph

    def _add_analysis(
        self,
        analysis: MultimodalAnalysis,
    ) -> GraphNode:
        node = self.graph.graph.get_node(
            analysis.content_id
        )

        if node is None:
            raise ValueError(
                "Cannot attach multimodal analysis: "
                f"content node does not exist: "
                f"{analysis.content_id}"
            )

        properties: dict = {
            "content_type": analysis.content_type,
        }

        if analysis.summary is not None:
            properties["summary"] = analysis.summary

        if analysis.extracted_text is not None:
            properties["extracted_text"] = (
                analysis.extracted_text
            )

        if analysis.entities:
            properties["entities"] = analysis.entities

        if analysis.relationships:
            properties["relationships"] = (
                analysis.relationships
            )

        properties.update(analysis.metadata)

        node.properties.update(properties)
        persist_node = getattr(self.graph.graph, "update_node", None)
        if persist_node is not None:
            persist_node(node)

        return node