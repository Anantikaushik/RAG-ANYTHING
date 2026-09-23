from __future__ import annotations

from app.ingestion.normalization.models import (
    NormalizedContent,
    NormalizedDocument,
)
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.semantic_processor import SemanticGraphProcessor
from app.knowledge_graph.storage.memory import InMemoryGraphStorage


def create_graph() -> GraphManager:
    return GraphManager(
        storage=InMemoryGraphStorage()
    )


def test_multimodal_extracted_text_creates_semantic_entities() -> None:
    graph = create_graph()

    content = NormalizedContent(
        element_id="image-1",
        document_id="doc-1",
        type="image",
        page_idx=0,
        position=0,
        payload={
            "img_path": "diagram.png",
        },
    )

    # The graph processor expects the content node to already exist.
    from app.knowledge_graph.models.node import GraphNode

    graph.add_node(
        GraphNode(
            node_id="image-1",
            node_type="content",
            label="image",
            properties={
                "type": "image",
            },
        )
    )

    processor = SemanticGraphProcessor(
        graph=graph,
    )

    # Replace the default multimodal analyzer with a deterministic
    # analyzer that exposes semantic text.
    from app.multimodal.analysis.base import MultimodalAnalyzer
    from app.multimodal.analysis.models import MultimodalAnalysis

    class TestImageAnalyzer(MultimodalAnalyzer):

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
                extracted_text="OpenAI develops Python systems.",
            )

    from app.multimodal.analysis.processor import (
        MultimodalAnalysisProcessor,
    )

    processor.multimodal_processor = MultimodalAnalysisProcessor(
        analyzers=[
            TestImageAnalyzer(),
        ]
    )

    document = NormalizedDocument(
        document_id="doc-1",
        filename="test.pdf",
        elements=[content],
    )

    processor.process(document)

    entity_nodes = graph.find_by_type("entity")

    assert len(entity_nodes) >= 1

    entity_names = {
        node.properties.get("name")
        for node in entity_nodes
    }

    assert "OpenAI" in entity_names


def test_multimodal_without_extracted_text_does_not_create_entities() -> None:
    graph = create_graph()

    content = NormalizedContent(
        element_id="table-1",
        document_id="doc-1",
        type="table",
        page_idx=0,
        position=0,
        payload={
            "table_body": "| A | B |\n|---|---|\n| 1 | 2 |",
        },
    )

    from app.knowledge_graph.models.node import GraphNode

    graph.add_node(
        GraphNode(
            node_id="table-1",
            node_type="content",
            label="table",
            properties={
                "type": "table",
            },
        )
    )

    processor = SemanticGraphProcessor(
        graph=graph,
    )

    document = NormalizedDocument(
        document_id="doc-1",
        filename="test.pdf",
        elements=[content],
    )

    processor.process(document)

    assert graph.find_by_type("entity") == []


def test_multimodal_processing_preserves_original_content_text() -> None:
    graph = create_graph()

    original_text = None

    content = NormalizedContent(
        element_id="equation-1",
        document_id="doc-1",
        type="equation",
        page_idx=0,
        position=0,
        text=original_text,
        payload={
            "latex": r"E = mc^2",
        },
    )

    graph.add_node(
        __import__(
            "app.knowledge_graph.models.node",
            fromlist=["GraphNode"],
        ).GraphNode(
            node_id="equation-1",
            node_type="content",
            label="equation",
            properties={
                "type": "equation",
            },
        )
    )

    processor = SemanticGraphProcessor(
        graph=graph,
    )

    document = NormalizedDocument(
        document_id="doc-1",
        filename="test.pdf",
        elements=[content],
    )

    processor.process(document)

    assert content.text == original_text