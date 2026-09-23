from app.ingestion.normalization.models import (
    NormalizedContent,
    NormalizedDocument,
)
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.models.node import GraphNode
from app.knowledge_graph.multimodal_processor import (
    MultimodalGraphProcessor,
)


def test_multimodal_analysis_updates_content_node():
    graph = GraphManager()

    graph.add_node(
        GraphNode(
            node_id="image-001",
            node_type="content",
            label="image",
        )
    )

    document = NormalizedDocument(
        document_id="doc-001",
        filename="test.pdf",
        elements=[
            NormalizedContent(
                element_id="image-001",
                document_id="doc-001",
                type="image",
                page_idx=0,
                position=0,
                payload={
                    "img_path": "diagram.png",
                },
            )
        ],
    )

    processor = MultimodalGraphProcessor(graph)

    processor.process(document)

    node = graph.graph.get_node("image-001")

    assert node is not None
    assert node.properties["content_type"] == "image"
    assert node.properties["summary"] == "Image content"
    assert node.properties["img_path"] == "diagram.png"


def test_multimodal_processor_handles_table():
    graph = GraphManager()

    graph.add_node(
        GraphNode(
            node_id="table-001",
            node_type="content",
            label="table",
        )
    )

    document = NormalizedDocument(
        document_id="doc-001",
        filename="test.pdf",
        elements=[
            NormalizedContent(
                element_id="table-001",
                document_id="doc-001",
                type="table",
                page_idx=0,
                position=0,
                payload={
                    "table_body": "| A | B |",
                },
            )
        ],
    )

    processor = MultimodalGraphProcessor(graph)

    processor.process(document)

    node = graph.graph.get_node("table-001")

    assert node is not None
    assert node.properties["content_type"] == "table"
    assert node.properties["extracted_text"] == "| A | B |"


def test_missing_content_node_fails_fast():
    graph = GraphManager()

    document = NormalizedDocument(
        document_id="doc-001",
        filename="test.pdf",
        elements=[
            NormalizedContent(
                element_id="image-001",
                document_id="doc-001",
                type="image",
                page_idx=0,
                position=0,
                payload={
                    "img_path": "diagram.png",
                },
            )
        ],
    )

    processor = MultimodalGraphProcessor(graph)

    try:
        processor.process(document)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "content node does not exist" in str(exc)