from app.ingestion.normalization.models import (
    NormalizedContent,
    NormalizedDocument,
)
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.semantic_processor import (
    SemanticGraphProcessor,
)


def test_semantic_processor_adds_entities_and_relations():
    graph = GraphManager()

    content = NormalizedContent(
        element_id="content-001",
        document_id="doc-001",
        type="text",
        page_idx=0,
        position=0,
        text="Microsoft develops Azure",
    )

    document = NormalizedDocument(
        document_id="doc-001",
        filename="test.txt",
        elements=[content],
    )

    processor = SemanticGraphProcessor(graph)

    # The content node must already exist because the semantic
    # processor enriches an existing structural graph.
    from app.knowledge_graph.models.node import GraphNode

    graph.add_node(
        GraphNode(
            node_id="content-001",
            node_type="content",
            label="text",
        )
    )

    processor.process(document)

    entity_nodes = graph.find_by_type("entity")

    assert len(entity_nodes) >= 2

    microsoft = next(
        node
        for node in entity_nodes
        if node.label == "Microsoft"
    )

    assert microsoft.node_type == "entity"

    mentions_edge = graph.graph.get_edge(
        "content-001",
        microsoft.node_id,
        "MENTIONS",
    )

    assert mentions_edge is not None


def test_semantic_processor_ignores_empty_content():
    graph = GraphManager()

    content = NormalizedContent(
        element_id="content-001",
        document_id="doc-001",
        type="text",
        page_idx=0,
        position=0,
        text="   ",
    )

    document = NormalizedDocument(
        document_id="doc-001",
        filename="test.txt",
        elements=[content],
    )

    processor = SemanticGraphProcessor(graph)

    processor.process(document)

    assert graph.find_by_type("entity") == []