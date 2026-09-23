from __future__ import annotations

from app.ingestion.content import ContentItem
from app.ingestion.content_list import ContentList
from app.ingestion.pipeline import IngestionPipeline
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.models.node import GraphNode
from app.knowledge_graph.storage.memory import InMemoryGraphStorage


def test_ingestion_pipeline_builds_complete_graph() -> None:
    """
    Verify the complete ingestion pipeline:

    parse
        -> normalize
        -> hierarchy
        -> multimodal processing
        -> structural graph
        -> semantic graph
        -> multimodal graph enrichment
    """

    content = ContentList(
        [
            ContentItem(
                type="text",
                text=(
                "OpenAI develops artificial intelligence systems. "
                "Python is used for building machine learning applications."
                ),
                page_idx=0,
            )
        ]
    )

    graph_manager = GraphManager(
        storage=InMemoryGraphStorage()
    )

    from app.knowledge_graph.builder import GraphBuilder

    graph_builder = GraphBuilder(
        graph_manager=graph_manager
    )

    pipeline = IngestionPipeline(
        graph_builder=graph_builder
    )

    result = pipeline.ingest_content_list(
        content,
        filename="sample.pdf",
    )

    # ---------------------------------------------------------
    # Document
    # ---------------------------------------------------------

    assert result.document.filename == "sample.pdf"
    assert result.document.content_count == 1

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------

    assert result.normalized_document.element_count == 1

    element = result.normalized_document.elements[0]

    assert element.type == "text"
    assert element.text is not None
    assert "OpenAI" in element.text

    # ---------------------------------------------------------
    # Hierarchy
    # ---------------------------------------------------------

    assert len(result.hierarchy.page_nodes) == 1

    # ---------------------------------------------------------
    # Graph
    # ---------------------------------------------------------

    graph = result.graph

    document_node = graph.graph.get_node(
        result.document.document_id
    )

    assert document_node is not None
    assert document_node.node_type == "document"

    content_node = graph.graph.get_node(
        element.element_id
    )

    assert content_node is not None
    assert content_node.node_type == "content"

    # ---------------------------------------------------------
    # Semantic enrichment
    # ---------------------------------------------------------

    entity_nodes = graph.find_by_type("entity")

    assert len(entity_nodes) >= 1

    entity_names = {
        node.properties.get("name")
        for node in entity_nodes
    }

    assert "OpenAI" in entity_names

    # ---------------------------------------------------------
    # Structural relationships
    # ---------------------------------------------------------

    contains_edges = [
        edge
        for edge in graph.graph.get_edges()
        if edge.relation == "CONTAINS"
    ]

    assert len(contains_edges) >= 2