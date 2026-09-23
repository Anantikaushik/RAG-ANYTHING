import pytest

from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.models.node import GraphNode


def test_add_node():
    graph = GraphManager()

    node = GraphNode(
        node_id="doc-001",
        node_type="document",
        label="sample.pdf",
    )

    graph.add_node(node)

    assert graph.graph.node_count == 1
    assert graph.graph.get_node("doc-001") == node


def test_duplicate_node_is_rejected():
    graph = GraphManager()

    node = GraphNode(
        node_id="doc-001",
        node_type="document",
    )

    graph.add_node(node)

    with pytest.raises(ValueError):
        graph.add_node(node)


def test_add_edge():
    graph = GraphManager()

    document = GraphNode(
        node_id="doc-001",
        node_type="document",
    )

    page = GraphNode(
        node_id="page-001",
        node_type="page",
    )

    graph.add_node(document)
    graph.add_node(page)

    edge = graph.add_edge(
        source_id="doc-001",
        target_id="page-001",
        relation="CONTAINS",
    )

    assert graph.graph.edge_count == 1
    assert edge.source_id == "doc-001"
    assert edge.target_id == "page-001"
    assert edge.relation == "CONTAINS"


def test_edge_requires_existing_source():
    graph = GraphManager()

    graph.add_node(
        GraphNode(
            node_id="page-001",
            node_type="page",
        )
    )

    with pytest.raises(ValueError):
        graph.add_edge(
            source_id="missing",
            target_id="page-001",
            relation="CONTAINS",
        )


def test_edge_requires_existing_target():
    graph = GraphManager()

    graph.add_node(
        GraphNode(
            node_id="doc-001",
            node_type="document",
        )
    )

    with pytest.raises(ValueError):
        graph.add_edge(
            source_id="doc-001",
            target_id="missing",
            relation="CONTAINS",
        )


def test_get_neighbors():
    graph = GraphManager()

    document = GraphNode(
        node_id="doc-001",
        node_type="document",
    )

    page = GraphNode(
        node_id="page-001",
        node_type="page",
    )

    graph.add_node(document)
    graph.add_node(page)

    graph.add_edge(
        source_id="doc-001",
        target_id="page-001",
        relation="CONTAINS",
    )

    neighbors = graph.get_neighbors("doc-001")

    assert len(neighbors) == 1
    assert neighbors[0].node_id == "page-001"


def test_find_nodes_by_type():
    graph = GraphManager()

    graph.add_node(
        GraphNode(
            node_id="doc-001",
            node_type="document",
        )
    )

    graph.add_node(
        GraphNode(
            node_id="page-001",
            node_type="page",
        )
    )

    graph.add_node(
        GraphNode(
            node_id="page-002",
            node_type="page",
        )
    )

    pages = graph.find_by_type("page")

    assert len(pages) == 2
    assert all(
        page.node_type == "page"
        for page in pages
    )