from __future__ import annotations

from uuid import uuid4

import pytest

from app.core.config import settings
from app.knowledge_graph.models.edge import GraphEdge
from app.knowledge_graph.models.node import GraphNode
from app.knowledge_graph.storage.neo4j import Neo4jGraphStorage


@pytest.fixture
def storage():
    if not settings.neo4j_password:
        pytest.skip("Neo4j password is not configured.")

    storage = Neo4jGraphStorage()

    try:
        storage.verify_connection()
    except Exception as exc:
        storage.close()
        pytest.skip(f"Neo4j is unavailable: {exc}")

    storage.clear()

    yield storage

    storage.clear()
    storage.close()


def test_connection(storage):
    assert storage.verify_connection() is True


def test_add_and_get_node(storage):
    node = GraphNode(
        node_id=f"test-node-{uuid4()}",
        node_type="document",
        label="Test Document",
        properties={"source": "integration-test"},
    )

    storage.add_node(node)

    result = storage.get_node(node.node_id)

    assert result is not None
    assert result.node_id == node.node_id
    assert result.node_type == node.node_type
    assert result.label == node.label
    assert result.properties == node.properties


def test_get_missing_node_returns_none(storage):
    assert storage.get_node("does-not-exist") is None


def test_add_and_get_edge(storage):
    source = GraphNode(
        node_id=f"source-{uuid4()}",
        node_type="document",
        label="Source",
    )

    target = GraphNode(
        node_id=f"target-{uuid4()}",
        node_type="page",
        label="Page 1",
    )

    storage.add_node(source)
    storage.add_node(target)

    edge = GraphEdge(
        edge_id=f"edge-{uuid4()}",
        source_id=source.node_id,
        target_id=target.node_id,
        relation="CONTAINS",
        properties={"test": True},
    )

    storage.add_edge(edge)

    result = storage.get_edge(edge.edge_id)

    assert result is not None
    assert result.edge_id == edge.edge_id
    assert result.source_id == source.node_id
    assert result.target_id == target.node_id
    assert result.relation == "CONTAINS"
    assert result.properties == {"test": True}


def test_get_missing_edge_returns_none(storage):
    assert storage.get_edge("does-not-exist") is None


def test_get_nodes(storage):
    nodes = [
        GraphNode(
            node_id=f"node-{uuid4()}",
            node_type="content",
            label="Content",
        )
        for _ in range(3)
    ]

    for node in nodes:
        storage.add_node(node)

    stored_nodes = storage.get_nodes()

    stored_ids = {node.node_id for node in stored_nodes}

    assert {node.node_id for node in nodes}.issubset(stored_ids)


def test_get_edges(storage):
    source = GraphNode(
        node_id=f"source-{uuid4()}",
        node_type="document",
    )

    target = GraphNode(
        node_id=f"target-{uuid4()}",
        node_type="content",
    )

    storage.add_node(source)
    storage.add_node(target)

    edge = GraphEdge(
        edge_id=f"edge-{uuid4()}",
        source_id=source.node_id,
        target_id=target.node_id,
        relation="CONTAINS",
    )

    storage.add_edge(edge)

    edges = storage.get_edges()

    assert any(item.edge_id == edge.edge_id for item in edges)


def test_clear(storage):
    node = GraphNode(
        node_id=f"node-{uuid4()}",
        node_type="document",
        label="Temporary",
    )

    storage.add_node(node)

    assert storage.get_node(node.node_id) is not None

    storage.clear()

    assert storage.get_node(node.node_id) is None