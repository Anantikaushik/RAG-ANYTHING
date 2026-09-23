from app.knowledge_graph.models.edge import GraphEdge
from app.knowledge_graph.models.node import GraphNode
from app.knowledge_graph.storage.memory import InMemoryGraphStorage


def make_node(node_id: str, node_type: str = "content") -> GraphNode:
    return GraphNode(
        node_id=node_id,
        node_type=node_type,
    )


def test_store_and_get_node():
    storage = InMemoryGraphStorage()

    node = make_node("node-1")

    storage.add_node(node)

    assert storage.get_node("node-1") == node


def test_missing_node_returns_none():
    storage = InMemoryGraphStorage()

    assert storage.get_node("missing") is None


def test_duplicate_node_rejected():
    storage = InMemoryGraphStorage()

    node = make_node("node-1")

    storage.add_node(node)

    try:
        storage.add_node(node)
        assert False
    except ValueError:
        assert True


def test_store_and_get_edge():
    storage = InMemoryGraphStorage()

    source = make_node("source")
    target = make_node("target")

    storage.add_node(source)
    storage.add_node(target)

    edge = GraphEdge(
        edge_id="edge-1",
        source_id="source",
        target_id="target",
        relation="RELATED_TO",
    )

    storage.add_edge(edge)

    assert storage.get_edge("edge-1") == edge


def test_edge_requires_source():
    storage = InMemoryGraphStorage()

    storage.add_node(make_node("target"))

    edge = GraphEdge(
        edge_id="edge-1",
        source_id="missing",
        target_id="target",
        relation="RELATED_TO",
    )

    try:
        storage.add_edge(edge)
        assert False
    except ValueError:
        assert True


def test_edge_requires_target():
    storage = InMemoryGraphStorage()

    storage.add_node(make_node("source"))

    edge = GraphEdge(
        edge_id="edge-1",
        source_id="source",
        target_id="missing",
        relation="RELATED_TO",
    )

    try:
        storage.add_edge(edge)
        assert False
    except ValueError:
        assert True


def test_get_nodes():
    storage = InMemoryGraphStorage()

    storage.add_node(make_node("node-1"))
    storage.add_node(make_node("node-2"))

    nodes = storage.get_nodes()

    assert len(nodes) == 2


def test_get_edges():
    storage = InMemoryGraphStorage()

    storage.add_node(make_node("source"))
    storage.add_node(make_node("target"))

    storage.add_edge(
        GraphEdge(
            edge_id="edge-1",
            source_id="source",
            target_id="target",
            relation="RELATED_TO",
        )
    )

    assert len(storage.get_edges()) == 1


def test_clear():
    storage = InMemoryGraphStorage()

    storage.add_node(make_node("node-1"))

    storage.clear()

    assert storage.get_nodes() == []
    assert storage.get_edges() == []