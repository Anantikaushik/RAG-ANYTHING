from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.models.node import GraphNode
from app.knowledge_graph.storage.memory import InMemoryGraphStorage
from app.retrieval.traversal import GraphTraversalRetriever


def create_graph() -> GraphManager:
    return GraphManager(
        storage=InMemoryGraphStorage()
    )


def test_traversal_retrieves_connected_nodes() -> None:
    graph = create_graph()

    graph.add_node(
        GraphNode(
            node_id="entity-1",
            node_type="entity",
            label="Python",
            properties={
                "name": "Python",
            },
        )
    )

    graph.add_node(
        GraphNode(
            node_id="content-1",
            node_type="content",
            label="text",
            properties={
                "text": "Python is used for machine learning.",
            },
        )
    )

    graph.add_edge(
        source_id="content-1",
        target_id="entity-1",
        relation="MENTIONS",
    )

    retriever = GraphTraversalRetriever(graph)

    response = retriever.retrieve(
        query="Python",
        top_k=5,
        max_depth=1,
    )

    ids = {
        result.item_id
        for result in response.results
    }

    assert "entity-1" in ids


def test_traversal_respects_depth() -> None:
    graph = create_graph()

    graph.add_node(
        GraphNode(
            node_id="entity-1",
            node_type="entity",
            label="Python",
            properties={
                "name": "Python",
            },
        )
    )

    graph.add_node(
        GraphNode(
            node_id="content-1",
            node_type="content",
            label="text",
            properties={
                "text": "Python content",
            },
        )
    )

    graph.add_node(
        GraphNode(
            node_id="document-1",
            node_type="document",
            label="Python Document",
            properties={},
        )
    )

    graph.add_edge(
        source_id="entity-1",
        target_id="content-1",
        relation="MENTIONS",
    )

    graph.add_edge(
        source_id="content-1",
        target_id="document-1",
        relation="CONTAINS",
    )

    retriever = GraphTraversalRetriever(graph)

    response = retriever.retrieve(
        query="Python",
        top_k=10,
        max_depth=1,
    )

    ids = {
        result.item_id
        for result in response.results
    }

    assert "entity-1" in ids
    assert "content-1" not in ids
    assert "document-1" not in ids


def test_traversal_rejects_invalid_depth() -> None:
    graph = create_graph()

    retriever = GraphTraversalRetriever(graph)

    try:
        retriever.retrieve(
            query="Python",
            max_depth=-1,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "negative" in str(exc).lower()