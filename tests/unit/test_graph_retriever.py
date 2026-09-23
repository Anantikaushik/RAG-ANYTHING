from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.models.node import GraphNode
from app.knowledge_graph.storage.memory import InMemoryGraphStorage
from app.retrieval.graph import GraphRetriever


def create_graph() -> GraphManager:
    return GraphManager(
        storage=InMemoryGraphStorage()
    )


def test_graph_retriever_matches_node_label() -> None:
    graph = create_graph()

    graph.add_node(
        GraphNode(
            node_id="entity-1",
            node_type="entity",
            label="OpenAI",
            properties={
                "name": "OpenAI",
            },
        )
    )

    retriever = GraphRetriever(graph)

    response = retriever.retrieve(
        query="OpenAI",
        top_k=5,
    )

    assert response.count == 1
    assert response.results[0].item_id == "entity-1"
    assert response.results[0].source == "graph"


def test_graph_retriever_matches_text_content() -> None:
    graph = create_graph()

    graph.add_node(
        GraphNode(
            node_id="content-1",
            node_type="content",
            label="text",
            properties={
                "text": (
                    "Python is used for machine learning "
                    "applications."
                ),
            },
        )
    )

    retriever = GraphRetriever(graph)

    response = retriever.retrieve(
        query="Python machine learning",
        top_k=5,
    )

    assert response.count == 1
    assert response.results[0].item_id == "content-1"


def test_graph_retriever_respects_top_k() -> None:
    graph = create_graph()

    for index in range(5):
        graph.add_node(
            GraphNode(
                node_id=f"entity-{index}",
                node_type="entity",
                label=f"Python Project {index}",
                properties={
                    "name": f"Python Project {index}",
                },
            )
        )

    retriever = GraphRetriever(graph)

    response = retriever.retrieve(
        query="Python",
        top_k=2,
    )

    assert response.count == 2


def test_graph_retriever_ignores_irrelevant_nodes() -> None:
    graph = create_graph()

    graph.add_node(
        GraphNode(
            node_id="relevant",
            node_type="entity",
            label="Python",
            properties={},
        )
    )

    graph.add_node(
        GraphNode(
            node_id="irrelevant",
            node_type="entity",
            label="Java",
            properties={},
        )
    )

    retriever = GraphRetriever(graph)

    response = retriever.retrieve(
        query="Python",
        top_k=5,
    )

    assert response.count == 1
    assert response.results[0].item_id == "relevant"


def test_graph_retriever_rejects_empty_query() -> None:
    graph = create_graph()

    retriever = GraphRetriever(graph)

    try:
        retriever.retrieve("")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "empty" in str(exc).lower()


def test_graph_retriever_rejects_invalid_top_k() -> None:
    graph = create_graph()

    retriever = GraphRetriever(graph)

    try:
        retriever.retrieve(
            query="Python",
            top_k=0,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "greater than zero" in str(exc)