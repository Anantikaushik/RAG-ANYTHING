from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.storage.memory import InMemoryGraphStorage
from app.knowledge_graph.models.node import GraphNode


def test_graph_manager_accepts_injected_storage():
    storage = InMemoryGraphStorage()

    manager = GraphManager(storage=storage)

    assert manager.storage is storage
    assert manager.graph is storage


def test_graph_manager_scopes_nodes_and_edges_to_document() -> None:
    manager = GraphManager(storage=InMemoryGraphStorage())
    manager.add_node(
        GraphNode(
            node_id="doc-a",
            node_type="document",
            properties={},
        )
    )
    manager.add_node(
        GraphNode(
            node_id="doc-a:page:0",
            node_type="page",
            properties={},
        )
    )
    manager.add_node(
        GraphNode(
            node_id="entity-a",
            node_type="entity",
            properties={"source_content_id": "doc-a:content:0"},
        )
    )
    manager.add_node(
        GraphNode(
            node_id="doc-b",
            node_type="document",
            properties={},
        )
    )
    manager.add_edge(
        source_id="doc-a",
        target_id="doc-a:page:0",
        relation="CONTAINS",
    )

    assert {
        node.node_id for node in manager.get_document_nodes("doc-a")
    } == {"doc-a", "doc-a:page:0"}
    assert manager.get_document_edges("doc-a") == [
        manager.graph.get_edges()[0]
    ]