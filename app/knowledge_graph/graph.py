from __future__ import annotations


from app.knowledge_graph.models.edge import GraphEdge
from app.knowledge_graph.models.node import GraphNode
from app.knowledge_graph.storage.base import GraphStorage
from app.knowledge_graph.storage.memory import InMemoryGraphStorage


class GraphManager:
    def __init__(self, storage: GraphStorage | None = None) -> None:
        self.storage = storage or InMemoryGraphStorage()
        self.active_document_id: str | None = None

    @property
    def graph(self):
        return self.storage

    def add_node(self, node: GraphNode) -> None:
        self.storage.add_node(node)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: dict | None = None,
    ) -> GraphEdge:
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            properties=properties or {},
        )
        self.storage.add_edge(edge)
        return edge

    def get_neighbors(self, node_id: str):
        return self.storage.get_neighbors(node_id)

    def find_by_type(self, node_type: str):
        return [
            node
            for node in self.storage.get_nodes()
            if node.node_type == node_type
        ]

    def get_document_nodes(self, document_id: str) -> list[GraphNode]:
        """Return structural and semantic nodes belonging to one document."""
        nodes = self.storage.get_nodes()
        content_ids = {
            node.node_id
            for node in nodes
            if node.node_id.startswith(f"{document_id}:")
        }
        content_ids.add(document_id)

        return [
            node
            for node in nodes
            if (
                node.node_id in content_ids
                or str(node.properties.get("source_content_id", ""))
                in content_ids
            )
        ]

    def get_document_edges(
        self,
        document_id: str,
    ) -> list[GraphEdge]:
        """Return relationships whose endpoints belong to one document."""
        node_ids = {
            node.node_id
            for node in self.get_document_nodes(document_id)
        }
        return [
            edge
            for edge in self.storage.get_edges()
            if edge.source_id in node_ids and edge.target_id in node_ids
        ]

    def clear(self) -> None:
        self.storage.clear()