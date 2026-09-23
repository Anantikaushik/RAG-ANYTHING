from __future__ import annotations

from abc import ABC, abstractmethod

from app.knowledge_graph.models.edge import GraphEdge
from app.knowledge_graph.models.node import GraphNode


class GraphStorage(ABC):
    """Storage contract for the knowledge graph."""

    @abstractmethod
    def add_node(self, node: GraphNode) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_node(self, node_id: str) -> GraphNode | None:
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_edge(self, edge_id: str) -> GraphEdge | None:
        raise NotImplementedError

    @abstractmethod
    def get_nodes(self) -> list[GraphNode]:
        raise NotImplementedError

    @abstractmethod
    def get_edges(self) -> list[GraphEdge]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError