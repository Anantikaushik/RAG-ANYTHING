from __future__ import annotations

from app.knowledge_graph.models.edge import GraphEdge
from app.knowledge_graph.models.node import GraphNode
from app.knowledge_graph.storage.base import GraphStorage


class InMemoryGraphStorage(GraphStorage):
    """In-memory graph storage implementation.

    Stores graph nodes and directed edges using Python dictionaries.
    Intended for development, testing, and local execution.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[tuple[str, str, str], GraphEdge] = {}

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def edges(self) -> dict[tuple[str, str, str], GraphEdge]:
        """Return the stored edges keyed by source, target, and relation."""
        return self._edges

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        if node.node_id in self._nodes:
            raise ValueError(f"Node already exists: {node.node_id}")

        self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return a node by ID."""
        return self._nodes.get(node_id)

    def get_nodes(self) -> list[GraphNode]:
        """Return all graph nodes."""
        return list(self._nodes.values())

    def add_edge(self, edge: GraphEdge) -> None:
        """Add a directed edge between two existing nodes."""
        if edge.source_id not in self._nodes:
            raise ValueError(
                f"Source node does not exist: {edge.source_id}"
            )

        if edge.target_id not in self._nodes:
            raise ValueError(
                f"Target node does not exist: {edge.target_id}"
            )

        key = (
            edge.source_id,
            edge.target_id,
            edge.relation,
        )

        if key in self._edges:
            raise ValueError(f"Edge already exists: {key}")

        self._edges[key] = edge

    def get_edge(
        self,
        source_id: str,
        target_id: str | None = None,
        relation: str | None = None,
    ) -> GraphEdge | None:
        """Return an edge by ID or by source, target, and relation."""
        if target_id is None and relation is None:
            return next(
                (
                    edge
                    for edge in self._edges.values()
                    if edge.edge_id == source_id
                ),
                None,
            )

        if target_id is None or relation is None:
            raise TypeError(
                "get_edge() requires either edge_id or "
                "source_id, target_id, and relation"
            )

        return self._edges.get((source_id, target_id, relation))

    def get_edges(self) -> list[GraphEdge]:
        """Return all graph edges."""
        return list(self._edges.values())

    def get_neighbors(self, node_id: str) -> list[GraphNode]:
        """Return nodes directly reachable from the given node."""
        neighbors: list[GraphNode] = []

        for edge in self._edges.values():
            if edge.source_id == node_id:
                target = self._nodes.get(edge.target_id)

                if target is not None:
                    neighbors.append(target)

        return neighbors

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._edges.clear()
        self._nodes.clear()