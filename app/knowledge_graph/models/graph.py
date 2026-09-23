from __future__ import annotations

from pydantic import BaseModel, Field

from app.knowledge_graph.models.edge import GraphEdge
from app.knowledge_graph.models.node import GraphNode


class KnowledgeGraph(BaseModel):
    """Storage-independent knowledge graph representation."""

    nodes: dict[str, GraphNode] = Field(
        default_factory=dict
    )

    edges: dict[str, GraphEdge] = Field(
        default_factory=dict
    )

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def get_node(
        self,
        node_id: str,
    ) -> GraphNode | None:
        return self.nodes.get(node_id)

    def get_edge(
        self,
        edge_id: str,
    ) -> GraphEdge | None:
        return self.edges.get(edge_id)