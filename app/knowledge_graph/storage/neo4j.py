from __future__ import annotations

import json
from contextlib import contextmanager
from collections.abc import Iterator

from neo4j import GraphDatabase

from app.core.config import settings
from app.knowledge_graph.models.edge import GraphEdge
from app.knowledge_graph.models.node import GraphNode
from app.knowledge_graph.storage.base import GraphStorage


class Neo4jGraphStorage(GraphStorage):
    """Neo4j-backed implementation of the graph storage interface."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database

        if not self.password:
            raise ValueError("Neo4j password is not configured.")

        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
        )
        self._transaction = None

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Run a group of graph operations in one Neo4j transaction."""
        if self._transaction is not None:
            yield
            return

        with self._driver.session(database=self.database) as session:
            transaction = session.begin_transaction()
            self._transaction = transaction
            try:
                yield
                transaction.commit()
            except Exception:
                transaction.rollback()
                raise
            finally:
                self._transaction = None

    @contextmanager
    def _session(self):
        if self._transaction is not None:
            yield self._transaction
            return

        with self._driver.session(database=self.database) as session:
            yield session

    def verify_connection(self) -> bool:
        """Verify that the configured Neo4j instance is reachable."""
        self._driver.verify_connectivity()
        return True

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._driver.close()

    def add_node(self, node: GraphNode) -> None:
        """Persist a graph node."""

        query = """
        MERGE (n:GraphNode {node_id: $node_id})
        SET
            n.node_type = $node_type,
            n.label = $label,
            n.properties = $properties,
            n.ingested_at = $ingested_at
        """

        with self._session() as session:
            session.run(
                query,
                node_id=node.node_id,
                node_type=node.node_type,
                label=node.label,
                properties=json.dumps(node.properties),
                ingested_at=node.properties.get("ingested_at"),
            ).consume()

    def update_node(self, node: GraphNode) -> None:
        """Persist updated properties for an existing graph node."""
        query = """
        MATCH (n:GraphNode {node_id: $node_id})
        SET
            n.node_type = $node_type,
            n.label = $label,
            n.properties = $properties
        """

        with self._session() as session:
            summary = session.run(
                query,
                node_id=node.node_id,
                node_type=node.node_type,
                label=node.label,
                properties=json.dumps(node.properties),
            ).consume()

            if summary.counters.properties_set == 0:
                raise ValueError(
                    f"Graph node does not exist: {node.node_id}"
                )

    def get_node(self, node_id: str) -> GraphNode | None:
        """Retrieve a graph node by ID."""

        query = """
        MATCH (n:GraphNode {node_id: $node_id})
        RETURN n
        """

        with self._session() as session:
            record = session.run(
                query,
                node_id=node_id,
            ).single()

        if record is None:
            return None

        node = record["n"]

        return GraphNode(
            node_id=node["node_id"],
            node_type=node["node_type"],
            label=node.get("label"),
            properties=json.loads(
                node.get("properties") or "{}"
            ),
        )

    def add_edge(self, edge: GraphEdge) -> None:
        """Persist a graph edge between two existing nodes."""

        query = """
        MATCH (source:GraphNode {node_id: $source_id})
        MATCH (target:GraphNode {node_id: $target_id})
        MERGE (source)-[r:GRAPH_RELATION {edge_id: $edge_id}]->(target)
        SET
            r.relation = $relation,
            r.properties = $properties
        """

        with self._session() as session:
            result = session.run(
                query,
                edge_id=edge.edge_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                relation=edge.relation,
                properties=json.dumps(edge.properties),
            )

            summary = result.consume()

            if summary.counters.relationships_created == 0:
                raise ValueError(
                    f"Edge already exists or source/target nodes "
                    f"do not exist: {edge.edge_id}"
                )

    def get_edge(
        self,
        source_id: str,
        target_id: str | None = None,
        relation: str | None = None,
    ) -> GraphEdge | None:
        """Retrieve an edge by ID or by source, target, and relation."""
        if target_id is None and relation is None:
            query = """
            MATCH (
                source:GraphNode
            )-[r:GRAPH_RELATION {edge_id: $edge_id}]->(
                target:GraphNode
            )
            RETURN
                source.node_id AS source_id,
                target.node_id AS target_id,
                r.edge_id AS edge_id,
                r.relation AS relation,
                r.properties AS properties
            """
            parameters = {"edge_id": source_id}
        elif target_id is not None and relation is not None:
            query = """
            MATCH (
                source:GraphNode {node_id: $source_id}
            )-[r:GRAPH_RELATION {relation: $relation}]->(
                target:GraphNode {node_id: $target_id}
            )
            RETURN
                source.node_id AS source_id,
                target.node_id AS target_id,
                r.edge_id AS edge_id,
                r.relation AS relation,
                r.properties AS properties
            LIMIT 1
            """
            parameters = {
                "source_id": source_id,
                "target_id": target_id,
                "relation": relation,
            }
        else:
            raise TypeError(
                "get_edge() requires either edge_id or "
                "source_id, target_id, and relation"
            )

        with self._session() as session:
            record = session.run(
                query,
                **parameters,
            ).single()

        if record is None:
            return None

        return GraphEdge(
            edge_id=record["edge_id"],
            source_id=record["source_id"],
            target_id=record["target_id"],
            relation=record["relation"],
            properties=json.loads(
                record["properties"] or "{}"
            ),
        )

    def get_nodes(self) -> list[GraphNode]:
        """Retrieve all graph nodes."""

        query = """
        MATCH (n:GraphNode)
        RETURN n
        ORDER BY n.node_id
        """

        with self._session() as session:
            records = session.run(query)

            return [
                GraphNode(
                    node_id=record["n"]["node_id"],
                    node_type=record["n"]["node_type"],
                    label=record["n"].get("label"),
                    properties=json.loads(
                        record["n"].get("properties") or "{}"
                    ),
                )
                for record in records
            ]

    def get_edges(self) -> list[GraphEdge]:
        """Retrieve all graph edges."""

        query = """
        MATCH (
            source:GraphNode
        )-[r:GRAPH_RELATION]->(
            target:GraphNode
        )
        RETURN
            source.node_id AS source_id,
            target.node_id AS target_id,
            r.edge_id AS edge_id,
            r.relation AS relation,
            r.properties AS properties
        ORDER BY r.edge_id
        """

        with self._session() as session:
            records = session.run(query)

            return [
                GraphEdge(
                    edge_id=record["edge_id"],
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    relation=record["relation"],
                    properties=json.loads(
                        record["properties"] or "{}"
                    ),
                )
                for record in records
            ]

    def clear(self) -> None:
        """Delete all nodes and relationships managed by this storage."""

        query = """
        MATCH (n:GraphNode)
        DETACH DELETE n
        """

        with self._session() as session:
            session.run(query).consume()