from __future__ import annotations

from app.core.config import settings
from app.knowledge_graph.storage.base import GraphStorage
from app.knowledge_graph.storage.memory import InMemoryGraphStorage
from app.knowledge_graph.storage.neo4j import Neo4jGraphStorage


def create_graph_storage() -> GraphStorage:
    """Create the graph storage backend configured for the application."""

    backend = settings.graph_storage_backend.lower().strip()

    if backend == "memory":
        return InMemoryGraphStorage()

    if backend == "neo4j":
        return Neo4jGraphStorage()

    raise ValueError(
        f"Unsupported graph storage backend: {settings.graph_storage_backend}"
    )