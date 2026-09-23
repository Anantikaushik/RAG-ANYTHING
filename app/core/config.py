from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Application configuration loaded from environment variables."""

    neo4j_uri: str = os.getenv(
        "NEO4J_URI",
        "neo4j://127.0.0.1:7687",
    )

    neo4j_username: str = os.getenv(
        "NEO4J_USERNAME",
        "neo4j",
    )

    neo4j_password: str = os.getenv(
        "NEO4J_PASSWORD",
        "",
    )

    neo4j_database: str = os.getenv(
        "NEO4J_DATABASE",
        "rag-anything-db",
    )
    graph_storage_backend: str = os.getenv(
        "GRAPH_STORAGE_BACKEND",
        "memory",
    )
    graph_enable_entity_extraction: bool = os.getenv(
        "GRAPH_ENABLE_ENTITY_EXTRACTION",
        "true",
    ).lower() == "true"
    graph_enable_relation_extraction: bool = os.getenv(
        "GRAPH_ENABLE_RELATION_EXTRACTION",
        "true",
    ).lower() == "true"

settings = Settings()