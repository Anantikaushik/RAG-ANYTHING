import pytest

from app.retrieval.embeddings import DeterministicEmbeddingProvider
from app.retrieval.vector import VectorRetriever
from app.retrieval.vector_store import InMemoryVectorStore


def create_retriever():
    provider = DeterministicEmbeddingProvider(dimensions=32)
    store = InMemoryVectorStore()

    texts = [
        ("python", "Python programming language"),
        ("machine-learning", "Machine learning applications"),
        ("neo4j", "Neo4j knowledge graph database"),
    ]

    for item_id, text in texts:
        store.add(
            item_id=item_id,
            vector=provider.embed(text),
            text=text,
            metadata={"content_type": "text"},
        )

    return VectorRetriever(
        vector_store=store,
        embedding_provider=provider,
    )


def test_vector_retriever_returns_results():
    retriever = create_retriever()

    response = retriever.retrieve(
        query="Python programming",
        top_k=2,
    )

    assert response.query == "Python programming"
    assert len(response.results) == 2


def test_results_have_vector_source():
    retriever = create_retriever()

    response = retriever.retrieve("Python")

    assert all(
        result.source == "vector"
        for result in response.results
    )


def test_top_k_is_respected():
    retriever = create_retriever()

    response = retriever.retrieve(
        query="knowledge graph",
        top_k=1,
    )

    assert response.count == 1


def test_empty_query_is_rejected():
    retriever = create_retriever()

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_invalid_top_k_is_rejected():
    retriever = create_retriever()

    with pytest.raises(ValueError):
        retriever.retrieve("Python", top_k=0)