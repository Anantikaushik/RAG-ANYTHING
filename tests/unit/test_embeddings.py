import pytest

from app.retrieval.embeddings import DeterministicEmbeddingProvider


def test_embedding_has_expected_dimensions():
    provider = DeterministicEmbeddingProvider(dimensions=32)

    vector = provider.embed("Python machine learning")

    assert len(vector) == 32


def test_embedding_is_deterministic():
    provider = DeterministicEmbeddingProvider(dimensions=32)

    first = provider.embed("Python machine learning")
    second = provider.embed("Python machine learning")

    assert first == second


def test_embed_many():
    provider = DeterministicEmbeddingProvider(dimensions=16)

    vectors = provider.embed_many(
        ["Python", "Machine Learning", "Knowledge Graph"]
    )

    assert len(vectors) == 3
    assert all(len(vector) == 16 for vector in vectors)


def test_empty_text_is_rejected():
    provider = DeterministicEmbeddingProvider()

    with pytest.raises(ValueError):
        provider.embed("")


def test_invalid_dimensions_are_rejected():
    with pytest.raises(ValueError):
        DeterministicEmbeddingProvider(dimensions=0)