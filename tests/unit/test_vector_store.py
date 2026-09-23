import pytest

from app.retrieval.vector_store import (
    InMemoryVectorStore,
    VectorRecord,
)


def test_add_and_get():
    store = InMemoryVectorStore()

    store.add(
        item_id="content-1",
        vector=[1.0, 0.0],
        text="Python",
        metadata={"page": 1},
    )

    record = store.get("content-1")

    assert record is not None
    assert record.text == "Python"
    assert record.metadata["page"] == 1


def test_search_returns_most_similar():
    store = InMemoryVectorStore()

    store.add(
        item_id="python",
        vector=[1.0, 0.0],
        text="Python",
    )

    store.add(
        item_id="java",
        vector=[0.0, 1.0],
        text="Java",
    )

    results = store.search(
        query_vector=[1.0, 0.0],
        top_k=2,
    )

    assert results[0][0].item_id == "python"
    assert results[0][1] > results[1][1]


def test_top_k_is_respected():
    store = InMemoryVectorStore()

    for index in range(5):
        store.add(
            item_id=f"item-{index}",
            vector=[1.0, 0.0],
            text=f"Item {index}",
        )

    results = store.search(
        query_vector=[1.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2


def test_duplicate_id_updates_record():
    store = InMemoryVectorStore()

    store.add(
        item_id="item-1",
        vector=[1.0, 0.0],
        text="Old",
    )

    store.add(
        item_id="item-1",
        vector=[0.0, 1.0],
        text="New",
    )

    assert len(store) == 1
    assert store.get("item-1").text == "New"


def test_dimension_mismatch_is_rejected():
    store = InMemoryVectorStore()

    store.add(
        item_id="item-1",
        vector=[1.0, 0.0],
        text="Test",
    )

    with pytest.raises(ValueError):
        store.search([1.0, 0.0, 0.0])


def test_invalid_top_k_is_rejected():
    store = InMemoryVectorStore()

    with pytest.raises(ValueError):
        store.search([1.0, 0.0], top_k=0)


def test_clear():
    store = InMemoryVectorStore()

    store.add(
        item_id="item-1",
        vector=[1.0, 0.0],
        text="Test",
    )

    store.clear()

    assert len(store) == 0