import pytest

from app.retrieval.hybrid import HybridRetriever
from app.retrieval.models import RetrievalResponse, RetrievalResult


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k=5):
        return RetrievalResponse(
            query=query,
            results=self.results[:top_k],
        )


def result(item_id, score, source):
    return RetrievalResult(
        item_id=item_id,
        score=score,
        source=source,
        text=item_id,
    )


def test_hybrid_combines_results():
    graph = FakeRetriever(
        [
            result("python", 1.0, "graph"),
        ]
    )

    vector = FakeRetriever(
        [
            result("machine-learning", 0.9, "vector"),
        ]
    )

    retriever = HybridRetriever(
        graph_retriever=graph,
        vector_retriever=vector,
    )

    response = retriever.retrieve(
        query="Python",
        top_k=5,
    )

    assert len(response.results) == 2
    assert {r.item_id for r in response.results} == {
        "python",
        "machine-learning",
    }


def test_hybrid_boosts_shared_result():
    graph = FakeRetriever(
        [
            result("python", 1.0, "graph"),
        ]
    )

    vector = FakeRetriever(
        [
            result("python", 0.8, "vector"),
        ]
    )

    retriever = HybridRetriever(
        graph_retriever=graph,
        vector_retriever=vector,
        graph_weight=0.5,
        vector_weight=0.5,
    )

    response = retriever.retrieve("Python")

    assert response.count == 1
    assert response.results[0].item_id == "python"
    assert response.results[0].source == "hybrid"
    assert response.results[0].score == pytest.approx(0.9)


def test_weights_are_normalized():
    graph = FakeRetriever([])
    vector = FakeRetriever([])

    retriever = HybridRetriever(
        graph_retriever=graph,
        vector_retriever=vector,
        graph_weight=2.0,
        vector_weight=1.0,
    )

    assert retriever.graph_weight == pytest.approx(2 / 3)
    assert retriever.vector_weight == pytest.approx(1 / 3)


def test_zero_total_weight_is_rejected():
    with pytest.raises(ValueError):
        HybridRetriever(
            graph_retriever=FakeRetriever([]),
            vector_retriever=FakeRetriever([]),
            graph_weight=0,
            vector_weight=0,
        )


def test_empty_query_is_rejected():
    retriever = HybridRetriever(
        graph_retriever=FakeRetriever([]),
        vector_retriever=FakeRetriever([]),
    )

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_top_k_is_respected():
    graph = FakeRetriever(
        [
            result("a", 1.0, "graph"),
            result("b", 0.9, "graph"),
        ]
    )

    vector = FakeRetriever(
        [
            result("c", 0.8, "vector"),
            result("d", 0.7, "vector"),
        ]
    )

    retriever = HybridRetriever(
        graph_retriever=graph,
        vector_retriever=vector,
    )

    response = retriever.retrieve(
        query="test",
        top_k=2,
    )

    assert response.count == 2