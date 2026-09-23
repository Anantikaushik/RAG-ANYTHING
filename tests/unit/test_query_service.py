import pytest

from app.models import *
from app.query.service import QueryService
from app.retrieval.models import RetrievalResponse, RetrievalResult


class FakeRetriever:
    def retrieve(self, query, top_k=5):
        return RetrievalResponse(
            query=query,
            results=[
                RetrievalResult(
                    item_id="content-1",
                    score=0.95,
                    source="hybrid",
                    content_type="text",
                    text="Python is a programming language.",
                )
            ][:top_k],
        )


def test_search_returns_retrieval_response():
    service = QueryService(FakeRetriever())

    response = service.search("What is Python?")

    assert isinstance(response, RetrievalResponse)
    assert response.count == 1
    assert response.results[0].item_id == "content-1"


def test_query_is_forwarded():
    service = QueryService(FakeRetriever())

    response = service.search("Python")

    assert response.query == "Python"


def test_empty_query_is_rejected():
    service = QueryService(FakeRetriever())

    with pytest.raises(ValueError):
        service.search("")


def test_invalid_query_is_rejected():
    service = QueryService(FakeRetriever())

    with pytest.raises(ValueError):
        service.search("   ")