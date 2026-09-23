import pytest

from app.query.answer import AnswerResponse, AnswerService
from app.query.providers import DeterministicGenerationProvider
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
                    metadata={"page_idx": 1},
                )
            ][:top_k],
        )


def create_service():
    return AnswerService(
        retriever=FakeRetriever(),
        generation_provider=DeterministicGenerationProvider(),
    )


def test_answer_returns_response():
    service = create_service()

    response = service.answer("What is Python?")

    assert isinstance(response, AnswerResponse)
    assert response.query == "What is Python?"
    assert "Python" in response.answer


def test_answer_contains_sources():
    service = create_service()

    response = service.answer("What is Python?")

    assert len(response.sources) == 1

    citation = response.sources[0]
    assert citation.item_id == "content-1"
    assert citation.page_idx == 1
    assert citation.content_type == "text"


def test_answer_contains_context():
    service = create_service()

    response = service.answer("What is Python?")

    assert response.context.count == 1
    assert response.context.items[0].item_id == "content-1"


def test_empty_query_is_rejected():
    service = create_service()

    with pytest.raises(ValueError):
        service.answer("")


def test_top_k_is_forwarded():
    service = create_service()

    response = service.answer(
        "What is Python?",
        top_k=1,
    )

    assert response.context.count == 1