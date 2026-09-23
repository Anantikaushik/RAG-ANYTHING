from app.retrieval.models import (
    RetrievalResponse,
    RetrievalResult,
)


def test_retrieval_result_defaults() -> None:
    result = RetrievalResult(
        item_id="content-1",
        source="graph",
    )

    assert result.item_id == "content-1"
    assert result.score == 0.0
    assert result.source == "graph"
    assert result.metadata == {}


def test_retrieval_result_supports_metadata() -> None:
    result = RetrievalResult(
        item_id="image-1",
        score=0.87,
        source="multimodal",
        content_type="image",
        text="Architecture diagram",
        metadata={
            "page_idx": 3,
            "document_id": "doc-1",
        },
    )

    assert result.score == 0.87
    assert result.content_type == "image"
    assert result.metadata["page_idx"] == 3


def test_retrieval_response_count() -> None:
    response = RetrievalResponse(
        query="What is the architecture?",
        results=[
            RetrievalResult(
                item_id="content-1",
                source="graph",
                score=0.9,
            ),
            RetrievalResult(
                item_id="content-2",
                source="vector",
                score=0.8,
            ),
        ],
    )

    assert response.count == 2