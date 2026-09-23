from __future__ import annotations

from app.query.answer import AnswerResponse, AnswerService
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.models import RetrievalResponse


class QueryService:
    """Application-level query interface."""

    def __init__(
        self,
        retriever: HybridRetriever,
        answer_service: AnswerService | None = None,
    ) -> None:
        self.retriever = retriever
        self.answer_service = answer_service

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResponse:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

    def ask(
        self,
        query: str,
        top_k: int = 5,
    ) -> AnswerResponse:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if self.answer_service is None:
            raise RuntimeError(
                "AnswerService is required for ask()."
            )

        return self.answer_service.answer(
            query=query,
            top_k=top_k,
        )