from __future__ import annotations

from pathlib import Path

from app.ingestion.pipeline import IngestionPipeline, IngestionResult
from app.query.answer import AnswerResponse, AnswerService
from app.query.providers.base import GenerationProvider
from app.query.service import QueryService
from app.retrieval.graph import GraphRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import VectorRetriever


class RAGAnythingApp:
    """
    Top-level application facade.

    Coordinates:
        document ingestion
        graph retrieval
        vector retrieval
        hybrid retrieval
        context assembly
        answer generation
    """

    def __init__(
        self,
        ingestion_pipeline: IngestionPipeline,
        generation_provider: GenerationProvider,
        graph_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> None:
        self.ingestion_pipeline = ingestion_pipeline
        self.generation_provider = generation_provider
        self.graph_weight = graph_weight
        self.vector_weight = vector_weight

        self._ingestion_result: IngestionResult | None = None
        self._ingestion_results: list[IngestionResult] = []
        self.query_service: QueryService | None = None

    @property
    def is_ready(self) -> bool:
        """Return True when at least one document has been ingested."""
        return bool(self._ingestion_results)

    @property
    def ingestion_results(self) -> list[IngestionResult]:
        """All documents in the shared workspace, in ingestion order."""
        return list(self._ingestion_results)

    @property
    def ingestion_result(self) -> IngestionResult:
        """Return the latest ingestion result."""
        if self._ingestion_result is None:
            raise RuntimeError(
                "No document has been ingested yet."
            )

        return self._ingestion_result

    def ingest(
        self,
        path: str | Path,
    ) -> IngestionResult:
        """Ingest a document and prepare the query pipeline."""

        result = self.ingestion_pipeline.ingest(path)

        graph_retriever = GraphRetriever(
            result.graph
        )

        vector_retriever = VectorRetriever(
            vector_store=result.vector_store,
            embedding_provider=self.ingestion_pipeline.embedding_provider,
        )

        hybrid_retriever = HybridRetriever(
            graph_retriever=graph_retriever,
            vector_retriever=vector_retriever,
            graph_weight=self.graph_weight,
            vector_weight=self.vector_weight,
        )

        answer_service = AnswerService(
            retriever=hybrid_retriever,
            generation_provider=self.generation_provider,
        )

        self.query_service = QueryService(
            retriever=hybrid_retriever,
            answer_service=answer_service,
        )

        self._ingestion_result = result
        self._ingestion_results.append(result)

        return result

    def ingest_many(self, paths: list[str | Path]) -> list[IngestionResult]:
        """Ingest files into one shared graph/vector retrieval workspace."""
        results = self.ingestion_pipeline.ingest_many(paths)
        if not results:
            raise ValueError("At least one document path is required.")

        graph_retriever = GraphRetriever(results[-1].graph)
        vector_retriever = VectorRetriever(
            vector_store=results[-1].vector_store,
            embedding_provider=self.ingestion_pipeline.embedding_provider,
        )
        hybrid_retriever = HybridRetriever(
            graph_retriever=graph_retriever,
            vector_retriever=vector_retriever,
            graph_weight=self.graph_weight,
            vector_weight=self.vector_weight,
        )
        answer_service = AnswerService(
            retriever=hybrid_retriever,
            generation_provider=self.generation_provider,
        )
        self.query_service = QueryService(
            retriever=hybrid_retriever,
            answer_service=answer_service,
        )
        self._ingestion_results.extend(results)
        self._ingestion_result = results[-1]
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        """Retrieve relevant graph/vector results without generation."""

        self._ensure_ready()

        return self.query_service.search(
            query=query,
            top_k=top_k,
        )

    def ask(
        self,
        query: str,
        top_k: int = 5,
    ) -> AnswerResponse:
        """Retrieve context and generate an answer."""

        self._ensure_ready()

        return self.query_service.ask(
            query=query,
            top_k=top_k,
        )

    def _ensure_ready(self) -> None:
        if self.query_service is None:
            raise RuntimeError(
                "RAGAnythingApp is not ready. "
                "Call ingest() before search() or ask()."
            )