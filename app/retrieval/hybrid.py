from __future__ import annotations

from app.retrieval.models import RetrievalResponse, RetrievalResult


class HybridRetriever:
    """Combines graph and vector retrieval results."""

    def __init__(
        self,
        graph_retriever,
        vector_retriever,
        graph_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> None:
        if graph_weight < 0:
            raise ValueError("graph_weight must not be negative.")

        if vector_weight < 0:
            raise ValueError("vector_weight must not be negative.")

        if graph_weight + vector_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be greater than zero."
            )

        self.graph_retriever = graph_retriever
        self.vector_retriever = vector_retriever

        total = graph_weight + vector_weight

        self.graph_weight = graph_weight / total
        self.vector_weight = vector_weight / total

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResponse:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        graph_response = self.graph_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        vector_response = self.vector_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        fused = self._fuse(
            graph_response.results,
            vector_response.results,
        )

        return RetrievalResponse(
            query=query,
            results=fused[:top_k],
        )

    def _fuse(
        self,
        graph_results: list[RetrievalResult],
        vector_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}

        for result in graph_results:
            scores[result.item_id] = (
                scores.get(result.item_id, 0.0)
                + result.score * self.graph_weight
            )
            result_map[result.item_id] = result

        for result in vector_results:
            scores[result.item_id] = (
                scores.get(result.item_id, 0.0)
                + result.score * self.vector_weight
            )

            existing = result_map.get(result.item_id)
            if existing is None:
                result_map[result.item_id] = result
            else:
                # Vector records contain the indexed document text; keep it
                # over graph labels when both retrievers return one item.
                result_map[result.item_id] = result.model_copy(
                    update={
                        "metadata": {
                            **existing.metadata,
                            **result.metadata,
                        },
                    }
                )

        fused_results = []

        for item_id, score in scores.items():
            result = result_map[item_id]

            fused_results.append(
                result.model_copy(
                    update={
                        "score": score,
                        "source": "hybrid",
                    }
                )
            )

        fused_results.sort(
            key=lambda result: (-result.score, result.item_id)
        )

        return fused_results