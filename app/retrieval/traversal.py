from __future__ import annotations

from collections import deque

from app.knowledge_graph.graph import GraphManager
from app.retrieval.models import RetrievalResponse, RetrievalResult


class GraphTraversalRetriever:
    """
    Retrieve graph candidates by traversing relationships from seed nodes.

    The retriever starts from nodes returned by GraphRetriever and expands
    through the existing graph relationships.
    """

    def __init__(
        self,
        graph: GraphManager,
        seed_retriever=None,
    ) -> None:
        from app.retrieval.graph import GraphRetriever

        self.graph = graph
        self.seed_retriever = (
            seed_retriever
            or GraphRetriever(graph)
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_depth: int = 2,
    ) -> RetrievalResponse:

        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if max_depth < 0:
            raise ValueError("max_depth must not be negative.")

        seed_response = self.seed_retriever.retrieve(
            query=query,
            top_k=1,
        )

        candidates: dict[str, RetrievalResult] = {}

        for seed in seed_response.results:
            self._add_candidate(
                candidates,
                seed,
            )

            self._traverse(
                seed_id=seed.item_id,
                seed_score=seed.score,
                max_depth=max_depth,
                candidates=candidates,
            )

        ranked = sorted(
            candidates.values(),
            key=lambda result: (
                -result.score,
                result.item_id,
            ),
        )

        return RetrievalResponse(
            query=query,
            results=ranked[:top_k],
        )

    def _traverse(
        self,
        seed_id: str,
        seed_score: float,
        max_depth: int,
        candidates: dict[str, RetrievalResult],
    ) -> None:

        queue = deque(
            [(seed_id, 1)]
        )

        visited = {
            seed_id,
        }

        while queue:

            node_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            neighbors = self.graph.get_neighbors(
                node_id
            )

            for neighbor in neighbors:

                if neighbor.node_id in visited:
                    continue

                visited.add(
                    neighbor.node_id
                )

                traversal_score = seed_score / (
                    depth + 1
                )

                result = self._node_to_result(
                    neighbor,
                    traversal_score,
                )

                existing = candidates.get(
                    neighbor.node_id
                )

                if (
                    existing is None
                    or result.score > existing.score
                ):
                    candidates[
                        neighbor.node_id
                    ] = result

                queue.append(
                    (
                        neighbor.node_id,
                        depth + 1,
                    )
                )

    @staticmethod
    def _add_candidate(
        candidates: dict[str, RetrievalResult],
        result: RetrievalResult,
    ) -> None:

        existing = candidates.get(
            result.item_id
        )

        if (
            existing is None
            or result.score > existing.score
        ):
            candidates[
                result.item_id
            ] = result

    @staticmethod
    def _node_to_result(
        node,
        score: float,
    ) -> RetrievalResult:

        return RetrievalResult(
            item_id=node.node_id,
            score=score,
            source="graph_traversal",
            content_type=node.node_type,
            text=(
                node.properties.get("text")
                or node.properties.get("extracted_text")
                or node.label
            ),
            metadata={
                **node.properties,
            },
        )