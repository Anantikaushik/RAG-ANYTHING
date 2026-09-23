from __future__ import annotations

import re

from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.models.node import GraphNode
from app.retrieval.models import RetrievalResponse, RetrievalResult


class GraphRetriever:
    """
    Retrieve graph content using deterministic lexical matching.

    This is the baseline graph retrieval strategy. It operates directly
    on the existing GraphManager and does not introduce another graph
    abstraction.

    Later, this component can be complemented by graph traversal and
    vector retrieval without changing the RetrievalResult contract.
    """

    def __init__(self, graph: GraphManager) -> None:
        self.graph = graph

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResponse:
        """
        Retrieve graph nodes relevant to the query.

        Matching is based on normalized query tokens against node labels,
        textual properties, and semantic metadata.
        """

        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = self._tokenize(query)

        candidates: list[RetrievalResult] = []

        # The graph manager is shared by all ingested documents.  Do not
        # restrict retrieval to the last active document: queries are
        # intentionally cross-document, while the UI can still scope views.
        nodes = self.graph.graph.get_nodes()

        for node in nodes:
            score = self._score_node(
                node,
                query_tokens,
            )

            if score <= 0:
                continue

            candidates.append(
                self._to_result(
                    node=node,
                    score=score,
                )
            )

        candidates.sort(
            key=lambda result: (
                -result.score,
                result.item_id,
            )
        )

        return RetrievalResponse(
            query=query,
            results=candidates[:top_k],
        )

    def _score_node(
        self,
        node: GraphNode,
        query_tokens: set[str],
    ) -> float:
        """
        Calculate a deterministic lexical relevance score.

        Higher weight is given to node labels and names because these
        fields represent semantic identity more directly than arbitrary
        metadata.
        """

        if not query_tokens:
            return 0.0

        label_tokens = self._tokenize(node.label or "")

        name_tokens = self._tokenize(
            str(node.properties.get("name", ""))
        )

        text_tokens = self._tokenize(
            str(node.properties.get("text", ""))
        )

        extracted_text_tokens = self._tokenize(
            str(
                node.properties.get(
                    "extracted_text",
                    "",
                )
            )
        )

        score = 0.0

        score += len(query_tokens & label_tokens) * 3.0
        score += len(query_tokens & name_tokens) * 3.0
        score += len(query_tokens & text_tokens) * 1.0
        score += (
            len(query_tokens & extracted_text_tokens)
            * 1.5
        )

        return score

    @staticmethod
    def _tokenize(value: str) -> set[str]:
        """
        Normalize text into lexical tokens.
        """

        return {
            token
            for token in re.findall(
                r"[a-zA-Z0-9_]+",
                value.lower(),
            )
            if len(token) > 1
        }

    @staticmethod
    def _to_result(
        node: GraphNode,
        score: float,
    ) -> RetrievalResult:
        """
        Convert a graph node into the canonical retrieval result.
        """

        return RetrievalResult(
            item_id=node.node_id,
            score=score,
            source="graph",
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