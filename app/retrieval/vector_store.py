from __future__ import annotations

from dataclasses import dataclass

import math


@dataclass
class VectorRecord:
    item_id: str
    vector: list[float]
    text: str
    metadata: dict


class InMemoryVectorStore:
    """Simple vector store using cosine similarity."""

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    @property
    def _items(self) -> dict[str, VectorRecord]:
        """Compatibility alias for the in-memory record collection."""
        return self._records

    def add(
        self,
        item_id: str,
        vector: list[float],
        text: str,
        metadata: dict | None = None,
    ) -> None:
        if not item_id:
            raise ValueError("item_id must not be empty.")

        if not vector:
            raise ValueError("vector must not be empty.")

        self._records[item_id] = VectorRecord(
            item_id=item_id,
            vector=vector,
            text=text,
            metadata=metadata or {},
        )

    def add_many(self, records: list[VectorRecord]) -> None:
        for record in records:
            self.add(
                item_id=record.item_id,
                vector=record.vector,
                text=record.text,
                metadata=record.metadata,
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[tuple[VectorRecord, float]]:
        if not query_vector:
            raise ValueError("query_vector must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        results: list[tuple[VectorRecord, float]] = []

        for record in self._records.values():
            score = self._cosine_similarity(
                query_vector,
                record.vector,
            )
            results.append((record, score))

        results.sort(
            key=lambda item: (-item[1], item[0].item_id)
        )

        return results[:top_k]

    def get(self, item_id: str) -> VectorRecord | None:
        return self._records.get(item_id)

    def delete(self, item_id: str) -> None:
        self._records.pop(item_id, None)

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        if len(vector_a) != len(vector_b):
            raise ValueError(
                "Vectors must have the same dimensions."
            )

        dot_product = sum(
            a * b for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = math.sqrt(
            sum(a * a for a in vector_a)
        )

        magnitude_b = math.sqrt(
            sum(b * b for b in vector_b)
        )

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)