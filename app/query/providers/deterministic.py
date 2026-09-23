from __future__ import annotations

from app.query.context import QueryContext
from app.query.providers.base import GenerationProvider


class DeterministicGenerationProvider(GenerationProvider):
    """Predictable provider used for tests and local development."""

    def generate(
        self,
        query: str,
        context: QueryContext,
    ) -> str:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if context.count == 0:
            return "No relevant context was found."

        first_item = context.items[0]

        return (
            f"Answer based on retrieved context: "
            f"{first_item.text}"
        )