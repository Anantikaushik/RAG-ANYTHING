from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from app.retrieval.models import RetrievalResult


class ContextItem(BaseModel):
    item_id: str
    content_type: str | None = None
    text: str
    score: float
    source: str
    metadata: dict = Field(default_factory=dict)


class QueryContext(BaseModel):
    query: str
    items: list[ContextItem] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)

    def as_text(self, max_chars: int | None = None) -> str:
        sections = []
        current_chars = 0

        for index, item in enumerate(self.items, start=1):
            section = (
                f"[Context {index}]\n"
                f"Type: {item.content_type or 'unknown'}\n"
                f"Source: {item.source}\n"
                f"Score: {item.score:.4f}\n"
                f"Content:\n{item.text}"
            )
            separator_size = 2 if sections else 0

            if (
                max_chars is not None
                and sections
                and current_chars + separator_size + len(section) > max_chars
            ):
                break

            sections.append(section)
            current_chars += separator_size + len(section)

        return "\n\n".join(sections)


class ContextAssembler:
    """Converts retrieval results into LLM-ready context."""

    def assemble(
        self,
        query: str,
        results: list[RetrievalResult | Mapping[str, Any]],
    ) -> QueryContext:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        items = []

        for result in results:
            if isinstance(result, Mapping):
                result = RetrievalResult.model_validate(result)

            text = result.text

            if not text or not text.strip():
                continue

            items.append(
                ContextItem(
                    item_id=result.item_id,
                    content_type=result.content_type,
                    text=text.strip(),
                    score=result.score,
                    source=result.source,
                    metadata=result.metadata,
                )
            )

        return QueryContext(
            query=query,
            items=items,
        )