from __future__ import annotations

from abc import ABC, abstractmethod

from app.query.context import QueryContext


class GenerationProvider(ABC):
    """Interface for LLM/VLM answer generation."""

    @abstractmethod
    def generate(
        self,
        query: str,
        context: QueryContext,
    ) -> str:
        raise NotImplementedError