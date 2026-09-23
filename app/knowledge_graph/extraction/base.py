from __future__ import annotations

from abc import ABC, abstractmethod

from app.ingestion.normalization.models import NormalizedContent
from app.knowledge_graph.extraction.entity import ExtractedEntity
from app.knowledge_graph.extraction.relation import ExtractedRelation


class EntityExtractor(ABC):
    """Contract for extracting entities from normalized content."""

    @abstractmethod
    def extract(
        self,
        content: NormalizedContent,
    ) -> list[ExtractedEntity]:
        raise NotImplementedError


class RelationExtractor(ABC):
    """Contract for extracting relations from normalized content."""

    @abstractmethod
    def extract(
        self,
        content: NormalizedContent,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        raise NotImplementedError
