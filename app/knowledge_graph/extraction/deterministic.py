from __future__ import annotations

import re
from app.knowledge_graph.extraction.identity import EntityIdentity

from app.ingestion.normalization.models import NormalizedContent
from app.knowledge_graph.extraction.base import (
    EntityExtractor,
    RelationExtractor,
)
from app.knowledge_graph.extraction.entity import ExtractedEntity
from app.knowledge_graph.extraction.relation import ExtractedRelation


class DeterministicEntityExtractor(EntityExtractor):
    """
    Lightweight deterministic entity extractor.

    This implementation intentionally avoids an LLM. It provides a
    predictable baseline for testing the semantic graph pipeline.
    """

    ENTITY_PATTERN = re.compile(
        r"\b[A-Z][A-Za-z0-9_-]{2,}(?:\s+[A-Z][A-Za-z0-9_-]{2,})*\b"
    )

    def extract(
        self,
        content: NormalizedContent,
    ) -> list[ExtractedEntity]:

        if not content.text:
            return []

        matches = self.ENTITY_PATTERN.findall(content.text)

        entities: list[ExtractedEntity] = []
        seen: set[str] = set()

        for match in matches:
            name = match.strip()

            if name in seen:
                continue

            seen.add(name)

            entities.append(
                ExtractedEntity(
                    entity_id=EntityIdentity.generate_id(
                        name=name,
                        entity_type="UNKNOWN",
                    ),
                    name=name,
                    entity_type="UNKNOWN",
                    source_content_id=content.element_id,
                )
            )

        return entities


class DeterministicRelationExtractor(RelationExtractor):
    """
    Deterministic baseline relation extractor.

    Relations are inferred only between entities appearing in the same
    content element. This implementation deliberately remains simple
    until a semantic extraction provider is introduced.
    """

    def extract(
        self,
        content: NormalizedContent,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:

        if len(entities) < 2:
            return []

        relations: list[ExtractedRelation] = []

        for source, target in zip(entities, entities[1:]):

            relations.append(
                ExtractedRelation(
                    relation_id=EntityIdentity.generate_id(
                        name=f"{source.name} -> {target.name}",
                        entity_type="UNKNOWN",
                    ),
                    source_entity_id=source.entity_id,
                    target_entity_id=target.entity_id,
                    relation="RELATED_TO",
                    properties={
                        "source_content_id": content.element_id,
                    },
                )
            )

        return relations