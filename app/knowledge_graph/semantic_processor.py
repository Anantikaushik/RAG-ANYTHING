from __future__ import annotations
from copy import copy

from app.ingestion.normalization.models import NormalizedDocument
from app.knowledge_graph.enricher import GraphEnricher
from app.knowledge_graph.extraction.base import (
    EntityExtractor,
    RelationExtractor,
)
from app.knowledge_graph.extraction.deterministic import (
    DeterministicEntityExtractor,
    DeterministicRelationExtractor,
)
from app.knowledge_graph.extraction.entity import ExtractedEntity
from app.knowledge_graph.extraction.relation import ExtractedRelation
from app.knowledge_graph.graph import GraphManager
from app.multimodal.analysis.models import MultimodalAnalysis
from app.multimodal.analysis.processor import MultimodalAnalysisProcessor


class SemanticGraphProcessor:
    """
    Extract semantic entities and relationships from document content.

    Text content is processed directly by the configured entity and
    relation extractors.

    Multimodal content is first converted into normalized semantic
    analysis by MultimodalAnalysisProcessor. Any extracted textual
    representation is then passed through the same semantic extraction
    pipeline.

    All resulting entities and relationships are enriched into the
    existing knowledge graph.
    """

    def __init__(
        self,
        graph: GraphManager,
        entity_extractor: EntityExtractor | None = None,
        relation_extractor: RelationExtractor | None = None,
        multimodal_processor: MultimodalAnalysisProcessor | None = None,
        enable_entity_extraction: bool = True,
        enable_relation_extraction: bool = True,
    ) -> None:
        self.graph = graph
        self.enable_entity_extraction = enable_entity_extraction
        self.enable_relation_extraction = enable_relation_extraction

        self.entity_extractor = (
            entity_extractor
            or DeterministicEntityExtractor()
        )

        self.relation_extractor = (
            relation_extractor
            or DeterministicRelationExtractor()
        )

        self.multimodal_processor = (
            multimodal_processor
            or MultimodalAnalysisProcessor()
        )

        self.enricher = GraphEnricher(graph)

    def process(
        self,
        document: NormalizedDocument,
    ) -> GraphManager:
        """
        Extract semantic entities and relationships from all supported
        document content.
        """

        if not self.enable_entity_extraction:
            return self.graph

        all_entities: list[ExtractedEntity] = []
        all_relations: list[ExtractedRelation] = []

        for element in document.elements:

            # -----------------------------------------------------
            # Text content
            # -----------------------------------------------------

            if element.type == "text":
                if not element.text or not element.text.strip():
                    continue

                entities = self.entity_extractor.extract(element)

                if not entities:
                    continue

                relations = (
                    self.relation_extractor.extract(
                        element,
                        entities,
                    )
                    if self.enable_relation_extraction
                    else []
                )

                all_entities.extend(entities)
                all_relations.extend(relations)

                continue

            # -----------------------------------------------------
            # Multimodal content
            # -----------------------------------------------------

            analysis = self.multimodal_processor.analyze(
                element
            )

            entities, relations = self._extract_from_multimodal_analysis(
                element,
                analysis,
            )

            all_entities.extend(entities)
            all_relations.extend(relations)

        # ---------------------------------------------------------
        # Enrich graph
        # ---------------------------------------------------------

        self.enricher.enrich(
            entities=all_entities,
            relations=all_relations,
        )

        return self.graph

    def _extract_from_multimodal_analysis(
        self,
        element,
        analysis: MultimodalAnalysis,
    ) -> tuple[
        list[ExtractedEntity],
        list[ExtractedRelation],
    ]:
        """
        Convert multimodal analysis into the canonical semantic
        entity/relation representation.

        The current deterministic analyzers may provide extracted text
        but do not yet produce structured entity/relation objects.
        Therefore extracted text is passed through the existing
        semantic extractors.
        """

        extracted_text = analysis.extracted_text

        if not extracted_text or not extracted_text.strip():
            return [], []

        # Create a lightweight normalized-content view so the existing
        # extractors can process multimodal extracted text without
        # introducing a second extraction contract.
        semantic_element = copy(element)
        semantic_element.text = extracted_text

        entities = self.entity_extractor.extract(
            semantic_element
        )

        if not entities:
            return [], []

        relations = (
            self.relation_extractor.extract(
                semantic_element,
                entities,
            )
            if self.enable_relation_extraction
            else []
        )

        return entities, relations