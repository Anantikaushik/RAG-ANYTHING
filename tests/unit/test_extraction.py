from app.ingestion.normalization.models import NormalizedContent
from app.knowledge_graph.extraction.deterministic import (
    DeterministicEntityExtractor,
    DeterministicRelationExtractor,
)


def make_content(text: str) -> NormalizedContent:
    return NormalizedContent(
        element_id="content-001",
        document_id="doc-001",
        type="text",
        page_idx=0,
        position=0,
        text=text,
    )


def test_entity_extractor_finds_entities():
    content = make_content(
        "Microsoft develops Azure services."
    )

    extractor = DeterministicEntityExtractor()

    entities = extractor.extract(content)

    names = [entity.name for entity in entities]

    assert "Microsoft" in names
    assert "Azure" in names


def test_entity_extractor_returns_empty_for_missing_text():
    content = make_content("")

    extractor = DeterministicEntityExtractor()

    assert extractor.extract(content) == []


def test_relation_extractor_connects_entities():
    content = make_content(
        "Microsoft develops Azure services."
    )

    entity_extractor = DeterministicEntityExtractor()
    relation_extractor = DeterministicRelationExtractor()

    entities = entity_extractor.extract(content)
    relations = relation_extractor.extract(content, entities)

    assert len(entities) >= 2
    assert len(relations) >= 1

    relation = relations[0]

    assert relation.source_entity_id == entities[0].entity_id
    assert relation.target_entity_id == entities[1].entity_id
    assert relation.relation == "RELATED_TO"


def test_relation_extractor_requires_two_entities():
    content = make_content("Microsoft")

    entity_extractor = DeterministicEntityExtractor()
    relation_extractor = DeterministicRelationExtractor()

    entities = entity_extractor.extract(content)

    assert len(entities) == 1
    assert relation_extractor.extract(content, entities) == []