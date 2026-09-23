from app.ingestion.content import ContentItem
from app.ingestion.document import Document
from app.ingestion.normalization.normalizer import ContentNormalizer


def build_document() -> Document:
    document = Document(
        document_id="doc-001",
        source_path="sample.pdf",
        filename="sample.pdf",
    )

    document.add_content(
        ContentItem(
            type="text",
            text="Introduction",
            page_idx=0,
        )
    )

    document.add_content(
        ContentItem(
            type="image",
            img_path="architecture.png",
            page_idx=1,
            caption=["Architecture"],
        )
    )

    document.add_content(
        ContentItem(
            type="table",
            table_body="| A | B |\n|---|---|\n| 1 | 2 |",
            page_idx=2,
        )
    )

    return document


def test_normalizer_preserves_document_identity():
    document = build_document()

    normalized = ContentNormalizer().normalize(document)

    assert normalized.document_id == "doc-001"
    assert normalized.filename == "sample.pdf"


def test_normalizer_preserves_content_order():
    document = build_document()

    normalized = ContentNormalizer().normalize(document)

    assert [element.position for element in normalized.elements] == [
        0,
        1,
        2,
    ]


def test_normalizer_assigns_unique_element_ids():
    document = build_document()

    normalized = ContentNormalizer().normalize(document)

    ids = [element.element_id for element in normalized.elements]

    assert len(ids) == 3
    assert len(set(ids)) == 3


def test_normalizer_creates_neighbor_relationships():
    document = build_document()

    normalized = ContentNormalizer().normalize(document)

    first = normalized.elements[0]
    second = normalized.elements[1]
    third = normalized.elements[2]

    assert first.previous_id is None
    assert first.next_id == second.element_id

    assert second.previous_id == first.element_id
    assert second.next_id == third.element_id

    assert third.previous_id == second.element_id
    assert third.next_id is None


def test_normalizer_preserves_multimodal_payload():
    document = build_document()

    normalized = ContentNormalizer().normalize(document)

    image = normalized.elements[1]
    table = normalized.elements[2]

    assert image.payload["img_path"] == "architecture.png"
    assert image.payload["caption"] == ["Architecture"]

    assert "| A | B |" in table.payload["table_body"]


def test_normalized_document_count():
    document = build_document()

    normalized = ContentNormalizer().normalize(document)

    assert normalized.element_count == 3