from app.ingestion.content import ContentItem
from app.ingestion.document import Document
from app.ingestion.hierarchy.builder import HierarchyBuilder
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
            page_idx=0,
        )
    )

    document.add_content(
        ContentItem(
            type="text",
            text="Implementation",
            page_idx=1,
        )
    )

    document.add_content(
        ContentItem(
            type="table",
            table_body="| A | B |\n|---|---|\n| 1 | 2 |",
            page_idx=1,
        )
    )

    document.add_content(
        ContentItem(
            type="equation",
            latex="x = 1",
            page_idx=2,
        )
    )

    return document


def test_hierarchy_preserves_document_identity():
    document = build_document()

    normalized = ContentNormalizer().normalize(document)
    hierarchy = HierarchyBuilder().build(normalized)

    assert hierarchy.document_id == "doc-001"


def test_hierarchy_creates_pages():
    document = build_document()

    normalized = ContentNormalizer().normalize(document)
    hierarchy = HierarchyBuilder().build(normalized)

    assert hierarchy.page_count == 3


def test_page_indexes_are_ordered():
    document = build_document()

    normalized = ContentNormalizer().normalize(document)
    hierarchy = HierarchyBuilder().build(normalized)

    assert [
        page.page_idx
        for page in hierarchy.page_nodes
    ] == [0, 1, 2]


def test_page_contains_correct_elements():
    document = build_document()

    normalized = ContentNormalizer().normalize(document)
    hierarchy = HierarchyBuilder().build(normalized)

    page_zero = hierarchy.page_nodes[0]
    page_one = hierarchy.page_nodes[1]
    page_two = hierarchy.page_nodes[2]

    assert len(page_zero.element_ids) == 2
    assert len(page_one.element_ids) == 2
    assert len(page_two.element_ids) == 1


def test_page_ids_are_unique():
    document = build_document()

    normalized = ContentNormalizer().normalize(document)
    hierarchy = HierarchyBuilder().build(normalized)

    ids = [
        page.page_id
        for page in hierarchy.page_nodes
    ]

    assert len(ids) == len(set(ids))


def test_hierarchy_handles_missing_page_indexes():
    document = Document(
        document_id="doc-002",
        source_path="sample.txt",
        filename="sample.txt",
    )

    document.add_content(
        ContentItem(
            type="text",
            text="No page information.",
        )
    )

    normalized = ContentNormalizer().normalize(document)
    hierarchy = HierarchyBuilder().build(normalized)

    assert hierarchy.page_count == 0