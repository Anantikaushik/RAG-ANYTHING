import pytest

from app.ingestion.content import ContentItem
from app.ingestion.content_list import ContentList
from app.ingestion.document import Document


def test_text_content():
    item = ContentItem(
        type="text",
        text="Hello RAG Anything",
        page_idx=0,
    )

    assert item.is_text
    assert not item.is_multimodal


def test_image_content():
    item = ContentItem(
        type="image",
        img_path="figure.png",
        page_idx=1,
        caption=["Architecture"],
    )

    assert item.is_multimodal
    assert item.img_path == "figure.png"


def test_table_content():
    item = ContentItem(
        type="table",
        table_body="| A | B |\n|---|---|\n| 1 | 2 |",
        page_idx=2,
    )

    assert item.type == "table"


def test_equation_content():
    item = ContentItem(
        type="equation",
        latex=r"E = mc^2",
        page_idx=3,
    )

    assert item.type == "equation"


def test_invalid_text():
    with pytest.raises(ValueError):
        ContentItem(
            type="text",
            text="",
        )


def test_content_list():
    items = ContentList()

    items.add(
        ContentItem(
            type="text",
            text="Hello",
        )
    )

    items.add(
        ContentItem(
            type="image",
            img_path="figure.png",
        )
    )

    assert len(items) == 2
    assert len(items.text_items) == 1
    assert len(items.multimodal_items) == 1


def test_document():
    document = Document(
        document_id="doc-001",
        source_path="data/uploads/test.pdf",
        filename="test.pdf",
    )

    document.add_content(
        ContentItem(
            type="text",
            text="Page one",
            page_idx=0,
        )
    )

    document.add_content(
        ContentItem(
            type="image",
            img_path="figure.png",
            page_idx=1,
        )
    )

    assert document.content_count == 2
    assert document.page_count == 2