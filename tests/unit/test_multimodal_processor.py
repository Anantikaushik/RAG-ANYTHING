import pytest

from app.ingestion.content import ContentItem
from app.ingestion.content_list import ContentList
from app.multimodal.processor import MultimodalProcessor


def test_processor_handles_text():
    content = ContentList(
        [
            ContentItem(
                type="text",
                text="Hello RAG Anything.",
                page_idx=0,
            )
        ]
    )

    result = MultimodalProcessor().process(content)

    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "Hello RAG Anything."


def test_processor_routes_image():
    content = ContentList(
        [
            ContentItem(
                type="image",
                img_path="figure.png",
                page_idx=1,
            )
        ]
    )

    result = MultimodalProcessor().process(content)

    assert len(result) == 1
    assert result[0]["type"] == "image"
    assert result[0]["path"] == "figure.png"


def test_processor_routes_table():
    content = ContentList(
        [
            ContentItem(
                type="table",
                table_body="| A | B |\n|---|---|\n| 1 | 2 |",
                page_idx=2,
            )
        ]
    )

    result = MultimodalProcessor().process(content)

    assert result[0]["type"] == "table"


def test_processor_routes_equation():
    content = ContentList(
        [
            ContentItem(
                type="equation",
                latex=r"E = mc^2",
                page_idx=3,
            )
        ]
    )

    result = MultimodalProcessor().process(content)

    assert result[0]["type"] == "equation"
    assert result[0]["latex"] == r"E = mc^2"


def test_processor_rejects_unregistered_type():
    content = ContentList(
        [
            ContentItem(
                type="generic",
                metadata={"source": "unknown"},
            )
        ]
    )

    with pytest.raises(ValueError):
        MultimodalProcessor().process(content)


def test_processor_preserves_order():
    content = ContentList(
        [
            ContentItem(type="text", text="First", page_idx=0),
            ContentItem(type="image", img_path="one.png", page_idx=0),
            ContentItem(
                type="table",
                table_body="| A |\n|---|\n| 1 |",
                page_idx=1,
            ),
            ContentItem(
                type="equation",
                latex="x = 1",
                page_idx=2,
            ),
        ]
    )

    result = MultimodalProcessor().process(content)

    assert [item["type"] for item in result] == [
        "text",
        "image",
        "table",
        "equation",
    ]
from pathlib import Path

from app.ingestion.ocr.deterministic import DeterministicOCRProvider
from app.multimodal.processor import MultimodalProcessor


def test_multimodal_processor_ocr_provider(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image")

    processor = MultimodalProcessor(
        ocr_provider=DeterministicOCRProvider()
    )

    assert processor.extract_image_text(image_path) == ""


def test_multimodal_processor_without_ocr_provider(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image")

    processor = MultimodalProcessor()

    assert processor.extract_image_text(image_path) == ""
