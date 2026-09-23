import pytest

from app.ingestion.content import ContentItem
from app.multimodal.extractors.equation import EquationExtractor
from app.multimodal.extractors.image import ImageExtractor
from app.multimodal.extractors.table import TableExtractor


def test_image_extractor():
    item = ContentItem(
        type="image",
        img_path="figure.png",
        page_idx=1,
        caption=["System architecture"],
    )

    result = ImageExtractor().extract(item)

    assert result["type"] == "image"
    assert result["path"] == "figure.png"
    assert result["page_idx"] == 1


def test_table_extractor():
    item = ContentItem(
        type="table",
        table_body="| A | B |\n|---|---|\n| 1 | 2 |",
        page_idx=2,
    )

    result = TableExtractor().extract(item)

    assert result["type"] == "table"
    assert "| A | B |" in result["body"]


def test_equation_extractor():
    item = ContentItem(
        type="equation",
        latex=r"E = mc^2",
        page_idx=3,
    )

    result = EquationExtractor().extract(item)

    assert result["type"] == "equation"
    assert result["latex"] == r"E = mc^2"


def test_image_extractor_rejects_text():
    item = ContentItem(
        type="text",
        text="This is text.",
    )

    with pytest.raises(ValueError):
        ImageExtractor().extract(item)