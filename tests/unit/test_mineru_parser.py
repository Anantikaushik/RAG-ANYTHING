from pathlib import Path

from app.ingestion.parsers.mineru import MinerUParser


def test_mineru_parser_stores_source_before_parsing(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"document bytes")

    parser = MinerUParser(
        output_dir=tmp_path / "outputs",
        storage_dir=tmp_path / "storage",
    )
    stored = parser._store_source(source, "document-id")

    assert stored == (
        tmp_path / "storage" / "document-id" / "source" / "report.pdf"
    )
    assert stored.read_bytes() == b"document bytes"


def test_mineru_parser_reports_insufficient_disk_space(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parser = MinerUParser(
        output_dir=tmp_path / "outputs",
        storage_dir=tmp_path / "storage",
        minimum_free_space_mb=1024,
    )

    class Usage:
        free = 1

    monkeypatch.setattr(
        "app.ingestion.parsers.mineru.shutil.disk_usage",
        lambda _: Usage(),
    )

    try:
        parser._ensure_disk_space()
    except RuntimeError as exc:
        assert "Not enough disk space" in str(exc)
    else:
        raise AssertionError("Expected the disk-space preflight to fail")


def test_mineru_parser_supports_official_formats() -> None:
    parser = MinerUParser()

    assert parser.supports(Path("document.pdf"))
    assert parser.supports(Path("document.docx"))
    assert parser.supports(Path("document.xlsx"))
    assert parser.supports(Path("document.png"))
    assert not parser.supports(Path("document.txt"))


def test_mineru_content_list_items_map_to_project_content() -> None:
    parser = MinerUParser()
    asset_dir = Path("data/outputs/mineru")

    items = [
        {"type": "text", "text": "Heading", "page_idx": 0},
        {"type": "table", "table_body": "<table />", "page_idx": 1},
        {"type": "equation", "latex": "E = mc^2", "page_idx": 2},
        {"type": "image", "img_path": "images/page.png", "page_idx": 3},
    ]

    content = [
        parser._content_item(item, asset_dir)
        for item in items
    ]

    assert [item.type for item in content if item is not None] == [
        "text",
        "table",
        "equation",
        "image",
    ]
    assert content[-1] is not None
    assert content[-1].img_path == str(
        asset_dir / "images/page.png"
    )


def test_mineru_official_metadata_and_block_shapes_are_preserved() -> None:
    parser = MinerUParser()
    asset_dir = Path("data/outputs/mineru")

    image = parser._content_item(
        {
            "type": "image",
            "img_path": "images/page.png",
            "image_caption": ["Architecture"],
            "image_footnote": ["Source"],
            "page_idx": 0,
        },
        asset_dir,
    )
    equation = parser._content_item(
        {
            "type": "equation",
            "text": "E = mc^2",
            "text_format": "latex",
            "page_idx": 1,
        },
        asset_dir,
    )
    list_item = parser._content_item(
        {
            "type": "list",
            "list_items": ["One", "Two"],
            "page_idx": 2,
        },
        asset_dir,
    )

    assert image is not None
    assert image.caption == ["Architecture"]
    assert image.footnote == ["Source"]
    assert equation is not None
    assert equation.latex == "E = mc^2"
    assert list_item is not None
    assert list_item.type == "text"
    assert list_item.text == "One\nTwo"


def test_mineru_decodes_url_encoded_windows_asset_paths() -> None:
    asset_dir = Path("data/outputs/mineru")

    content = MinerUParser._content_item(
        {
            "type": "image",
            "img_path": "D:%5CRAG_ANYTHING_DATA%5Coutputs%5Cimages%5Cpage.png",
            "page_idx": 0,
        },
        asset_dir,
    )

    assert content is not None
    assert content.img_path == r"D:\RAG_ANYTHING_DATA\outputs\images\page.png"
