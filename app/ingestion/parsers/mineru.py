from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv

load_dotenv()

_data_root = os.getenv("RAG_ANYTHING_DATA")
if _data_root:
    _mineru_temp = str(Path(_data_root) / "mineru" / "temp")
    os.environ["TEMP"] = _mineru_temp
    os.environ["TMP"] = _mineru_temp
    tempfile.tempdir = _mineru_temp

_mineru_config = os.getenv("MINERU_CONFIG")
if _mineru_config and not Path(_mineru_config).is_absolute():
    _project_root = Path(__file__).resolve().parents[3]
    os.environ["MINERU_CONFIG"] = str(
        (_project_root / _mineru_config).resolve()
    )

from mineru.parser.mineru_parser import (
    MinerUParser as OfficialMinerUParser,
)
from mineru.parser.writer import FileBasedDataWriter
from mineru.render.api import render
from mineru.render.contracts import (
    ContentListRenderOptions,
    RenderFormat,
)

from app.ingestion.content import ContentItem
from app.ingestion.document import Document
from app.ingestion.parsers.base import DocumentParser


class MinerUParser(DocumentParser):
    """Parse supported document formats through MinerU's unified pipeline."""

    def __init__(
        self,
        output_dir: Path | None = None,
        storage_dir: Path | None = None,
        minimum_free_space_mb: int | None = None,
    ) -> None:
        self.output_dir = output_dir or Path(
            os.getenv(
                "MINERU_OUTPUT_DIR",
                "D:\\RAG_ANYTHING_DATA\\outputs",
            )
        )
        self.storage_dir = storage_dir or Path(
            os.getenv(
                "MINERU_STORAGE_DIR",
                "D:\\RAG_ANYTHING_DATA\\uploads",
            )
        )
        self._ensure_directories()
        self.minimum_free_space_mb = minimum_free_space_mb or int(
            os.getenv("MINERU_MIN_FREE_SPACE_MB", "1024")
        )

    @property
    def supported_extensions(self) -> set[str]:
        # MinerU accepts more formats than the old parser pair. Keep this
        # registry open so new MinerU-supported formats need no code change.
        return {
            ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
            ".csv", ".tsv", ".html", ".epub", ".ofd", ".rtf", ".odt",
            ".ods", ".odp", ".png", ".jpg", ".jpeg", ".webp", ".bmp",
        }

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions

    def parse(self, path: Path) -> Document:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        if not path.is_file():
            raise ValueError(f"Expected a file, got: {path}")

        self._ensure_disk_space()
        document_id = str(uuid4())
        stored_path = self._store_source(path, document_id)
        return self._parse_with_mineru(
            stored_path,
            document_id=document_id,
            original_path=path,
        )

    def _ensure_disk_space(self) -> None:
        required_bytes = self.minimum_free_space_mb * 1024 * 1024
        free_bytes = shutil.disk_usage(self.storage_dir).free
        if free_bytes < required_bytes:
            free_mb = free_bytes / (1024 * 1024)
            raise RuntimeError(
                "Not enough disk space for MinerU model initialization. "
                f"Free space: {free_mb:.1f} MB; required minimum: "
                f"{self.minimum_free_space_mb} MB. "
                "Free disk space and retry."
            )

    def _ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        mineru_home = os.getenv("MINERU_HOME")
        if mineru_home:
            for name in ("models", "cache", "temp"):
                (Path(mineru_home) / name).mkdir(
                    parents=True,
                    exist_ok=True,
                )

    def _store_source(self, path: Path, document_id: str) -> Path:
        source_dir = self.storage_dir / document_id / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        stored_path = source_dir / path.name
        shutil.copy2(path, stored_path)
        return stored_path

    def _parse_with_mineru(
        self,
        path: Path,
        *,
        document_id: str,
        original_path: Path,
    ) -> Document:
        output_dir = self.output_dir / document_id
        output_dir.mkdir(parents=True, exist_ok=True)

        parser = OfficialMinerUParser(
            tier=os.getenv("MINERU_TIER", "standard"),
            parse_mode=os.getenv("MINERU_PARSE_MODE", "auto"),
            image_analysis=os.getenv(
                "ENABLE_IMAGE_PROCESSING",
                "true",
            ).lower() == "true",
        )
        result = parser.parse(path)
        result.save(FileBasedDataWriter(parent_dir=str(output_dir)))
        raw_items = render(
            result.middle_json,
            RenderFormat.CONTENT_LIST,
            options=ContentListRenderOptions(
                asset_base_url=str(output_dir),
            ),
        )

        document = Document(
            document_id=document_id,
            source_path=str(path),
            filename=path.name,
            metadata={
                "parser": "mineru",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "original_source_path": str(original_path),
                "stored_source_path": str(path),
                "mineru_output_dir": str(output_dir),
            },
        )
        for item in raw_items:
            content = self._content_item(item, output_dir)
            if content is not None:
                document.add_content(content)

        page_indexes = [
            item.page_idx
            for item in document.content
            if item.page_idx is not None
        ]
        document.metadata["page_count"] = (
            max(page_indexes) + 1 if page_indexes else 0
        )
        return document

    @staticmethod
    def _content_item(item: object, asset_dir: Path) -> ContentItem | None:
        if not isinstance(item, dict):
            raise ValueError("MinerU content-list entries must be objects")

        item_type = item.get("type")
        page_idx = item.get("page_idx", item.get("page_id"))
        if page_idx is not None:
            page_idx = int(page_idx)
        caption = MinerUParser._as_list(
            item.get("caption")
            or item.get("image_caption")
            or item.get("table_caption")
            or item.get("chart_caption")
        )
        footnote = MinerUParser._as_list(
            item.get("footnote")
            or item.get("image_footnote")
            or item.get("table_footnote")
            or item.get("chart_footnote")
        )
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {
                "type", "text", "img_path", "table_body", "table",
                "latex", "caption", "footnote", "page_idx", "page_id",
                "image_caption", "image_footnote", "table_caption",
                "table_footnote", "chart_caption", "chart_footnote",
            }
        }

        if item_type == "text" and item.get("text", "").strip():
            return ContentItem(
                type="text",
                text=item["text"],
                page_idx=page_idx,
                caption=caption,
                footnote=footnote,
                metadata=metadata,
            )
        if item_type in {"list", "index", "code", "page_footnote"}:
            text = item.get("text")
            list_items = item.get("list_items")
            if not text and isinstance(list_items, list):
                text = "\n".join(
                    str(value)
                    for value in list_items
                    if str(value).strip()
                )
            if text and str(text).strip():
                return ContentItem(
                    type="text",
                    text=str(text),
                    page_idx=page_idx,
                    caption=caption,
                    footnote=footnote,
                    metadata={**metadata, "source_type": item_type},
                )
        if item_type in {"image", "chart"} and item.get("img_path"):
            image_path = MinerUParser._asset_path(
                item["img_path"],
                asset_dir,
            )
            return ContentItem(
                type="image",
                img_path=str(image_path),
                page_idx=page_idx,
                caption=caption,
                footnote=footnote,
                metadata={**metadata, "source_type": item_type},
            )
        if item_type == "table":
            table_body = item.get("table_body", item.get("table"))
            if table_body:
                return ContentItem(
                    type="table",
                    table_body=table_body,
                    page_idx=page_idx,
                    caption=caption,
                    footnote=footnote,
                    metadata=metadata,
                )
            if item.get("img_path"):
                image_path = MinerUParser._asset_path(
                    item["img_path"],
                    asset_dir,
                )
                return ContentItem(
                    type="image",
                    img_path=str(image_path),
                    page_idx=page_idx,
                    caption=caption,
                    footnote=footnote,
                    metadata={**metadata, "source_type": "table"},
                )
        latex = item.get("latex")
        if not latex and item.get("text_format") == "latex":
            latex = item.get("text")
        if item_type in {"equation", "latex"} and latex:
            return ContentItem(
                type="equation",
                latex=latex,
                page_idx=page_idx,
                caption=caption,
                footnote=footnote,
                metadata=metadata,
            )
        return None

    @staticmethod
    def _asset_path(value: object, asset_dir: Path) -> Path:
        """Resolve MinerU asset URLs and paths to a local filesystem path."""
        if not isinstance(value, str):
            raise ValueError("MinerU asset paths must be strings")

        parsed = urlparse(value)
        if parsed.scheme == "file":
            decoded = unquote(parsed.path)
            if len(decoded) > 2 and decoded[0] == "/" and decoded[2] == ":":
                decoded = decoded[1:]
        else:
            decoded = unquote(value)

        image_path = Path(decoded)
        if not image_path.is_absolute():
            image_path = asset_dir / image_path
        return image_path

    @staticmethod
    def _as_list(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and all(
            isinstance(item, str) for item in value
        ):
            return value
        raise ValueError("MinerU captions and footnotes must be strings")
