from __future__ import annotations

from pathlib import Path

from app.ingestion.ocr.base import OCRProvider


class DeterministicOCRProvider(OCRProvider):
    """
    Deterministic OCR fallback.

    This provider does not perform actual OCR. It validates the image
    path and returns an empty result. Real OCR providers can replace it.
    """

    def extract_text(self, image_path: str | Path) -> str:
        path = Path(image_path).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        if not path.is_file():
            raise ValueError(f"Image path is not a file: {path}")

        return ""