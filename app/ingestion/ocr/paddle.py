from __future__ import annotations

from pathlib import Path

from app.ingestion.ocr.base import OCRProvider


class PaddleOCRProvider(OCRProvider):
    """OCR provider backed by PaddleOCR."""

    def __init__(
        self,
        lang: str = "en",
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        enable_mkldnn: bool = False,
    ) -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            enable_mkldnn=enable_mkldnn,
        )

    def extract_text(self, image_path: str | Path) -> str:
        path = Path(image_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        if not path.is_file():
            raise ValueError(f"Image path is not a file: {path}")

        result = self._ocr.predict(str(path))

        texts: list[str] = []

        for page_result in result:
            if hasattr(page_result, "json"):
                data = page_result.json
                if callable(data):
                    data = data()

                if isinstance(data, dict):
                    data = data.get("res", data)
                    texts.extend(
                        str(text)
                        for text in data.get("rec_texts", [])
                        if text
                    )

        return "\n".join(texts).strip()