from app.ingestion.ocr.base import OCRProvider
from app.ingestion.ocr.deterministic import DeterministicOCRProvider
from app.ingestion.ocr.paddle import PaddleOCRProvider

__all__ = [
    "OCRProvider",
    "DeterministicOCRProvider",
    "PaddleOCRProvider",
]