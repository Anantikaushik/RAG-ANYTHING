from pathlib import Path

import pytest

from app.ingestion.ocr.deterministic import DeterministicOCRProvider


def test_deterministic_ocr_returns_empty_text(tmp_path):
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"fake-image")

    provider = DeterministicOCRProvider()

    assert provider.extract_text(image_path) == ""


def test_deterministic_ocr_missing_file():
    provider = DeterministicOCRProvider()

    with pytest.raises(FileNotFoundError):
        provider.extract_text("missing.png")


def test_deterministic_ocr_rejects_directory(tmp_path):
    provider = DeterministicOCRProvider()

    with pytest.raises(ValueError):
        provider.extract_text(tmp_path)