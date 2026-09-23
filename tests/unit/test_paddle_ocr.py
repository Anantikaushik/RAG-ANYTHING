from PIL import Image, ImageDraw

from app.ingestion.ocr.paddle import PaddleOCRProvider


def test_paddle_ocr_extracts_text(tmp_path):
    image_path = tmp_path / "ocr_test.png"

    image = Image.new("RGB", (600, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 70), "RAG Anything OCR Test", fill="black")
    image.save(image_path)

    provider = PaddleOCRProvider()

    text = provider.extract_text(image_path)

    assert isinstance(text, str)
    assert "RAG" in text.upper()