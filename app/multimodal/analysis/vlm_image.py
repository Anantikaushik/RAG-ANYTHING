from __future__ import annotations

from app.ingestion.ocr.base import OCRProvider
from app.multimodal.analysis.models import MultimodalAnalysis
from app.multimodal.analysis.vlm import VLMProvider


class VLMImageAnalyzer:
    """Analyzes image content using a vision-language model and optional OCR."""

    supported_type = "image"

    def __init__(
        self,
        vlm_provider: VLMProvider,
        ocr_provider: OCRProvider | None = None,
    ) -> None:
        self.vlm_provider = vlm_provider
        self.ocr_provider = ocr_provider

    def supports(self, content) -> bool:
        return getattr(content, "type", None) == self.supported_type

    def analyze(self, content) -> MultimodalAnalysis:
        image_path = (
            content.payload.get("img_path")
            if hasattr(content, "payload")
            else None
        )

        if not image_path:
            raise ValueError("Image content requires 'img_path'.")

        ocr_text = ""
        if self.ocr_provider is not None:
            ocr_text = self.ocr_provider.extract_text(image_path)

        prompt = (
            "Analyze this document image for RAG retrieval. "
            "Extract all important visible information, including "
            "text, objects, diagrams, charts, labels, and relationships. "
            "Return a concise but information-rich description that "
            "can be used to answer questions about the image."
        )

        if ocr_text:
            prompt += (
                "\n\nOCR text extracted from this image is provided below. "
                "Use it to improve textual accuracy, but rely on the image "
                "itself for visual context:\n"
                f"{ocr_text}"
            )

        extracted_text = self.vlm_provider.analyze_image(
            image_path=image_path,
            prompt=prompt,
        )

        combined_text = extracted_text

        if ocr_text:
            combined_text = (
                f"{extracted_text}\n\n"
                f"OCR Text:\n{ocr_text}"
            ).strip()

        return MultimodalAnalysis(
            content_id=content.element_id,
            content_type="image",
            summary=extracted_text,
            extracted_text=combined_text,
            metadata={
                "img_path": image_path,
                "analyzer": "vlm",
                "model": getattr(
                    self.vlm_provider,
                    "model",
                    None,
                ),
                "ocr_enabled": self.ocr_provider is not None,
                "ocr_text": ocr_text,
            },
        )