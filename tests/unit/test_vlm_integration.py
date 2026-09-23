from __future__ import annotations

from unittest.mock import Mock

from app.ingestion.normalization.models import (
    NormalizedContent,
    NormalizedDocument,
)
from app.multimodal.analysis.models import MultimodalAnalysis
from app.multimodal.analysis.processor import (
    MultimodalAnalysisProcessor,
)


class FakeVLMAnalyzer:
    supported_type = "image"

    def __init__(self) -> None:
        self.calls = 0

    def supports(self, content) -> bool:
        return content.type == "image"

    def analyze(self, content) -> MultimodalAnalysis:
        self.calls += 1

        return MultimodalAnalysis(
            content_id=content.element_id,
            content_type="image",
            summary="Test image analysis",
            extracted_text="Important information from image",
            metadata={
                "analyzer": "fake-vlm",
            },
        )


def test_vlm_analysis_is_cached() -> None:
    analyzer = FakeVLMAnalyzer()

    processor = MultimodalAnalysisProcessor(
        analyzers=[analyzer]
    )

    content = NormalizedContent(
        element_id="image-001",
        document_id="document-001",
        type="image",
        page_idx=0,
        position=0,
        text=None,
        payload={
            "img_path": "test.png",
        },
        parent_id=None,
        previous_id=None,
        next_id=None,
    )

    first = processor.analyze(content)
    second = processor.analyze(content)

    assert first.content_id == "image-001"
    assert second.content_id == "image-001"

    # The analyzer must only run once.
    assert analyzer.calls == 1


def test_materialize_writes_vlm_output_to_document() -> None:
    analyzer = FakeVLMAnalyzer()

    processor = MultimodalAnalysisProcessor(
        analyzers=[analyzer]
    )

    document = NormalizedDocument(
        document_id="document-001",
        filename="test.png",
        elements=[
            NormalizedContent(
                element_id="image-001",
                document_id="document-001",
                type="image",
                page_idx=0,
                position=0,
                text=None,
                payload={
                    "img_path": "test.png",
                },
                parent_id=None,
                previous_id=None,
                next_id=None,
            )
        ],
        metadata={},
    )

    results = processor.materialize(document)

    element = document.elements[0]

    assert len(results) == 1
    assert element.text == (
        "Important information from image"
    )

    assert element.payload["extracted_text"] == (
        "Important information from image"
    )

    assert element.payload["analysis"]["content_type"] == "image"

    # Still only one analyzer invocation.
    assert analyzer.calls == 1

def test_vlm_image_analyzer_merges_ocr_text() -> None:
    from app.multimodal.analysis.vlm_image import VLMImageAnalyzer

    class FakeVLMProvider:
        model = "fake-vlm"

        def analyze_image(self, image_path, prompt):
            assert image_path == "test.png"
            assert "OCR text" in prompt
            assert "Text detected by OCR" in prompt

            return "Visual description of image"

    class FakeOCRProvider:
        def extract_text(self, image_path):
            assert image_path == "test.png"
            return "Text detected by OCR"

    analyzer = VLMImageAnalyzer(
        vlm_provider=FakeVLMProvider(),
        ocr_provider=FakeOCRProvider(),
    )

    content = NormalizedContent(
        element_id="image-002",
        document_id="document-001",
        type="image",
        page_idx=0,
        position=0,
        text=None,
        payload={
            "img_path": "test.png",
        },
        parent_id=None,
        previous_id=None,
        next_id=None,
    )

    result = analyzer.analyze(content)

    assert result.content_id == "image-002"
    assert result.content_type == "image"

    assert result.summary == "Visual description of image"

    assert "Visual description of image" in result.extracted_text
    assert "Text detected by OCR" in result.extracted_text

    assert result.metadata["ocr_enabled"] is True
    assert result.metadata["ocr_text"] == "Text detected by OCR"    
