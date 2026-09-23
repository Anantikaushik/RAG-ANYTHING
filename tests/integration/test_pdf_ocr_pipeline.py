from __future__ import annotations

from pathlib import Path

import fitz

from app.ingestion.pipeline import IngestionPipeline
from app.knowledge_graph.builder import GraphBuilder
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.storage.memory import InMemoryGraphStorage
from app.multimodal.analysis.processor import (
    MultimodalAnalysisProcessor,
)
from app.multimodal.analysis.vlm import VLMProvider
from app.multimodal.analysis.vlm_image import VLMImageAnalyzer
from app.ingestion.ocr.base import OCRProvider


class FakeOCRProvider(OCRProvider):
    def extract_text(self, image_path: str | Path) -> str:
        return "OpenAI API"


class FakeVLMProvider(VLMProvider):
    model = "fake-vlm"

    def analyze_image(self, image_path: str, prompt: str) -> str:
        assert "OpenAI API" in prompt
        return "The diagram shows an OpenAI API service."


def _create_image_pdf(path: Path) -> None:
    pdf = fitz.open()
    page = pdf.new_page()
    image = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40))
    image.clear_with(255)
    page.insert_image(
        fitz.Rect(72, 72, 112, 112),
        stream=image.tobytes("png"),
    )
    pdf.save(path)
    pdf.close()


def test_pdf_image_ocr_reaches_graph_and_vector_index(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "ocr-pipeline.pdf"
    _create_image_pdf(pdf_path)

    analyzer = VLMImageAnalyzer(
        FakeVLMProvider(),
        FakeOCRProvider(),
    )
    processor = MultimodalAnalysisProcessor(
        analyzers=[analyzer],
    )

    monkeypatch.setattr(
        "app.ingestion.pipeline.create_vlm_analysis_processor",
        lambda *, use_ocr: processor,
    )

    pipeline = IngestionPipeline(
        graph_builder=GraphBuilder(
            graph_manager=GraphManager(
                storage=InMemoryGraphStorage()
            )
        ),
        use_vlm=True,
        use_ocr=True,
    )

    result = pipeline.ingest(pdf_path)

    image = next(
        element
        for element in result.normalized_document.elements
        if element.type == "image"
    )

    assert "OpenAI API" in image.text
    assert image.payload["extracted_text"].endswith(
        "OCR Text:\nOpenAI API"
    )

    record = result.vector_store.get(image.element_id)
    assert record is not None
    assert "OpenAI API" in record.text

    entity_names = {
        node.properties.get("name")
        for node in result.graph.find_by_type("entity")
    }
    assert "OpenAI API" in entity_names

    graph_node = result.graph.graph.get_node(image.element_id)
    assert graph_node is not None
    assert graph_node.properties["ocr_text"] == "OpenAI API"
