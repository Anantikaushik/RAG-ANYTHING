from __future__ import annotations

from app.ingestion.pipeline import IngestionPipeline
from app.knowledge_graph.builder import GraphBuilder
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.storage.memory import InMemoryGraphStorage
from app.multimodal.analysis.processor import (
    MultimodalAnalysisProcessor,
)


def _memory_graph_builder() -> GraphBuilder:
    return GraphBuilder(
        graph_manager=GraphManager(
            storage=InMemoryGraphStorage()
        )
    )


def test_pipeline_wires_vlm_with_ocr(monkeypatch):
    calls: list[bool] = []
    processor = MultimodalAnalysisProcessor()

    def create_processor(*, use_ocr: bool):
        calls.append(use_ocr)
        return processor

    monkeypatch.setattr(
        "app.ingestion.pipeline.create_vlm_analysis_processor",
        create_processor,
    )

    pipeline = IngestionPipeline(
        graph_builder=_memory_graph_builder(),
        use_vlm=True,
        use_ocr=True,
    )

    assert pipeline.multimodal_analysis_processor is processor
    assert calls == [True]


def test_pipeline_wires_vlm_without_ocr(monkeypatch):
    calls: list[bool] = []
    processor = MultimodalAnalysisProcessor()

    def create_processor(*, use_ocr: bool):
        calls.append(use_ocr)
        return processor

    monkeypatch.setattr(
        "app.ingestion.pipeline.create_vlm_analysis_processor",
        create_processor,
    )

    pipeline = IngestionPipeline(
        graph_builder=_memory_graph_builder(),
        use_vlm=True,
        use_ocr=False,
    )

    assert pipeline.multimodal_analysis_processor is processor
    assert calls == [False]


def test_default_pipeline_does_not_create_vlm_or_ocr(monkeypatch):
    def fail_if_created(*args, **kwargs):
        raise AssertionError(
            "VLM/OCR processor should not be created"
        )

    monkeypatch.setattr(
        "app.ingestion.pipeline.create_vlm_analysis_processor",
        fail_if_created,
    )

    pipeline = IngestionPipeline(
        graph_builder=_memory_graph_builder(),
    )

    assert isinstance(
        pipeline.multimodal_analysis_processor,
        MultimodalAnalysisProcessor,
    )
