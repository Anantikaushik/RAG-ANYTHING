from app.ingestion.ocr.paddle import PaddleOCRProvider
from app.multimodal.analysis.deterministic import (
    DeterministicEquationAnalyzer,
    DeterministicTableAnalyzer,
)
from app.multimodal.analysis.groq_vlm import GroqVLMProvider
from app.multimodal.analysis.processor import MultimodalAnalysisProcessor
from app.multimodal.analysis.vlm_image import VLMImageAnalyzer


def create_vlm_analysis_processor(
    use_ocr: bool = True,
) -> MultimodalAnalysisProcessor:
    """Create a multimodal processor using Groq VLM + PaddleOCR."""

    vlm_provider = GroqVLMProvider()
    ocr_provider = PaddleOCRProvider() if use_ocr else None

    analyzers = [
        VLMImageAnalyzer(
            vlm_provider,
            ocr_provider,
        ),
        DeterministicTableAnalyzer(),
        DeterministicEquationAnalyzer(),
    ]

    return MultimodalAnalysisProcessor(
        analyzers=analyzers,
    )