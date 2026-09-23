from app.ingestion.content import ContentItem
from app.ingestion.content_list import ContentList
from app.ingestion.pipeline import IngestionPipeline
from app.multimodal.analysis.models import MultimodalAnalysis


class FakeVLMAnalyzer:
    supported_type = "image"

    def supports(self, content):
        return content.type == "image"

    def analyze(self, content):
        return MultimodalAnalysis(
            content_id=content.element_id,
            content_type="image",
            summary="A system architecture diagram.",
            extracted_text="The diagram contains an API service and database.",
            entities=["API service", "database"],
            relationships=[
                {
                    "source": "API service",
                    "target": "database",
                    "relation": "CONNECTS_TO",
                }
            ],
            metadata={
                "analyzer": "fake-vlm",
            },
        )


def test_vlm_flows_through_pipeline():
    content = ContentList(
        [
            ContentItem(
                type="text",
                text="The architecture is shown below.",
                page_idx=0,
            ),
            ContentItem(
                type="image",
                img_path="data/uploads/test_image.png",
                image_caption=["Architecture"],
                page_idx=1,
            ),
        ]
    )

    pipeline = IngestionPipeline(
        multimodal_analysis_processor=None,
    )

    # Replace the default processor with a deterministic fake VLM
    # while keeping the real pipeline components.
    pipeline.multimodal_analysis_processor._analyzers = [
        FakeVLMAnalyzer()
    ]

    result = pipeline.ingest_content_list(
        content,
        filename="vlm-pipeline-demo",
        document_id="vlm-pipeline-document",
    )

    image = result.normalized_document.elements[1]

    assert image.type == "image"
    assert image.text == (
        "The diagram contains an API service and database."
    )

    assert image.payload["extracted_text"] == (
        "The diagram contains an API service and database."
    )

    analysis = image.payload["analysis"]

    assert analysis["analyzer"] == "fake-vlm"
    assert analysis["content_type"] == "image"

    graph_node = result.graph.graph.get_node(
        image.element_id
    )

    assert graph_node is not None
    assert graph_node.properties["analyzer"] == "fake-vlm"

    assert len(result.vector_store._items) > 0