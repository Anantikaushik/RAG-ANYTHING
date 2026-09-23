from pathlib import Path

from dotenv import load_dotenv

from app.ingestion.normalization.models import (
    NormalizedContent,
    NormalizedDocument,
)
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.builder import GraphBuilder
from app.multimodal.analysis.vlm_processor import (
    create_vlm_analysis_processor,
)
from app.knowledge_graph.multimodal_processor import (
    MultimodalGraphProcessor,
)


def main() -> None:
    load_dotenv()

    image_path = Path("data/uploads/test_image.png")

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    document = NormalizedDocument(
        document_id="vlm-graph-test",
        filename=image_path.name,
        elements=[
            NormalizedContent(
                element_id="vlm-image-001",
                document_id="vlm-graph-test",
                type="image",
                page_idx=0,
                position=0,
                text=None,
                payload={
                    "img_path": str(image_path),
                },
                parent_id=None,
                previous_id=None,
                next_id=None,
            )
        ],
        metadata={},
    )

    # -----------------------------------------
    # 1. Create graph
    # -----------------------------------------
    graph = GraphManager()

    graph_builder = GraphBuilder(
        graph_manager=graph
    )

    # -----------------------------------------
    # 2. Build structural graph
    # -----------------------------------------
    from app.ingestion.hierarchy.builder import (
        HierarchyBuilder,
    )

    hierarchy = HierarchyBuilder().build(document)

    graph_builder.build(
        document,
        hierarchy,
    )

    # -----------------------------------------
    # 3. Run VLM analysis
    # -----------------------------------------
    analysis_processor = (
        create_vlm_analysis_processor()
    )

    print("\n--- VLM ANALYSIS ---")

    analysis_processor.materialize(document)

    # -----------------------------------------
    # 4. Add VLM analysis to graph
    # -----------------------------------------
    multimodal_graph_processor = (
        MultimodalGraphProcessor(
            graph=graph,
            analysis_processor=analysis_processor,
        )
    )

    multimodal_graph_processor.process(
        document
    )

    # -----------------------------------------
    # 5. Inspect graph node
    # -----------------------------------------
    node = graph.graph.get_node(
        "vlm-image-001"
    )

    if node is None:
        raise RuntimeError(
            "Image graph node was not created."
        )

    print("\n--- GRAPH NODE ---")
    print(f"Node ID: {node.node_id}")
    print(f"Node Type: {node.node_type}")
    print(f"Label: {node.label}")

    print("\n--- NODE PROPERTIES ---")

    for key, value in node.properties.items():
        print(f"{key}: {value}")

    # -----------------------------------------
    # 6. Validate VLM data
    # -----------------------------------------
    if "summary" not in node.properties:
        raise AssertionError(
            "VLM summary was not stored in graph."
        )

    if "extracted_text" not in node.properties:
        raise AssertionError(
            "VLM extracted text was not stored in graph."
        )

    print("\n--- SUCCESS ---")
    print(
        "Image -> VLM -> Analysis -> Knowledge Graph"
    )


if __name__ == "__main__":
    main()