from app.ingestion.content import ContentItem
from app.ingestion.pipeline import IngestionPipeline
from app.knowledge_graph.builder import GraphBuilder
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.storage.memory import InMemoryGraphStorage
from app.retrieval.graph import GraphRetriever


def test_multiple_content_lists_share_indexes_and_link_entities() -> None:
    manager = GraphManager(storage=InMemoryGraphStorage())
    pipeline = IngestionPipeline(
        graph_builder=GraphBuilder(graph_manager=manager)
    )

    first = pipeline.ingest_content_list(
        [ContentItem(type="text", text="OpenAI builds Python systems.", page_idx=0)],
        filename="first.txt",
        document_id="doc-first",
    )
    second = pipeline.ingest_content_list(
        [ContentItem(type="text", text="OpenAI researches machine learning.", page_idx=1)],
        filename="second.txt",
        document_id="doc-second",
    )

    assert first.graph is second.graph
    assert first.vector_store is second.vector_store
    assert manager.graph.get_node("doc-first") is not None
    assert manager.graph.get_node("doc-second") is not None
    assert len(first.vector_store) == 2

    openai_nodes = [
        node for node in manager.find_by_type("entity")
        if node.properties.get("name") == "OpenAI"
    ]
    assert len(openai_nodes) == 1
    mentions = [
        edge for edge in manager.graph.get_edges()
        if edge.target_id == openai_nodes[0].node_id
        and edge.relation == "MENTIONS"
    ]
    assert len(mentions) == 2

    retrieved = GraphRetriever(manager).retrieve("machine learning", top_k=5)
    assert any(
        "machine learning" in (result.text or "").lower()
        for result in retrieved.results
    )
