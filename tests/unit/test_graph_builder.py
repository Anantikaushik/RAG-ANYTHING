from app.ingestion.content import ContentItem
from app.ingestion.document import Document
from app.ingestion.hierarchy.builder import HierarchyBuilder
from app.ingestion.normalization.normalizer import ContentNormalizer
from app.knowledge_graph.builder import GraphBuilder


def build_document() -> Document:
    document = Document(
        document_id="doc-001",
        source_path="sample.pdf",
        filename="sample.pdf",
    )

    document.add_content(
        ContentItem(
            type="text",
            text="Introduction",
            page_idx=0,
        )
    )

    document.add_content(
        ContentItem(
            type="image",
            img_path="architecture.png",
            page_idx=0,
        )
    )

    document.add_content(
        ContentItem(
            type="table",
            table_body="| A | B |\n|---|---|\n| 1 | 2 |",
            page_idx=1,
        )
    )

    return document


def build_graph():
    document = build_document()

    normalized = ContentNormalizer().normalize(document)

    hierarchy = HierarchyBuilder().build(normalized)

    graph = GraphBuilder().build(
        normalized,
        hierarchy,
    )

    return graph, normalized, hierarchy


def test_graph_contains_document_node():
    graph, _, _ = build_graph()

    node = graph.graph.get_node("doc-001")

    assert node is not None
    assert node.node_type == "document"
    assert node.label == "sample.pdf"


def test_graph_contains_page_nodes():
    graph, _, _ = build_graph()

    pages = graph.find_by_type("page")

    assert len(pages) == 2


def test_graph_contains_content_nodes():
    graph, _, _ = build_graph()

    content = graph.find_by_type("content")

    assert len(content) == 3


def test_document_contains_pages():
    graph, _, _ = build_graph()

    neighbors = graph.get_neighbors("doc-001")

    page_ids = {
        node.node_id
        for node in neighbors
        if node.node_type == "page"
    }

    assert page_ids == {
        "doc-001:page:0",
        "doc-001:page:1",
    }


def test_pages_contain_correct_content():
    graph, normalized, _ = build_graph()

    page_zero = graph.get_neighbors(
        "doc-001:page:0"
    )

    page_zero_ids = {
        node.node_id
        for node in page_zero
    }

    assert page_zero_ids == {
        normalized.elements[0].element_id,
        normalized.elements[1].element_id,
    }


def test_content_has_document_provenance():
    graph, normalized, _ = build_graph()

    first_element = normalized.elements[0]

    neighbors = graph.get_neighbors("doc-001")

    neighbor_ids = {
        node.node_id
        for node in neighbors
    }

    assert first_element.element_id in neighbor_ids


def test_next_relationships_are_created():
    graph, normalized, _ = build_graph()

    first = normalized.elements[0]
    second = normalized.elements[1]

    edges = graph.graph.edges.values()

    next_edges = [
        edge
        for edge in edges
        if edge.relation == "NEXT"
    ]

    assert any(
        edge.source_id == first.element_id
        and edge.target_id == second.element_id
        for edge in next_edges
    )


def test_graph_contains_expected_relationships():
    graph, _, _ = build_graph()

    relations = {
        edge.relation
        for edge in graph.graph.edges.values()
    }

    assert "CONTAINS" in relations
    assert "NEXT" in relations