from app.knowledge_graph.extraction.entity import ExtractedEntity
from app.knowledge_graph.extraction.relation import ExtractedRelation
from app.knowledge_graph.enricher import GraphEnricher
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.models.node import GraphNode


def test_add_entity_creates_node_and_provenance_edge():
    graph = GraphManager()

    content = GraphNode(
        node_id="content-001",
        node_type="content",
    )
    graph.add_node(content)

    entity = ExtractedEntity(
        entity_id="entity-001",
        name="Microsoft",
        entity_type="ORGANIZATION",
        source_content_id="content-001",
    )

    enricher = GraphEnricher(graph)
    node = enricher.add_entity(entity)

    assert node.node_id == "entity-001"
    assert node.node_type == "entity"
    assert node.label == "Microsoft"

    assert graph.graph.get_node("entity-001") is not None

    edge = graph.graph.get_edge(
        "content-001",
        "entity-001",
        "MENTIONS",
    )

    assert edge is not None


def test_add_relation_creates_graph_edge():
    graph = GraphManager()

    graph.add_node(
        GraphNode(
            node_id="entity-001",
            node_type="entity",
        )
    )

    graph.add_node(
        GraphNode(
            node_id="entity-002",
            node_type="entity",
        )
    )

    relation = ExtractedRelation(
        relation_id="relation-001",
        source_entity_id="entity-001",
        target_entity_id="entity-002",
        relation="RELATED_TO",
    )

    enricher = GraphEnricher(graph)
    enricher.add_relation(relation)

    edge = graph.graph.get_edge(
        "entity-001",
        "entity-002",
        "RELATED_TO",
    )

    assert edge is not None
    assert edge.properties["relation_id"] == "relation-001"