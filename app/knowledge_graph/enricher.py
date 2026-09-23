from __future__ import annotations

from app.knowledge_graph.extraction.entity import ExtractedEntity
from app.knowledge_graph.extraction.relation import ExtractedRelation
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.models.node import GraphNode


class GraphEnricher:
    """Adds semantic entities and relationships to an existing graph."""

    def __init__(self, graph: GraphManager) -> None:
        self.graph = graph

    def add_entity(self, entity: ExtractedEntity) -> GraphNode:
        node = GraphNode(
            node_id=entity.entity_id,
            node_type="entity",
            label=entity.name,
            properties={
                "name": entity.name,
                "entity_type": entity.entity_type,
                "source_content_id": entity.source_content_id,
                **entity.properties,
            },
        )

        existing_node = self.graph.graph.get_node(
            entity.entity_id
        )

        if existing_node is None:
            self.graph.add_node(node)
        else:
            node = existing_node

        # Preserve provenance:
        # Content -> Entity
        if (
            self.graph.graph.get_edge(
                entity.source_content_id,
                entity.entity_id,
                "MENTIONS",
            )
            is None
        ):
            self.graph.add_edge(
                source_id=entity.source_content_id,
                target_id=entity.entity_id,
                relation="MENTIONS",
            )

        return node

    def add_relation(self, relation: ExtractedRelation) -> None:
        if (
            self.graph.graph.get_edge(
                relation.source_entity_id,
                relation.target_entity_id,
                relation.relation,
            )
            is not None
        ):
            return

        self.graph.add_edge(
            source_id=relation.source_entity_id,
            target_id=relation.target_entity_id,
            relation=relation.relation,
            properties={
                "relation_id": relation.relation_id,
                **relation.properties,
            },
        )

    def enrich(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> GraphManager:
        for entity in entities:
            self.add_entity(entity)

        for relation in relations:
            self.add_relation(relation)

        return self.graph