from __future__ import annotations

from app.ingestion.hierarchy.models import DocumentHierarchy
from app.ingestion.normalization.models import NormalizedDocument
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.models.node import GraphNode


class GraphBuilder:
    """Build a knowledge graph from normalized document structure."""
    

    def __init__(self, graph_manager=None):
        self.graph_manager = graph_manager or GraphManager()

    def build(
        self,
        document: NormalizedDocument,
        hierarchy: DocumentHierarchy,
    ) -> GraphManager:

        graph = self.graph_manager
        graph.active_document_id = document.document_id

        # ============================================================
        # PHASE 1: CREATE ALL NODES
        # ============================================================

        # Document node
        document_node = GraphNode(
            node_id=document.document_id,
            node_type="document",
            label=document.filename,
            properties={
                "filename": document.filename,
                **document.metadata,
            },
        )

        graph.add_node(document_node)

        # Page nodes
        for page in hierarchy.page_nodes:
            page_node = GraphNode(
                node_id=page.page_id,
                node_type="page",
                label=f"Page {page.page_idx + 1}",
                properties={
                    "page_idx": page.page_idx,
                    **page.metadata,
                },
            )

            graph.add_node(page_node)

        # Content nodes
        for element in document.elements:
            content_node = GraphNode(
                node_id=element.element_id,
                node_type="content",
                label=element.type,
                properties={
                    "type": element.type,
                    "page_idx": element.page_idx,
                    "position": element.position,
                    "text": element.text,
                    **element.payload,
                },
            )

            graph.add_node(content_node)

        # ============================================================
        # PHASE 2: CREATE ALL RELATIONSHIPS
        # ============================================================

        # Document -> Page
        for page in hierarchy.page_nodes:
            graph.add_edge(
                source_id=document.document_id,
                target_id=page.page_id,
                relation="CONTAINS",
            )

        # Document/Page -> Content
        for element in document.elements:

            # Document provenance
            graph.add_edge(
                source_id=document.document_id,
                target_id=element.element_id,
                relation="CONTAINS",
            )

            # Page provenance
            if element.page_idx is not None:

                page_id = (
                    f"{document.document_id}:page:{element.page_idx}"
                )

                if graph.graph.get_node(page_id):
                    graph.add_edge(
                        source_id=page_id,
                        target_id=element.element_id,
                        relation="CONTAINS",
                    )

            # Sequential content relationship
            if element.next_id:
                graph.add_edge(
                    source_id=element.element_id,
                    target_id=element.next_id,
                    relation="NEXT",
                )

        return graph