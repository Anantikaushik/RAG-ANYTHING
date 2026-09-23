from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

from app.ingestion.content_list import ContentList
from app.ingestion.document import Document
from app.ingestion.hierarchy.builder import HierarchyBuilder
from app.ingestion.hierarchy.models import DocumentHierarchy
from app.ingestion.normalization.models import NormalizedDocument
from app.ingestion.normalization.normalizer import ContentNormalizer
from app.ingestion.parser_registry import ParserRegistry
from app.multimodal.processor import MultimodalProcessor
from app.multimodal.analysis.processor import MultimodalAnalysisProcessor

from app.knowledge_graph.builder import GraphBuilder
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.multimodal_processor import MultimodalGraphProcessor
from app.knowledge_graph.semantic_processor import SemanticGraphProcessor
from app.knowledge_graph.storage.factory import create_graph_storage
from app.multimodal.analysis.vlm_processor import create_vlm_analysis_processor
from app.core.config import settings

from app.retrieval.embeddings import (
    EmbeddingProvider,
    DeterministicEmbeddingProvider,
)
from app.retrieval.indexer import DocumentVectorIndexer
from app.retrieval.vector_store import InMemoryVectorStore


class IngestionResult:
    def __init__(
        self,
        document: Document,
        normalized_document: NormalizedDocument,
        hierarchy: DocumentHierarchy,
        graph,
        vector_store,
    ) -> None:
        self.document = document
        self.normalized_document = normalized_document
        self.hierarchy = hierarchy
        self.graph = graph
        self.vector_store = vector_store


class IngestionPipeline:
    def __init__(
        self,
        parser_registry: ParserRegistry | None = None,
        normalizer: ContentNormalizer | None = None,
        hierarchy_builder: HierarchyBuilder | None = None,
        multimodal_processor: MultimodalProcessor | None = None,
        multimodal_analysis_processor: MultimodalAnalysisProcessor | None = None,
        graph_builder: GraphBuilder | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        use_vlm: bool = False,
        use_ocr: bool = True,
    ) -> None:

        # -----------------------------
        # Ingestion components
        # -----------------------------
        self.parser_registry = parser_registry or ParserRegistry()
        self.normalizer = normalizer or ContentNormalizer()
        self.hierarchy_builder = hierarchy_builder or HierarchyBuilder()

        self.multimodal_processor = (
            multimodal_processor or MultimodalProcessor()
        )

        # Shared analysis processor.
        #
        # This same instance is used by both semantic and
        # multimodal graph processing.
        if multimodal_analysis_processor is not None:
            self.multimodal_analysis_processor = (
                multimodal_analysis_processor
            )
        elif use_vlm:
            self.multimodal_analysis_processor = (
                create_vlm_analysis_processor(use_ocr=use_ocr)
            )
        else:
            self.multimodal_analysis_processor = (
                MultimodalAnalysisProcessor()
            )

        # -----------------------------
        # Knowledge graph
        # -----------------------------
        if graph_builder is None:
            graph_manager = GraphManager(
                storage=create_graph_storage()
            )

            self.graph_builder = GraphBuilder(
                graph_manager=graph_manager
            )
        else:
            self.graph_builder = graph_builder

        graph_manager = self.graph_builder.graph_manager
        semantic_flags_from_settings = graph_builder is None

        self.semantic_processor = SemanticGraphProcessor(
            graph=graph_manager,
            multimodal_processor=self.multimodal_analysis_processor,
            enable_entity_extraction=(
                settings.graph_enable_entity_extraction
                if semantic_flags_from_settings
                else True
            ),
            enable_relation_extraction=(
                settings.graph_enable_relation_extraction
                if semantic_flags_from_settings
                else True
            ),
        )

        self.multimodal_graph_processor = MultimodalGraphProcessor(
            graph=graph_manager,
            analysis_processor=self.multimodal_analysis_processor,
        )

        # -----------------------------
        # Vector retrieval
        # -----------------------------
        self.vector_store = InMemoryVectorStore()

        self.embedding_provider = (
            embedding_provider
            or DeterministicEmbeddingProvider()
        )

        self.vector_indexer = DocumentVectorIndexer(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
        )

    # -----------------------------
    # Shared processing
    # -----------------------------
    def _process_document(
        self,
        document: Document,
    ) -> IngestionResult:

        # 1. Normalize
        normalized_document = self.normalizer.normalize(
            document
        )

        # 2. Build hierarchy
        hierarchy = self.hierarchy_builder.build(
            normalized_document
        )

        # 3. Process multimodal content
        self.multimodal_processor.process_document(
            normalized_document
        )

        # 4. Analyze multimodal content
        #
        # Analysis results are materialized back into the
        # normalized elements so downstream graph/vector
        # components can consume them.
        self.multimodal_analysis_processor.materialize(
            normalized_document
        )

        graph_storage = self.graph_builder.graph_manager.graph
        batch = getattr(graph_storage, "batch", None)
        graph_transaction = batch() if batch is not None else nullcontext()

        with graph_transaction:
            # 5. Build structural graph
            graph = self.graph_builder.build(
                normalized_document,
                hierarchy,
            )

            # 6. Extract semantic entities/relationships
            self.semantic_processor.process(
                normalized_document
            )

            # 7. Add multimodal analysis to graph
            self.multimodal_graph_processor.process(
                normalized_document
            )

        # 8. Index searchable content
        self.vector_indexer.index(
            normalized_document
        )

        return IngestionResult(
            document=document,
            normalized_document=normalized_document,
            hierarchy=hierarchy,
            graph=graph,
            vector_store=self.vector_store,
        )

    # -----------------------------
    # File-based ingestion
    # -----------------------------
    def ingest(
        self,
        path: str | Path,
    ) -> IngestionResult:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Expected a file, got: {path}"
            )

        document = self.parser_registry.parse(path)

        return self._process_document(document)

    def ingest_many(
        self,
        paths: list[str | Path],
    ) -> list[IngestionResult]:
        """Ingest several files into this pipeline's shared graph and index.

        The pipeline owns one graph manager and vector store, so processing
        files sequentially deliberately creates a single cross-document
        retrieval surface while each parser result retains its own ID/output.
        """
        if not paths:
            raise ValueError("At least one document path is required.")
        return [self.ingest(path) for path in paths]

    # -----------------------------
    # Direct content-list ingestion
    # -----------------------------
    def ingest_content_list(
        self,
        content_list: ContentList | list,
        filename: str = "content-list",
        document_id: str | None = None,
        metadata: dict | None = None,
    ) -> IngestionResult:

        if isinstance(content_list, ContentList):
            items = list(content_list.items)
        else:
            items = ContentList.from_list(
                content_list
            ).items

        if not items:
            raise ValueError(
                "Content list cannot be empty."
            )

        document_id = document_id or str(uuid4())

        document = Document(
            document_id=document_id,
            source_path="",
            filename=filename,
            content=items,
            metadata=metadata or {},
        )

        return self._process_document(document)