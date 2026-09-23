from pathlib import Path
import sys

from dotenv import load_dotenv

from app.application import RAGAnythingApp
from app.ingestion.pipeline import IngestionPipeline
from app.knowledge_graph.builder import GraphBuilder
from app.knowledge_graph.graph import GraphManager
from app.knowledge_graph.storage.memory import InMemoryGraphStorage
from app.query.providers.groq import GroqGenerationProvider
from app.retrieval.embeddings import SentenceTransformerEmbeddingProvider


def main():
    load_dotenv()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    demo_file = Path("data/uploads/demo.txt")

    demo_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    demo_file.write_text(
        (
            "Python is a high-level programming language. "
            "It is widely used for software development, "
            "data science, machine learning, and automation.\n\n"
            "Machine learning is a field of artificial intelligence "
            "that enables systems to learn patterns from data."
        ),
        encoding="utf-8",
    )

    embedding_provider = SentenceTransformerEmbeddingProvider()

    pipeline = IngestionPipeline(
        graph_builder=GraphBuilder(
            graph_manager=GraphManager(
                storage=InMemoryGraphStorage()
            )
        ),
        embedding_provider=embedding_provider,
    )

    generation_provider = GroqGenerationProvider()

    app = RAGAnythingApp(
        ingestion_pipeline=pipeline,
        generation_provider=generation_provider,
    )

    result = app.ingest(demo_file)

    print("\n--- INGESTION ---")
    print("Document:", result.document.filename)
    print("Content items:", result.document.content_count)
    print("Graph nodes:", len(result.graph.graph.get_nodes()))
    print("Vector records:", len(result.vector_store))

    response = app.ask(
        "What is Python?",
        top_k=3,
    )

    print("\n--- ANSWER ---")
    print(response.answer)

    print("\n--- SOURCES ---")
    for source in response.sources:
        print(f"[{source.citation_id}] {source.label}")

    print("\n--- TOKEN USAGE ---")
    usage = generation_provider.last_usage
    print("Input tokens:", usage["prompt_tokens"])
    print("Output tokens:", usage["completion_tokens"])
    print("Total tokens:", usage["total_tokens"])


if __name__ == "__main__":
    main()