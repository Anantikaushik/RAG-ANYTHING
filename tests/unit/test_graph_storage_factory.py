from app.knowledge_graph.storage.factory import create_graph_storage
from app.knowledge_graph.storage.memory import InMemoryGraphStorage


def test_factory_creates_memory_storage(monkeypatch):
    monkeypatch.setattr(
        "app.knowledge_graph.storage.factory.settings.graph_storage_backend",
        "memory",
    )

    storage = create_graph_storage()

    assert isinstance(storage, InMemoryGraphStorage)