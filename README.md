# RAG-ANYTHING

### Multimodal, Graph-Grounded Retrieval-Augmented Generation for Multi-Document Knowledge Bases

RAG-ANYTHING is a **clean-room implementation of a multimodal RAG system inspired by the RAG-Anything architecture**.

The project is designed to process heterogeneous documents, preserve their structure and provenance, extract multimodal information, build a Knowledge Graph, create vector indexes, perform hybrid retrieval, and generate grounded answers with an LLM.

Instead of treating a document as plain text, the system preserves **text, images, tables, equations, charts, page structure, entities, and relationships** throughout the pipeline.

---

## 🚀 Overview

Traditional RAG systems often follow:

```text

Documents
     │
     ▼
   MinerU
     │
     ▼
Structured Multimodal Content
     │
     ▼
Normalization + Hierarchy
     │
     ├──────────────────────┐
     ▼                      ▼
Knowledge Graph        Vector Index
     │                      │
Entities                Embeddings
Relations                  │
     │                      │
     └──────────┬───────────┘
                ▼
        Hybrid Retrieval
                │
                ▼
        Context Assembly
                │
                ▼
               LLM
                │
                ▼
       Grounded Answer
       + Source Context
Document → Chunking → Embeddings → Vector Search → LLM
