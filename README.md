# RAG-ANYTHING

### Multimodal, Graph-Grounded Retrieval-Augmented Generation for Multi-Document Knowledge Bases

RAG-ANYTHING is a **clean-room implementation of a multimodal RAG system inspired by the RAG-Anything architecture**.

The project is designed to process heterogeneous documents, preserve their structure and provenance, extract multimodal information, build a Knowledge Graph, create vector indexes, perform hybrid retrieval, and generate grounded answers with an LLM.

Instead of treating a document as plain text, the system preserves **text, images, tables, equations, charts, page structure, entities, and relationships** throughout the pipeline.

---

## 🚀 Overview

Traditional RAG systems often follow:

```text
Document → Chunking → Embeddings → Vector Search → LLM

RAG-ANYTHING extends this approach:

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
✨ Key Features
📄 Unified Multimodal Document Parsing

MinerU is used as the primary document parsing engine.

The ingestion layer supports MinerU-compatible formats including:

PDF
DOC / DOCX
PPT / PPTX
XLS / XLSX
CSV / TSV
HTML
EPUB
OFD
RTF
ODT / ODS / ODP
PNG / JPG / JPEG
WEBP / BMP

MinerU extracts structured content such as:

Text
Images
Tables
Equations
Charts
Lists
Captions
Footnotes
Page-level metadata

The raw MinerU output is converted into application-owned models such as ContentItem, ContentList, and Document.

🧩 Multimodal Understanding

The system preserves different content modalities instead of flattening everything into plain text.

Text
 ├── Documents
 ├── Pages
 └── Sections

Images
Tables
Equations
Charts

Optional VLM-based analysis can further enrich image content with semantic descriptions and extracted information.

🕸️ Knowledge Graph

RAG-ANYTHING builds a graph representation of the ingested knowledge.

At the structural level:

Document
   │
 CONTAINS
   ▼
 Page
   │
 CONTAINS
   ▼
 Content
   │
 MENTIONS
   ▼
 Entity
   │
RELATED_TO
   ▼
 Entity

The Knowledge Graph can represent:

Documents
Pages
Sections
Content
Entities
Relationships
Multimodal content
Provenance metadata

The graph can use an in-memory backend for development or Neo4j for persistent graph storage.

🔗 Cross-Document Knowledge

Multiple documents can be processed into the same knowledge base.

For example:

                    ┌─────────────┐
                    │    Entity   │
                    │    Apple    │
                    └──────┬──────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       Report.pdf     Research.pdf    Slides.pptx
             │             │             │
          Revenue          AI          iPhone

This allows the system to discover relationships across documents rather than treating every file as an isolated RAG dataset.

🔎 Vector Retrieval

Relevant content is transformed into embeddings and stored in a vector index.

User Query
    │
    ▼
Query Embedding
    │
    ▼
Vector Search
    │
    ▼
Semantically Relevant Content

The retrieval layer supports configurable embedding providers, including sentence-transformer-based embeddings.

🔀 Hybrid Retrieval

The system combines two complementary retrieval strategies:

Vector Retrieval

Finds semantically similar content.

Graph Retrieval

Finds connected entities, relationships, and structurally relevant content.

Together:

Graph Retrieval
       +
Vector Retrieval
       ↓
Hybrid Retrieval
       ↓
Relevant Context

This provides both semantic similarity and relationship-aware context.

🤖 LLM-Powered Question Answering

After retrieval, the system assembles the relevant context and passes it to a generation provider.

User Question
      │
      ▼
 Query Processing
      │
      ▼
Graph + Vector Retrieval
      │
      ▼
 Context Assembly
      │
      ▼
      LLM
      │
      ▼
Final Answer

The project includes a configurable generation layer with Groq support.

📚 Multi-Document RAG

RAG-ANYTHING supports uploading multiple files in one ingestion workflow.

File A ─┐
File B ─┤
File C ─┼──► MinerU ──► Shared Knowledge Base
File D ─┘

Each document maintains its own:

document_id
source file
parsed output
page information
provenance

while contributing to the shared:

Knowledge Graph
Vector Index
Retrieval Layer

This enables queries such as:

Compare the findings across these documents.

Which documents mention this entity?

What information is shared across the reports?

How are these entities related across the uploaded files?
🧠 Provenance and Source Tracking

A major design goal is preserving the origin of information throughout the pipeline.

Document
   ↓
Page
   ↓
Content
   ↓
Entity / Retrieved Chunk

This makes retrieved information traceable back to the original document and page.

🗂️ Storage Architecture

Large processing artifacts are kept outside the source repository.

Example:

D:\RAG_ANYTHING_DATA\
│
├── uploads\
│   └── <document_id>\
│       └── source\
│
├── outputs\
│   └── <document_id>\
│       ├── MinerU output
│       └── extracted assets
│
└── mineru\
    ├── models\
    ├── cache\
    └── temp\

The project separates:

Original documents
MinerU-generated outputs
Extracted assets
Model/cache data

This keeps the Git repository lightweight and avoids storing large generated artifacts in source control.

🏗️ System Architecture
                         ┌────────────────────┐
                         │   User Documents   │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │      MinerU        │
                         │  Unified Parser    │
                         └─────────┬──────────┘
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │ Structured Multimodal    │
                      │ Content                  │
                      └────────────┬─────────────┘
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │ ContentItem / ContentList│
                      └────────────┬─────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
             Normalization                   Hierarchy
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
            Knowledge Graph                 Vector Index
            ────────────────                ────────────
            Entities                        Embeddings
            Relations                           │
            Provenance                          │
                    │                           │
                    └────────────┬──────────────┘
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
📁 Project Structure
RAG_ANYTHING/
│
├── app/
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── ingestion/
│   │   ├── content.py
│   │   ├── content_list.py
│   │   ├── document.py
│   │   ├── parser_registry.py
│   │   ├── pipeline.py
│   │   │
│   │   ├── parsers/
│   │   │   ├── base.py
│   │   │   └── mineru.py
│   │   │
│   │   ├── normalization/
│   │   ├── hierarchy/
│   │   └── ocr/
│   │
│   ├── multimodal/
│   │   ├── analysis/
│   │   ├── extractors/
│   │   └── processor.py
│   │
│   ├── knowledge_graph/
│   │   ├── extraction/
│   │   ├── models/
│   │   ├── storage/
│   │   ├── builder.py
│   │   ├── enricher.py
│   │   └── semantic_processor.py
│   │
│   ├── retrieval/
│   │   ├── embeddings/
│   │   ├── graph.py
│   │   ├── vector.py
│   │   ├── hybrid.py
│   │   ├── traversal.py
│   │   ├── indexer.py
│   │   └── vector_store.py
│   │
│   ├── query/
│   │   ├── context.py
│   │   ├── answer.py
│   │   ├── service.py
│   │   └── providers/
│   │
│   └── ui/
│       └── main.py
│
├── config/
│
├── scripts/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── .env.example
├── .gitignore
├── pytest.ini
└── README.md
🛠️ Tech Stack
Layer	Technology
Language	Python
UI	Streamlit
Document Parsing	MinerU
Data Models	Pydantic
Knowledge Graph	Custom Graph Layer + Neo4j
Embeddings	Sentence Transformers / Configurable Providers
Retrieval	Vector + Graph + Hybrid
LLM	Groq / Configurable Provider
Multimodal Analysis	VLM-compatible architecture
Testing	Pytest
Configuration	Python-dotenv
⚙️ Installation

Clone the repository:

git clone <YOUR_REPOSITORY_URL>
cd RAG_ANYTHING

Create a virtual environment:

python -m venv .venv

Activate it:

.\.venv\Scripts\Activate.ps1

Create your environment configuration:

Copy-Item .env.example .env

Configure the required environment variables in .env.

Never commit .env, API keys, model files, or generated document data.

Install the project's required dependencies inside the virtual environment.

▶️ Run the Application

Start the Streamlit application:

streamlit run app/ui/main.py

Typical workflow:

Upload Documents
      ↓
MinerU Parsing
      ↓
Content Conversion
      ↓
Normalization
      ↓
Knowledge Graph + Vector Index
      ↓
Hybrid Retrieval
      ↓
LLM
      ↓
Answer
🧪 Testing

Run the complete suite:

python -m pytest -q

Run unit tests:

python -m pytest tests/unit -q

Run integration tests:

python -m pytest tests/integration -q

Run MinerU parser tests:

python -m pytest tests/unit/test_mineru_parser.py -q
💬 Example Queries

The system can be used for questions such as:

What are the major findings in this document?

What entities are related to Project Nova?

What does the table on page 7 show?

Compare the findings across the uploaded reports.

Which documents mention this organization?

What relationships exist between Entity A and Entity B?

Summarize all information about this topic across the document collection.
🔐 Configuration Areas

The application supports configuration for:

MinerU
Output directory
Storage directory
Model/cache location
Temporary files
Parsing tier
Parsing mode
Knowledge Graph
Storage backend
Neo4j connection
Database
Entity extraction
Relationship extraction
LLM
Groq API configuration
Generation provider
Multimodal Processing
Image processing
VLM integration
Optional OCR/analyzer components
🧱 Design Principles
1. Separation of Responsibilities

Parsing, normalization, graph construction, retrieval, and generation remain separate layers.

2. Application-Owned Content Models

MinerU is responsible for parsing; the application maintains its own internal representation.

3. Multimodal by Design

Images, tables, equations, charts, and text remain first-class content types.

4. Shared Multi-Document Knowledge Base

Multiple documents contribute to a unified retrieval and graph environment.

5. Provenance Preservation

Document and page information is retained throughout ingestion and retrieval.

6. Extensibility

Graph backends, embedding providers, LLM providers, and multimodal components can be extended independently.

7. Clean-Room Implementation

The project implements the architecture independently and does not depend on the official raganything Python package.

🔬 What This Project Demonstrates

This project brings together several important Data Science and Generative AI concepts:

Multimodal document processing
Document intelligence
Retrieval-Augmented Generation
Knowledge Graph construction
Entity and relationship extraction
Vector embeddings
Semantic search
Graph retrieval
Hybrid retrieval
LLM-based answer generation
Multi-document reasoning
Source/provenance tracking
Modular AI system design
End-to-end testing
🗺️ Future Improvements

Potential future extensions include:

Persistent document history
Background/asynchronous document processing
Advanced Knowledge Graph visualization
Retrieval reranking
More multimodal VLM providers
Improved entity resolution
Streaming responses
Authentication and authorization
Production document-serving endpoints
Scalable vector database integration
Large-scale graph traversal
👩‍💻 Author

Anantika Kaushik

B.Tech Information Technology
Generative AI • Data Science • Multimodal AI

📌 Project Philosophy

RAG-ANYTHING is built around a simple idea:

Documents contain more than text.

A useful RAG system should preserve the structure, relationships, multimodal information, and provenance contained within those documents.

This project explores that idea by combining:

Multimodal Parsing
        +
Knowledge Graphs
        +
Vector Retrieval
        +
Hybrid Search
        +
LLM Generation

to create a unified, extensible, multimodal RAG pipeline.


This is the version I would put on GitHub: it presents the project as an actual **AI system architecture**, rather than just a collection of Python modules.
