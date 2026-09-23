import re
import logging
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from pathlib import Path
from threading import Thread
from urllib.parse import quote
from datetime import datetime, timezone

import streamlit as st

# Make the repository package importable when Streamlit runs this nested entrypoint.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.application import RAGAnythingApp
from app.ingestion.pipeline import IngestionPipeline
from app.query.providers.groq import GroqGenerationProvider

logger = logging.getLogger(__name__)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Anything",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def _start_pdf_server(directory: str) -> str:
    """Serve local PDFs over HTTP so browser citation links are usable."""
    port = int(os.getenv("RAG_PDF_SERVER_PORT", "8765"))
    handler = partial(SimpleHTTPRequestHandler, directory=directory)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}"

# ============================================================
# APPLICATION STATE
# ============================================================

if "rag_app" not in st.session_state:
    st.session_state.rag_app = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

if "ingestion_result" not in st.session_state:
    st.session_state.ingestion_result = None

if "ingestion_results" not in st.session_state:
    st.session_state.ingestion_results = []

if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None

if "processing_stage" not in st.session_state:
    st.session_state.processing_stage = "Ready"

if "processing_error" not in st.session_state:
    st.session_state.processing_error = None

if "document_history" not in st.session_state:
    st.session_state.document_history = []


def _friendly_error(exc: Exception) -> str:
    """Convert expected operational failures into actionable UI text."""
    message = str(exc)
    exception_types: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        exception_types.append(type(current).__name__.lower())
        current = current.__cause__ or current.__context__
    lowered = " ".join([message.lower(), *exception_types])

    if "groq_api_key" in lowered:
        return "Answer generation is not configured. Add GROQ_API_KEY to .env and restart the app."
    if (
        "not enough disk space" in lowered
        or "no space left" in lowered
        or "os error 112" in lowered
    ):
        return "MinerU needs more free disk space for its models. Free at least 1 GB on the drive containing the MinerU cache, then retry."
    if (
        "neo4j" in lowered
        or "connection" in lowered
        or "connectivity" in lowered
        or "routing" in lowered
    ):
        return (
            "Neo4j could not be reached at the configured address. "
            "Start Neo4j and confirm NEO4J_URI, NEO4J_DATABASE, and credentials, "
            "or set GRAPH_STORAGE_BACKEND=memory in .env for local processing, "
            "then restart Streamlit."
        )
    if "model" in lowered and ("not found" in lowered or "404" in lowered):
        return "The configured AI model is unavailable. Check GROQ_VLM_MODEL or the generation model configuration."
    if "api" in lowered or "rate_limit" in lowered or "429" in lowered:
        return "The AI provider is unavailable right now. Check your API key, quota, and network connection."
    if "context" in lowered or "token" in lowered or "too large" in lowered:
        return "The retrieved context is too large for the configured model. Try a more specific question or reduce the number of indexed documents."
    if "unsupported" in lowered or "extension" in lowered:
        return "This file type is not supported. Upload a PDF or TXT document."
    if "pdf" in lowered or "parser" in lowered:
        return "The document could not be parsed. Confirm it is a readable, non-corrupt PDF or TXT file."
    return "The operation failed. Check the application logs for technical details and try again."


def _graph_dot(graph, max_nodes: int = 100) -> str:
    """Create a bounded Graphviz view from the existing graph storage."""
    if graph.active_document_id is not None:
        nodes = graph.get_document_nodes(graph.active_document_id)
        edges = graph.get_document_edges(graph.active_document_id)
    else:
        nodes = graph.graph.get_nodes()
        edges = graph.graph.get_edges()
    selected = nodes[:max_nodes]
    selected_ids = {node.node_id for node in selected}
    palette = {
        "document": "#69d391",
        "page": "#9be7b2",
        "content": "#c9f5d5",
        "entity": "#f3c969",
        "section": "#83c8e8",
    }
    lines = ["digraph G {", "rankdir=LR;", "bgcolor=\"transparent\";"]

    for node in selected:
        label = escape(
            f"{node.label or node.node_type}\\n{node.node_type}",
            quote=True,
        ).replace("\\n", "\\\\n")
        color = palette.get(node.node_type, "#d8eee0")
        lines.append(
            f'"{escape(node.node_id, quote=True)}" '
            f'[label="{label}", style="filled", fillcolor="{color}", '
            'fontname="Arial", fontsize=10];'
        )

    for edge in edges:
        if edge.source_id in selected_ids and edge.target_id in selected_ids:
            lines.append(
                f'"{escape(edge.source_id, quote=True)}" -> '
                f'"{escape(edge.target_id, quote=True)}" '
                f'[label="{escape(edge.relation, quote=True)}", '
                'fontname="Arial", fontsize=8];'
            )

    lines.append("}")
    return "\n".join(lines)


def _render_documents_page() -> None:
    st.title("Documents")
    st.caption("Processed sources, indexing status, and document metadata.")
    history = st.session_state.document_history

    if not history:
        st.info("No documents have been processed in this session yet.")
        return

    for document in reversed(history):
        with st.container(border=True):
            title_col, status_col = st.columns([3, 1])
            with title_col:
                st.subheader(document["filename"])
                st.caption(f"Processed {document['processed_at']}")
            with status_col:
                st.success(document["status"])

            metric_cols = st.columns(4)
            metric_cols[0].metric("Pages", document["pages"])
            metric_cols[1].metric("Content", document["content_count"])
            metric_cols[2].metric("Entities", document["entity_count"])
            metric_cols[3].metric("Indexed", document["indexed_count"])

            with st.expander("Document metadata"):
                st.json(document["metadata"])


def _render_graph_page() -> None:
    st.title("Knowledge Graph")
    results = st.session_state.get("ingestion_results", [])
    result = st.session_state.ingestion_result
    if result is None or not results:
        st.info("Process a document first to explore its knowledge graph.")
        return

    graph = result.graph
    document_ids = [item.document.document_id for item in results]
    selected_document = st.selectbox(
        "Graph scope",
        ["All documents", *[
            item.document.filename for item in results
        ]],
    )
    if selected_document == "All documents":
        nodes = graph.graph.get_nodes()
        edges = graph.graph.get_edges()
    else:
        scoped = next(
            item for item in results
            if item.document.filename == selected_document
        )
        nodes = graph.get_document_nodes(scoped.document.document_id)
        edges = graph.get_document_edges(scoped.document.document_id)
    st.caption("A bounded visualization is shown for responsiveness; all graph data remains available below.")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Nodes", len(nodes))
    metric_cols[1].metric("Relationships", len(edges))
    metric_cols[2].metric(
        "Entities",
        sum(1 for node in nodes if node.node_type == "entity"),
    )

    previous_active = graph.active_document_id
    graph.active_document_id = (
        None if selected_document == "All documents"
        else next(
            item.document.document_id for item in results
            if item.document.filename == selected_document
        )
    )
    st.graphviz_chart(_graph_dot(graph))
    graph.active_document_id = previous_active

    node_types = sorted({node.node_type for node in nodes})
    selected_type = st.selectbox("Filter node details", ["All", *node_types])
    visible_nodes = [
        node for node in nodes
        if selected_type == "All" or node.node_type == selected_type
    ]
    st.dataframe(
        [
            {
                "id": node.node_id,
                "type": node.node_type,
                "label": node.label,
                "properties": node.properties,
            }
            for node in visible_nodes
        ],
        width="stretch",
        hide_index=True,
    )

    with st.expander("Relationship details"):
        st.dataframe(
            [
                {
                    "source": edge.source_id,
                    "relation": edge.relation,
                    "target": edge.target_id,
                    "properties": edge.properties,
                }
                for edge in edges
            ],
            width="stretch",
            hide_index=True,
        )

def _citation_url(
    citation: dict,
    document_path: str | dict[str, str],
) -> str:
    if isinstance(document_path, dict):
        document_path = document_path.get(citation.get("document_id"))
    if not document_path:
        return "#"

    resolved_path = Path(document_path).resolve()
    pdf_base_url = _start_pdf_server(str(resolved_path.parent))
    url = f"{pdf_base_url}/{quote(resolved_path.name)}"
    page_idx = citation.get("page_idx")
    return f"{url}#page={int(page_idx) + 1}" if page_idx is not None else url


def citation_markdown(
    answer: str,
    citations: list[dict],
    document_path: str | dict[str, str] | None = None,
) -> str:
    """Turn citation markers into links that open the cited PDF page."""
    citations = _referenced_citations(answer, citations)

    if not citations:
        return answer

    citation_urls = {
        citation["citation_id"]: _citation_url(
            citation,
            document_path,
        )
        for citation in citations
    }
    answer = re.sub(
        r"\[\[(\d+)\]\]\([^)]*\)",
        r"[\1]",
        answer,
    )
    answer = re.sub(
        r"\[(\d+)\]\([^)]*\)",
        r"[\1]",
        answer,
    )
    answer = re.sub(
        r"【(\d+)】",
        r"[\1]",
        answer,
    )
    answer = re.sub(
        r"\[Context\s+(\d+)\]",
        r"[\1]",
        answer,
        flags=re.IGNORECASE,
    )

    linked_answer = re.sub(
        r"\[(\d+)\]",
        lambda match: (
            f'<a href="{citation_urls[int(match.group(1))]}" '
            'target="_blank" rel="noopener" '
            f'class="inline-citation">[{match.group(1)}]</a>'
            if any(
                citation["citation_id"] == int(match.group(1))
                for citation in citations
            )
            else match.group(0)
        ),
        answer,
    )

    return linked_answer


def _used_citation_ids(
    answer: str,
    citations: list[dict],
) -> set[int]:
    """Return only citation IDs explicitly referenced by the answer."""
    available_ids = {
        int(citation["citation_id"])
        for citation in citations
    }
    marker_ids = {
        int(value)
        for pattern in (
            r"\[\[?(\d+)\]?\]",
            r"【(\d+)】",
            r"\[Context\s+(\d+)\]",
        )
        for value in re.findall(pattern, answer, flags=re.IGNORECASE)
    }
    return marker_ids & available_ids


def _referenced_citations(
    answer: str,
    citations: list[dict],
) -> list[dict]:
    used_citation_ids = _used_citation_ids(answer, citations)
    return [
        citation
        for citation in citations
        if citation["citation_id"] in used_citation_ids
    ]


def _citation_payloads(
    citations,
    ingestion_result,
) -> list[dict]:
    """Ensure every citation has the page index of its indexed element."""
    if ingestion_result is None:
        return [citation.model_dump() for citation in citations]

    ingestion_results = (
        ingestion_result
        if isinstance(ingestion_result, list)
        else [ingestion_result]
    )
    page_by_item_id = {
        element.element_id: element.page_idx
        for item in ingestion_results
        for element in item.normalized_document.elements
    }
    graph = ingestion_results[-1].graph
    all_graph_nodes = graph.graph.get_nodes()
    for node in all_graph_nodes:
        node_page_idx = node.properties.get("page_idx")
        if node_page_idx is not None:
            page_by_item_id[node.node_id] = node_page_idx
        source_content_id = node.properties.get("source_content_id")
        if source_content_id:
            source_page_idx = page_by_item_id.get(source_content_id)
            if source_page_idx is not None:
                page_by_item_id[node.node_id] = source_page_idx
    payloads = []

    for citation in citations:
        payload = (
            citation.model_dump()
            if hasattr(citation, "model_dump")
            else dict(citation)
        )
        if payload.get("page_idx") is None:
            payload["page_idx"] = page_by_item_id.get(payload.get("item_id"))
        if payload.get("page_idx") is None:
            page_match = re.search(
                r"\bPage\s+(\d+)\b",
                str(payload.get("label", "")),
                flags=re.IGNORECASE,
            )
            if page_match:
                payload["page_idx"] = int(page_match.group(1)) - 1
        payloads.append(payload)

    return payloads


def _render_source_details(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander("Sources used for this answer"):
        st.dataframe(
            [
                {
                    "citation": f"[{citation['citation_id']}]",
                    "document": citation.get("filename") or "Unknown",
                    "page": (
                        int(citation["page_idx"]) + 1
                        if citation.get("page_idx") is not None
                        else "Unknown"
                    ),
                    "content type": citation.get("content_type") or "Unknown",
                    "source": citation.get("label") or citation.get("item_id"),
                }
                for citation in citations
            ],
            width="stretch",
            hide_index=True,
        )


# ============================================================
# GLOBAL CSS
# ============================================================

st.html("""
<style>
    :root {
        --bg-0: #081a13;
        --bg-1: #0d281d;
        --panel-0: rgba(10, 31, 23, 0.9);
        --panel-1: rgba(15, 46, 35, 0.92);
        --panel-2: rgba(21, 59, 45, 0.97);
        --line: rgba(141, 219, 172, 0.28);
        --line-soft: rgba(141, 219, 172, 0.15);
        --text: #ecfff3;
        --muted: #a9c7b5;
        --accent: #83e4b0;
        --accent-strong: #4bc67d;
        --accent-deep: #2ba861;
        --blue-soft: #a3c8ff;
    }

    .stApp {
        background:
            radial-gradient(circle at 72% 4%, rgba(105, 211, 145, 0.18), transparent 26%),
            linear-gradient(145deg, var(--bg-0) 0%, var(--bg-1) 52%, #0d1d18 100%);
        color: var(--text);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.28;
        background-image:
            linear-gradient(rgba(131, 228, 176, 0.045) 1px, transparent 1px),
            linear-gradient(90deg, rgba(131, 228, 176, 0.045) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: linear-gradient(to bottom, black, transparent 78%);
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .top-command-bar {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.4rem 0.75rem;
        margin-bottom: 1.1rem;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: rgba(8, 21, 16, 0.82);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
    }

    .top-brand {
        color: var(--text);
        font-size: 0.9rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        white-space: nowrap;
        padding: 0 0.55rem;
    }

    .top-command-bar .stButton {
        margin: 0;
    }

    .top-command-bar .stButton > button {
        min-height: 2.2rem;
        padding: 0.35rem 0.75rem;
        border: 1px solid transparent;
        background: rgba(34, 74, 58, 0.5);
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 700;
        transition: all 180ms ease;
    }

    .top-command-bar .stButton > button:hover,
    .top-command-bar .stButton > button[kind="primary"] {
        border-color: rgba(141, 219, 172, 0.4);
        background: linear-gradient(135deg, var(--accent-deep), var(--accent-strong));
        color: #071b12;
    }

    .main-uploader [data-testid="stFileUploader"] {
        padding: 0;
    }

    .main-uploader [data-testid="stFileUploaderDropzone"] {
        min-height: 140px;
        padding: 1rem;
        border: 1px dashed rgba(131, 228, 176, 0.8);
        border-radius: 14px;
        background: radial-gradient(circle at center, rgba(76, 198, 125, 0.12), transparent 60%), rgba(9, 29, 22, 0.8);
        box-shadow: inset 0 0 0 1px rgba(131, 228, 176, 0.08);
    }

    .main-uploader [data-testid="stFileUploaderDropzoneInstructions"] {
        padding: 0;
    }

    .main-uploader [data-testid="stFileUploaderDropzoneInstructions"] > div {
        font-size: 0.72rem;
        color: var(--muted);
    }

    .main-uploader small {
        color: var(--muted);
    }

    .hero {
        position: relative;
        overflow: hidden;
        min-height: 270px;
        border-radius: 16px;
        padding: 2.1rem;
        background:
            radial-gradient(circle at 82% 50%, rgba(131, 228, 176, 0.26), transparent 30%),
            linear-gradient(120deg, #123b2a 0%, #0b2a1d 48%, #071b14 100%);
        border: 1px solid var(--line);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.22);
        margin-bottom: 1.1rem;
    }

    .hero::before {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        right: 7%;
        top: -100px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(131, 228, 176, 0.38), transparent 68%);
        filter: blur(4px);
        animation: drift 8s ease-in-out infinite alternate;
    }

    .hero::after {
        content: "✦  ◌  ✧";
        position: absolute;
        right: 8%;
        bottom: 28px;
        color: rgba(182, 246, 206, 0.42);
        font-size: 1.6rem;
        letter-spacing: 1.4rem;
        transform: rotate(-12deg);
    }

    @keyframes drift {
        from { transform: translate3d(0, 0, 0) scale(1); }
        to { transform: translate3d(-28px, 22px, 0) scale(1.12); }
    }

    .hero-content {
        position: relative;
        z-index: 1;
    }

    .hero-kicker,
    .section-kicker,
    .metric-label,
    .topbar-eyebrow {
        color: var(--accent);
        font-size: 0.62rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .ai-badge {
        display: inline-block;
        padding: 0.42rem 0.8rem;
        border-radius: 20px;
        background: rgba(105, 211, 145, 0.16);
        border: 1px solid rgba(155, 231, 178, 0.32);
        color: #d6ffe4;
        font-size: 0.74rem;
        margin-bottom: 0.85rem;
    }

    .hero-title {
        font-size: clamp(2.3rem, 4vw, 3.2rem);
        line-height: 0.98;
        font-weight: 800;
        color: white;
        letter-spacing: -0.04em;
    }

    .hero-title span {
        color: var(--accent);
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #dfeee4;
        max-width: 680px;
        margin-top: 1rem;
        line-height: 1.6;
    }

    .topbar {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        margin: 0.2rem 0 1.1rem;
        gap: 1rem;
    }

    .topbar-title {
        color: #f5f8ff;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 0.24rem;
    }

    .topbar-status,
    .signal-footer {
        color: #c6dfd1;
        font-size: 0.6rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .pulse-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        margin-right: 0.35rem;
        border-radius: 50%;
        background: var(--accent-strong);
        box-shadow: 0 0 0 4px rgba(105, 211, 145, 0.12), 0 0 14px var(--accent-strong);
        vertical-align: middle;
    }

    .metric-strip {
        display: grid;
        grid-template-columns: 1.25fr repeat(3, minmax(0, 1fr));
        gap: 0.55rem;
        margin: -0.15rem 0 1.1rem;
    }

    .metric-highlight {
        min-height: 88px;
        padding: 0.85rem 0.95rem;
        border: 1px solid var(--line-soft);
        border-radius: 12px;
        background: rgba(13, 35, 27, 0.7);
    }

    .metric-highlight strong {
        display: block;
        color: #f5f8ff;
        font-size: 1.35rem;
        line-height: 1.15;
        margin-top: 0.45rem;
    }

    .metric-highlight small {
        color: #a8bdd0;
        display: block;
        font-size: 0.6rem;
        margin-top: 0.25rem;
    }

    .workspace-card,
    .card,
    .workflow,
    .insight,
    .source-card {
        background: linear-gradient(145deg, rgba(18, 57, 39, 0.94), rgba(7, 27, 19, 0.98));
        border: 1px solid var(--line-soft);
        border-radius: 14px;
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.14);
    }

    .workspace-card {
        min-height: 280px;
        padding: 1.35rem;
    }

    .workspace-title {
        color: #f4f7ff;
        font-size: 1.35rem;
        font-weight: 750;
        line-height: 1.08;
        margin-top: 0.6rem;
    }

    .workspace-copy {
        color: #9db6ae;
        font-size: 0.72rem;
        line-height: 1.5;
        margin: 0.7rem 0 1rem;
    }

    .signal-card {
        background: radial-gradient(circle at 90% 5%, rgba(131, 228, 176, 0.2), transparent 35%), linear-gradient(145deg, rgba(18, 57, 39, 0.96), rgba(7, 27, 19, 0.98));
    }

    .signal-list {
        margin-top: 1.45rem;
    }

    .signal-row {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid rgba(141, 219, 172, 0.12);
        color: #bfd8c8;
        font-size: 0.68rem;
    }

    .signal-row strong {
        color: #f5f8ff;
        font-size: 0.85rem;
        margin-left: auto;
    }

    .signal-icon {
        color: var(--accent);
        font-size: 1rem;
    }

    .workflow {
        padding: 1.25rem;
        margin-top: 1.1rem;
    }

    .workflow-intro {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
    }

    .workflow-title {
        font-size: 0.96rem;
        font-weight: 700;
        color: var(--text);
        margin-top: 0.42rem;
    }

    .workflow-subtitle {
        font-size: 0.68rem;
        color: #a8bfd0;
        margin-top: 0.15rem;
    }

    .step {
        text-align: center;
        padding-top: 1rem;
    }

    .step-number {
        width: 36px;
        height: 36px;
        margin: auto;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #3eae70, #78d99a);
        color: #062015;
        font-weight: 800;
        font-size: 0.8rem;
    }

    .step-title {
        color: #f1f5ff;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 0.6rem;
    }

    .step-text {
        color: #b7cde2;
        font-size: 0.62rem;
        margin-top: 0.18rem;
    }

    .insight {
        min-height: 155px;
        background: radial-gradient(circle at 100% 100%, rgba(105,78,225,0.45), transparent 42%), linear-gradient(145deg, #121d40, #0a1426);
        padding: 1.2rem;
        position: relative;
        overflow: hidden;
    }

    .insight::after {
        content: "AI";
        position: absolute;
        right: 1rem;
        bottom: -1.3rem;
        color: rgba(137, 132, 255, 0.15);
        font-size: 6rem;
        font-weight: 900;
    }

    .insight-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: white;
    }

    .insight-text {
        color: #aebfe1;
        font-size: 0.7rem;
        line-height: 1.5;
        margin-top: 0.6rem;
    }

    .source-card {
        scroll-margin-top: 1.5rem;
        margin: 0.65rem 0;
        padding: 0.9rem 1rem;
    }

    .source-card-header {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: #b0c9e9;
        font-size: 0.7rem;
    }

    .source-card-header strong {
        color: #edf6ff;
        margin-left: auto;
    }

    .source-number {
        color: var(--accent);
        font-weight: 800;
    }

    .source-card-text {
        color: #8ba7c0;
        font-size: 0.68rem;
        line-height: 1.5;
        margin-top: 0.55rem;
    }

    .source-anchor {
        display: inline-block;
        color: #7fbbff;
        font-size: 0.62rem;
        margin-top: 0.55rem;
        text-decoration: none;
    }

    .source-anchor:hover {
        color: #dfeeff;
        text-decoration: underline;
    }

    .inline-citation {
        display: inline-block;
        min-width: 1.35rem;
        margin: 0 0.08rem;
        padding: 0.05rem 0.26rem;
        border: 1px solid rgba(131, 228, 176, 0.6);
        border-radius: 5px;
        color: #dfffe9 !important;
        background: rgba(64, 165, 101, 0.22);
        font-size: 0.78em;
        font-weight: 800;
        text-align: center;
        text-decoration: none !important;
        vertical-align: baseline;
    }

    .inline-citation:hover {
        border-color: var(--accent);
        background: rgba(105, 211, 145, 0.4);
        color: white !important;
    }

    .upload-main {
        min-height: 215px;
        border: 1px dashed rgba(131, 228, 176, 0.75);
        border-radius: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        background: radial-gradient(circle at center, rgba(105, 211, 145, 0.14), transparent 64%), rgba(8, 37, 24, 0.7);
        transition: border-color 180ms ease, background 180ms ease;
    }

    .upload-main:hover {
        border-color: rgba(131, 228, 176, 0.95);
        background: radial-gradient(circle at center, rgba(105, 211, 145, 0.2), transparent 64%), rgba(8, 37, 24, 0.74);
    }

    .upload-icon {
        font-size: 2.5rem;
    }

    .upload-title {
        font-size: 1rem;
        font-weight: 700;
        color: #f5f8ff;
        margin-top: 0.45rem;
    }

    .upload-subtitle {
        font-size: 0.72rem;
        color: #8da4c8;
        margin-top: 0.3rem;
    }

    .stat {
        background: rgba(20,38,65,0.55);
        border: 1px solid rgba(95,130,190,0.14);
        border-radius: 11px;
        padding: 0.7rem;
        margin-bottom: 0.55rem;
        position: relative;
        overflow: hidden;
    }

    .stat::after {
        content: "";
        position: absolute;
        width: 44px;
        height: 44px;
        right: -16px;
        bottom: -18px;
        border-radius: 50%;
        background: rgba(78, 139, 255, 0.18);
    }

    .stat-number {
        font-size: 1.15rem;
        font-weight: 700;
        color: white;
    }

    .stat-label {
        font-size: 0.65rem;
        color: #8fa4c4;
    }

    .empty-state {
        text-align: center;
        padding: 2rem 0.5rem;
    }

    .empty-icon {
        font-size: 2rem;
    }

    .empty-title {
        color: #e7edf9;
        font-weight: 600;
        font-size: 0.82rem;
        margin-top: 0.5rem;
    }

    .empty-text {
        color: #8fa4c4;
        font-size: 0.68rem;
        margin-top: 0.35rem;
        line-height: 1.45;
    }

    .stButton > button {
        border-radius: 9px;
        border: 1px solid rgba(131, 228, 176, 0.28);
        background: linear-gradient(135deg, var(--accent-deep), var(--accent-strong));
        color: #062015;
        font-weight: 700;
    }

    .stButton > button:hover {
        filter: brightness(1.03);
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
</style>
""")


# ============================================================
# MAIN COMMAND BAR
# ============================================================

st.html("""
<div class="top-command-bar">
    <div class="top-brand">◈ RAG Anything</div>
</div>
""")

nav_items = [
    ("⌂", "Home"),
    ("▣", "Documents"),
    ("⌘", "Knowledge Graph"),
    ("◌", "Query & Chat"),
    ("⚙", "Settings"),
]

nav_columns = st.columns([1.1, 1, 1.25, 1.1, 0.9, 2.4])
for column, (icon, label) in zip(nav_columns[:5], nav_items):
    with column:
        is_active = st.session_state.active_page == label
        if st.button(
            f"{icon}  {label}",
            key=f"top_nav_{label.lower().replace(' ', '_')}",
            type="primary" if is_active else "secondary",
            width="stretch",
        ):
            st.session_state.active_page = label
            st.rerun()

if st.session_state.active_page == "Documents":
    _render_documents_page()
    st.stop()

if st.session_state.active_page == "Knowledge Graph":
    _render_graph_page()
    st.stop()

if st.session_state.active_page == "Settings":
    st.title("Settings")
    st.caption("Runtime configuration used by the current session.")
    st.write(
        {
            "graph_backend": os.getenv("GRAPH_STORAGE_BACKEND", "memory"),
            "generation_model": os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-20b",
            ),
            "vlm_enabled": os.getenv("USE_VLM", "false").lower() == "true",
            "ocr_enabled": os.getenv("USE_OCR", "true").lower() == "true",
        }
    )
    st.info("Secrets are read from environment variables and are never displayed.")
    st.stop()

# ============================================================
# HERO
# ============================================================

current_result = st.session_state.get("ingestion_result")
all_results = st.session_state.get("ingestion_results", [])
current_document = (
    current_result.normalized_document
    if current_result is not None
    else None
)
document_count = len(all_results)
page_count = sum(
    item.document.page_count
    for item in all_results
)
multimodal_count = (
    sum(
        1
        for item in all_results
        for element in item.normalized_document.elements
        if element.type != "text"
    )
)
entity_count = (
    len(
        {
            node.node_id
            for item in all_results
            for node in item.graph.find_by_type("entity")
        }
    )
)

st.html(f"""
<div class="topbar">
    <div>
        <div class="topbar-eyebrow">INTELLIGENCE CONSOLE / 01</div>
        <div class="topbar-title">Your knowledge, amplified.</div>
    </div>
    <div class="topbar-status"><span class="pulse-dot"></span> ALL SYSTEMS OPERATIONAL</div>
</div>

<div class="hero">
    <div class="hero-content">
        <div class="ai-badge">✦ &nbsp; MULTIMODAL INTELLIGENCE</div>
        <div class="hero-kicker">A private workspace for your most important files</div>
        <div class="hero-title">Make every document<br><span>think back.</span></div>
        <div class="hero-subtitle">
            Transform scattered PDFs and notes into a living knowledge system
            with context-aware search, visual understanding, and connected insights.
        </div>
    </div>
    <div class="hero-orbit">
        <div class="orbit-ring orbit-ring-one"></div>
        <div class="orbit-ring orbit-ring-two"></div>
        <div class="orbit-core">RA<br><span>AI</span></div>
    </div>
</div>

<div class="metric-strip">
    <div class="metric-highlight"><span class="metric-label">WORKSPACE</span><strong>READY</strong><small>Awaiting your next insight</small></div>
    <div class="metric-highlight"><span class="metric-label">DOCUMENTS</span><strong>{document_count:02d}</strong><small>Indexed sources</small></div>
    <div class="metric-highlight"><span class="metric-label">PAGES ANALYZED</span><strong>{page_count:02d}</strong><small>Across your library</small></div>
    <div class="metric-highlight"><span class="metric-label">ENTITIES FOUND</span><strong>{entity_count:02d}</strong><small>Knowledge connections</small></div>
</div>
""")


# ============================================================
# MAIN PANELS
# ============================================================

left, right = st.columns([1.65, 1], gap="medium")


# ============================================================
# UPLOAD PANEL
# ============================================================

with left:

    st.html("""
    <div class="workspace-card upload-workspace">
        <div class="section-kicker">START A NEW THREAD</div>
        <div class="workspace-title">Bring a source<br>into focus.</div>
        <div class="workspace-copy">Drop in a document and let the pipeline reveal the structure hiding inside it.</div>
    </div>
    """)

    st.markdown('<div class="main-uploader">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop a document here",
        type=["pdf", "txt"],
        key="main_upload",
        accept_multiple_files=True,
        help="PDF or TXT files up to 200MB",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("PDF or TXT · up to 200MB")

with right:
    st.html(f"""
    <div class="workspace-card signal-card">
        <div class="section-kicker">LIVE SIGNALS</div>
        <div class="workspace-title">Your workspace<br>at a glance.</div>
        <div class="signal-list">
            <div class="signal-row"><span class="signal-icon">◈</span><span>Sources indexed</span><strong>{document_count}</strong></div>
            <div class="signal-row"><span class="signal-icon">◌</span><span>Multimodal items</span><strong>{multimodal_count}</strong></div>
            <div class="signal-row"><span class="signal-icon">✧</span><span>Graph entities</span><strong>{entity_count}</strong></div>
        </div>
        <div class="signal-footer"><span class="pulse-dot"></span> Pipeline standing by</div>
    </div>
    """)


# ============================================================
# WORKFLOW
# ============================================================

st.html("""
<div class="workflow workflow-intro">
    <div>
        <div class="section-kicker">THE RAG LOOP</div>
        <div class="workflow-title">From raw file to useful thought.</div>
    </div>
    <div class="workflow-subtitle">Every source passes through a transparent intelligence pipeline.</div>
</div>
""")


steps = [
    ("01", "Capture", "Parse every page"),
    ("02", "Understand", "OCR + visual analysis"),
    ("03", "Connect", "Build the knowledge graph"),
    ("04", "Recall", "Search with context"),
]

cols = st.columns(4)

for col, (number, title, description) in zip(cols, steps):

    with col:

        st.html(
            f"""
            <div class="step">

                <div class="step-number">
                    {number}
                </div>

                <div class="step-title">
                    {title}
                </div>

                <div class="step-text">
                    {description}
                </div>

            </div>
            """
        )


# ============================================================
# INSIGHT
# ============================================================

with st.container():

    st.html("""
    <div class="insight">
        <div class="section-kicker">WHY IT FEELS DIFFERENT</div>
        <div class="insight-title">Context is the<br>new interface.</div>
        <div class="insight-text">RAG Anything turns documents into a connected surface you can explore, not a folder you forget.</div>
        <div class="insight-mark">✦</div>
    </div>
    """)


# ============================================================
# DOCUMENT PROCESSING
# ============================================================


if uploaded_files:

    st.success(
        f"📄 Selected {len(uploaded_files)} document(s): "
        + ", ".join(f"**{item.name}**" for item in uploaded_files)
    )
    selected_names = tuple(item.name for item in uploaded_files)
    stage = (
        st.session_state.processing_stage
        if st.session_state.processed_filename == selected_names
        else "Awaiting processing"
    )
    if stage == "Ready":
        st.success("Pipeline status: Ready")
    elif stage == "Failed":
        st.error(
            f"Pipeline status: Failed — {st.session_state.processing_error}"
        )
    else:
        st.info(f"Pipeline status: {stage}")

    process_col, clear_col = st.columns([1, 5])

    with process_col:

        process_clicked = st.button(
            "🚀 Process Documents",
            type="primary",
            width="stretch",
        )

    with clear_col:

        if st.button(
            "Clear",
            width="content",
        ):
            st.session_state.ingestion_result = None
            st.session_state.ingestion_results = []
            st.session_state.rag_app = None
            st.session_state.processed_filename = None
            st.rerun()


    if process_clicked:
        upload_dir = Path(
            os.getenv(
                "UPLOAD_DIR",
                "D:\\RAG_ANYTHING_DATA\\uploads",
            )
        )
        upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_paths = []
        existing_filenames = {
            item.document.filename
            for item in st.session_state.get("ingestion_results", [])
        }

        st.session_state.processing_error = None
        st.session_state.processing_stage = "Uploading"
        processing_status = st.status(
            f"Uploading {len(uploaded_files)} document(s)...",
            expanded=True,
        )

        for uploaded_file in uploaded_files:
            safe_filename = Path(uploaded_file.name).name
            file_path = upload_dir / safe_filename
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())
            if safe_filename not in existing_filenames:
                file_paths.append(file_path)

        try:
            if not file_paths:
                st.warning(
                    "These documents are already indexed in the current workspace."
                )
                st.stop()

            st.session_state.processing_stage = "Processing"
            processing_status.update(
                label=f"Processing {len(file_paths)} new document(s)...",
                state="running",
            )
            rag_app = st.session_state.rag_app
            if rag_app is None:
                pipeline = IngestionPipeline(
                    use_vlm=os.getenv("USE_VLM", "false").lower() == "true",
                    use_ocr=os.getenv("USE_OCR", "true").lower() == "true",
                )
                generation_provider = GroqGenerationProvider()
                rag_app = RAGAnythingApp(
                    ingestion_pipeline=pipeline,
                    generation_provider=generation_provider,
                )

            st.session_state.processing_stage = "Indexing"
            processing_status.update(
                label="Indexing vectors and graph...",
                state="running",
            )
            results = rag_app.ingest_many(file_paths)
            all_results = rag_app.ingestion_results
            result = all_results[-1]
            st.session_state.rag_app = rag_app
            st.session_state.ingestion_result = result
            st.session_state.ingestion_results = all_results
            st.session_state.processed_filename = selected_names
            st.session_state.processing_stage = "Ready"
            processing_status.update(
                label=f"{len(results)} document(s) ready",
                state="complete",
            )
            existing_history_ids = {
                item["document_id"]
                for item in st.session_state.document_history
            }
            for result in results:
                if result.document.document_id in existing_history_ids:
                    continue
                st.session_state.document_history.append(
                    {
                        "filename": result.document.filename,
                        "document_id": result.document.document_id,
                        "processed_at": datetime.now(
                            timezone.utc
                        ).strftime("%Y-%m-%d %H:%M UTC"),
                        "status": "Ready",
                        "pages": result.document.page_count,
                        "content_count": result.document.content_count,
                        "entity_count": sum(
                            1 for node in result.graph.get_document_nodes(
                                result.document.document_id
                            ) if node.node_type == "entity"
                        ),
                        "indexed_count": len(result.vector_store),
                        "metadata": result.document.metadata,
                    }
                )
            st.rerun()

        except Exception as exc:
            logger.exception("Document processing failed", exc_info=exc)
            st.session_state.processing_stage = "Failed"
            st.session_state.processing_error = _friendly_error(exc)
            processing_status.update(
                label="Document processing failed",
                state="error",
            )
            st.error(f"Document processing failed: {st.session_state.processing_error}")


# ============================================================
# REAL DOCUMENT STATISTICS
# ============================================================

result = st.session_state.get(
    "ingestion_result"
)


if result is not None:

    document = result.document
    normalized = result.normalized_document
    graph = result.graph


    total_elements = len(
        normalized.elements
    )

    multimodal_elements = sum(
        1
        for element in normalized.elements
        if element.type != "text"
    )

    entity_count = len(graph.find_by_type("entity"))


    st.markdown("---")

    st.subheader("📊 Processed Document")


    col1, col2, col3, col4 = st.columns(4)


    with col1:
        st.metric(
            "Pages",
            document.page_count,
        )


    with col2:
        st.metric(
            "Content Elements",
            total_elements,
        )


    with col3:
        st.metric(
            "Multimodal Items",
            multimodal_elements,
        )


    with col4:
        st.metric(
            "Entities",
            entity_count,
        )


    with st.expander(
        "📄 Document Details",
        expanded=False,
    ):

        st.write(
            {
                "Filename": document.filename,
                "Document ID": document.document_id,
                "Pages": document.page_count,
                "Content Items": document.content_count,
                "Normalized Elements": total_elements,
                "Multimodal Elements": multimodal_elements,
            }
        )
# ============================================================
# QUERY & CHAT
# ============================================================

if st.session_state.rag_app is not None:

    st.markdown("---")

    st.subheader("💬 Ask Your Document")

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                message_citations = _citation_payloads(
                    message.get("citations", []),
                    all_results,
                )
                st.markdown(
                    citation_markdown(
                        message["content"],
                        message_citations,
                        {
                            item.document.document_id: item.document.source_path
                            for item in all_results
                        },
                    ),
                    unsafe_allow_html=True,
                )
                _render_source_details(
                    _referenced_citations(
                        message["content"],
                        message_citations,
                    )
                )
            else:
                st.markdown(message["content"])

    question = st.chat_input(
        "Ask anything about your document..."
    )

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching your document..."):

                try:

                    response = st.session_state.rag_app.ask(
                        query=question,
                        top_k=min(
                            max(8, len(all_results) * 2),
                            16,
                        ),
                    )

                    answer = response.answer

                    citations = _citation_payloads(
                        response.sources,
                        all_results,
                    )
                    st.markdown(
                        citation_markdown(
                            answer,
                            citations,
                            {
                                item.document.document_id: item.document.source_path
                                for item in all_results
                            },
                        ),
                        unsafe_allow_html=True,
                    )
                    _render_source_details(
                        _referenced_citations(answer, citations)
                    )

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                        }
                    )

                except Exception as exc:
                    logger.exception(
                        "Document query failed",
                        exc_info=exc,
                    )
                    error_message = (
                        "Unable to answer the question: "
                        f"{_friendly_error(exc)} "
                        f"(error: {type(exc).__name__})"
                    )

                    st.error(error_message)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                        }
                    )        