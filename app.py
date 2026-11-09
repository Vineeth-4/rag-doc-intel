"""Streamlit UI for the RAG Document Intelligence System."""
import os
import re

import streamlit as st

# Streamlit Cloud: copy the key from secrets into the environment for src/llm.py
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

from src.llm import LLMError  # noqa: E402
from src.ingestion import SUPPORTED_EXTENSIONS, ingest_file  # noqa: E402
from src.pipeline import answer_question  # noqa: E402
from src.retrieval import MODES, Retriever  # noqa: E402
from src.store import DocumentStore  # noqa: E402

st.set_page_config(page_title="Document Intelligence", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run downloads them)...")
def load_system():
    store = DocumentStore()
    return store, Retriever(store)


store, retriever = load_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📁 Documents")
    uploads = st.file_uploader(
        "Upload files", type=[e.strip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if st.button("Index uploaded files", disabled=not uploads, type="primary"):
        progress = st.progress(0.0)
        for i, f in enumerate(uploads, 1):
            try:
                chunks = ingest_file(f.name, f.getvalue())
                store.add_chunks(chunks)
                st.success(f"{f.name}: {len(chunks)} chunks")
            except Exception as e:
                st.error(f"{f.name}: {e}")
            progress.progress(i / len(uploads))

    docs = store.list_documents()
    st.subheader(f"Indexed ({len(docs)})")
    for name, n in docs.items():
        c1, c2 = st.columns([4, 1])
        c1.caption(f"{name} · {n} chunks")
        if c2.button("🗑", key=f"del_{name}", help="Remove this document"):
            store.delete_document(name)
            st.rerun()

    st.divider()
    st.header("⚙️ Settings")
    selected = st.multiselect("Search only in", list(docs), help="Empty = search all documents")
    decompose = st.toggle("Break down complex questions", value=True,
                       help="Only used when a question looks multi-part. Turn off to save API calls.")
    mode = st.selectbox("Retrieval mode", MODES, index=2,
                        help="vector → hybrid (+BM25) → hybrid_rerank (+cross-encoder)")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def render_sources(sources: list[dict], answer: str = "") -> None:
    """Show cited passages first; anything retrieved but not cited goes in a second expander."""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    numbered = list(enumerate(sources, 1))
    used = [(i, s) for i, s in numbered if i in cited] if cited else numbered
    other = [(i, s) for i, s in numbered if i not in cited] if cited else []

    def show(items):
        for i, s in items:
            st.markdown(f"**[{i}] {s['source']}** — page {s['page']}")
            st.caption(s["text"])

    with st.expander(f"📎 Sources cited ({len(used)})"):
        show(used)
    if other:
        with st.expander(f"Other retrieved passages, not cited ({len(other)})"):
            show(other)


# ---------------- Chat ----------------
st.title("📚 Document Intelligence")
st.caption("Ask questions about your documents. Answers cite the exact source passages.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("error"):
            st.error(m["content"])
            continue
        st.markdown(m["content"])
        if m.get("sub_questions") and len(m["sub_questions"]) > 1:
            st.caption("Searched for: " + " · ".join(m["sub_questions"]))
        if m.get("sources"):
            render_sources(m["sources"], m["content"])

if question := st.chat_input("Ask a question about your documents"):
    if not docs:
        st.warning("Upload and index at least one document first.")
        st.stop()

    # Errors are not part of the conversation the model should see.
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages if not m.get("error")]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    result, error = None, None
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and reasoning..."):
                result = answer_question(
                    retriever, question, sources=selected or None, history=history,
                    use_decomposition=decompose, mode=mode,
                )
        except LLMError as e:
            error = str(e)
        except Exception as e:  # anything unexpected: show it, never hang
            error = f"Unexpected error: {e}"

        if error:
            st.error(error)
        else:
            st.markdown(result["answer"])
            if len(result["sub_questions"]) > 1:
                st.caption("Searched for: " + " · ".join(result["sub_questions"]))
            if result["sources"]:
                render_sources(result["sources"], result["answer"])

    if error:
        st.session_state.messages.append(
            {"role": "assistant", "content": error, "error": True})
    else:
        st.session_state.messages.append({
            "role": "assistant", "content": result["answer"],
            "sources": result["sources"], "sub_questions": result["sub_questions"],
        })
