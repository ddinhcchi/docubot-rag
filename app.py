import tempfile
from pathlib import Path

import streamlit as st

from src.chat import Chatter
from src.config import settings
from src.ingest import chunk_document, load_document
from src.rate_limit import RateLimiter
from src.store import VectorStore

st.set_page_config(page_title="DocuBot — RAG Q&A", page_icon="🤖", layout="wide")
st.title("🤖 DocuBot — Ask your documents")
st.caption(
    "Upload PDF / DOCX / TXT. Multilingual embeddings (incl. Vietnamese). "
    "Answers grounded in the source with page-level citations."
)


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore(
        persist_dir=settings.chroma_dir,
        embedding_model=settings.embedding_model,
    )


@st.cache_resource
def get_chatter() -> Chatter | None:
    if not settings.groq_api_key:
        return None
    return Chatter(api_key=settings.groq_api_key, model=settings.groq_model)


@st.cache_resource
def get_limiter() -> RateLimiter:
    return RateLimiter(per_minute=settings.rate_limit_per_minute)


store = get_store()
chatter = get_chatter()
limiter = get_limiter()

with st.sidebar:
    st.subheader("Retrieval")
    hybrid_alpha = st.slider(
        "BM25 weight (0 = semantic only, 1 = keyword only)",
        0.0, 1.0, settings.hybrid_alpha, 0.05,
        help="Dense + BM25 hybrid. 0.3 default leans semantic but lets exact-match queries through.",
    )
    st.subheader("Index")
    st.metric("Chunks indexed", store.count())
    if store.sources():
        st.write("**Sources:**")
        for s in store.sources():
            st.write(f"- {s}")
    st.divider()
    uploaded = st.file_uploader(
        "Add documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded:
        added_total = 0
        with st.spinner(f"Indexing {len(uploaded)} file(s)…"):
            for file in uploaded:
                suffix = Path(file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.getbuffer())
                    tmp_path = Path(tmp.name)
                try:
                    pages = load_document(tmp_path)
                    chunks = chunk_document(
                        pages,
                        source_name=file.name,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                    )
                    added = store.add(chunks)
                    added_total += added
                finally:
                    tmp_path.unlink(missing_ok=True)
        st.success(f"Added {added_total} new chunks across {len(uploaded)} file(s).")
        st.rerun()

    if st.button("🗑️ Reset index", use_container_width=True):
        store.reset()
        st.rerun()

    st.divider()
    st.caption(
        "Configured ✅" if chatter else "Set `GROQ_API_KEY` in `.env` to enable answers"
    )
    st.caption(f"Model: `{settings.groq_model}`")

if "history" not in st.session_state:
    st.session_state.history = []

def _expander_label(source: str, page: int, score: float | None) -> str:
    base = f"📄 {source}, page {page}"
    if score is None:
        return base
    return f"{base}  ·  similarity {score:.2f}"


for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        for src in entry.get("sources", []):
            with st.expander(_expander_label(src["source"], src["page"], src.get("score"))):
                st.write(src["text"])

prompt = st.chat_input("Ask a question about your documents…")
if prompt:
    if chatter is None:
        st.error("Groq API key not configured.")
        st.stop()
    if not limiter.allow("global"):
        st.warning(
            f"Rate limit reached ({settings.rate_limit_per_minute}/min). "
            f"Retry in {limiter.reset_in('global'):.0f}s."
        )
        st.stop()

    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            hits = store.search(prompt, top_k=settings.top_k, alpha=hybrid_alpha)

        if not hits:
            # Avoid streaming a hard-coded "no docs" message — render once.
            answer_text = chatter.ask(prompt, hits).text
            st.markdown(answer_text)
        else:
            # Streaming response — first token arrives in <200 ms typically
            answer_text = st.write_stream(chatter.ask_stream(prompt, hits))

        for src in hits:
            with st.expander(_expander_label(src.source, src.page, src.score)):
                st.write(src.text)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": [
                {"source": s.source, "page": s.page, "text": s.text, "score": s.score}
                for s in hits
            ],
        }
    )
