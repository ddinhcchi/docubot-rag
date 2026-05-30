# 🤖 DocuBot — RAG Q&A over your documents

Upload a PDF / DOCX / TXT, ask questions in English **or** Vietnamese, get answers grounded in the source with page-level citations. Built on Llama 3.1 (Groq), multilingual sentence-transformer embeddings, and ChromaDB.

![demo](demo/demo.gif)

> Demo GIF placeholder: upload a PDF → ask 2 English questions → ask 1 Vietnamese question → expand "Sources" panel to show the cited chunks.

---

## Why this project

"Chat with my PDFs" is the smallest useful LLM product a small business will ever pay for — and the one most likely to get hand-waved by a junior who copies a tutorial that hard-codes OpenAI, English-only embeddings, and zero citations. This repo shows the path that *actually* works in production:

- **Citations every answer** — the system prompt forces `[source, page N]` after each claim. No hallucinated page references.
- **Multilingual out of the box** — `paraphrase-multilingual-MiniLM-L12-v2` handles VN/EN/50+ langs.
- **Cheap inference** — Groq free tier (Llama 3.1 8B) at ~200 tok/s, no credit card.
- **Sliding-window rate limit** for safe public deploys — your free quota stays yours.
- **Persistent ChromaDB** so re-launching the app keeps your index.

---

## Architecture

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌─────────┐
│ PDF/DOCX/TXT │ → │  page-aware     │ → │  multilingual │ → │ Chroma  │
│   upload     │    │  chunk splitter │    │   embedder   │    │ vector  │
└──────────────┘    └─────────────────┘    └──────────────┘    │  store  │
                                                                └────┬────┘
                            ┌──────────────────────────────────────┘
                            ▼
                ┌─────────────────────────┐    ┌───────────────────┐
                │ top-k cosine retrieval  │ → │ Groq Llama 3.1 8B │
                │   + page metadata       │    │   + citation rule │
                └─────────────────────────┘    └───────────────────┘
```

### Code layout

| File | Responsibility |
|---|---|
| [`src/ingest.py`](src/ingest.py) | PDF / DOCX / TXT loaders + recursive chunk splitter with page metadata |
| [`src/store.py`](src/store.py) | `VectorStore` — persistent ChromaDB + sentence-transformer encoder |
| [`src/chat.py`](src/chat.py) | Groq client + system prompt that forces inline citations |
| [`src/rate_limit.py`](src/rate_limit.py) | Sliding-window rate limiter (per-IP / per-key) |
| [`src/config.py`](src/config.py) | `.env`-driven settings |
| [`app.py`](app.py) | Streamlit chat UI + file upload sidebar |

---

## Quick start

You need a Groq API key — sign up free at <https://console.groq.com>, no credit card.

```bash
git clone <this-repo>
cd docubot-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: GROQ_API_KEY=gsk_...

streamlit run app.py
```

Open <http://localhost:8501> → upload a PDF → ask away.

First run downloads `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) — cached afterwards.

---

## Sample interaction

```
You: How many parameters does YOLOv8n have?
Bot: YOLOv8n has 3.2M parameters, making it suitable for edge deployment.
     [yolov8-paper.pdf, page 1]

You: Trong tài liệu này, "object detection" tiếng Việt là gì?
Bot: Trong tài liệu này, "object detection" trong tiếng Việt là
     "phát hiện đối tượng". [yolov8-paper.pdf, page 1]
```

The Sources expander beneath each answer shows the exact chunk(s) the model was given.

---

## Configuration

All knobs live in `.env`:

| Variable | Default | Effect |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Get one at console.groq.com |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Swap for `llama-3.3-70b-versatile` for higher quality |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Any sentence-transformer model |
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 80 | Sliding overlap between chunks |
| `TOP_K` | 4 | Number of chunks retrieved per question |
| `CHROMA_DIR` | `.chroma` | Local persistent vector store |
| `RATE_LIMIT_PER_MINUTE` | 10 | Questions per minute (sliding window). `0` disables. |

### Picking a Groq model

Groq's free tier is generous but rate-limited per model. As of writing the limits are roughly 30 req/min for the 8B models and 30 req/min for the 70B (your dashboard shows live quotas). Pick by job:

| Model | When to use it |
|---|---|
| `llama-3.1-8b-instant` *(default)* | Demos, short factual answers, dense citations. ~200 tok/s, lowest latency, follows the language-lock prompt well after the v2 fix. |
| `llama-3.3-70b-versatile` | Long synthesis questions, contracts with cross-references, anything where reasoning matters more than latency. ~100 tok/s. |
| `mixtral-8x7b-32768` | Long single-document QA (32k context) where you'd rather skip retrieval entirely and stuff the whole doc. Higher hallucination risk if you do. |

Switch any time by editing `GROQ_MODEL` in `.env` and restarting — no code change required.

---

## Run with Docker

```bash
docker build -t docubot-rag .
docker run --rm -p 8501:8501 --env-file .env docubot-rag
```

The image pre-downloads the embedding model so first query is fast.

---

## Design notes

- **Idempotent ingestion** — chunk IDs are hashes of `(source, page, text)`. Re-uploading the same PDF adds zero new rows.
- **Cosine, not L2** — created with `hnsw:space: cosine` so similarity scores are intuitive.
- **Page numbers are preserved end-to-end** — from `pypdf` reader → chunk metadata → retrieval result → LLM prompt → citation in answer.
- **System prompt is strict** — explicit "if not in sources, say so" reduces hallucinated citations dramatically vs. an unconstrained prompt.
- **Rate limit is per-key** — drop in a real per-IP key from `streamlit.session_state` or your reverse proxy when going public.

---

## Security — secret scanning

The repo wires up [`gitleaks`](https://github.com/gitleaks/gitleaks) via [`pre-commit`](https://pre-commit.com) so a real Groq API key (or any other credential) can never end up in a commit by accident. The hook runs locally on every `git commit` and blocks the commit if anything matches.

```bash
brew install gitleaks pre-commit   # macOS — Linux users: pipx install both
pre-commit install                  # installs the .git/hooks/pre-commit shim
pre-commit run --all-files          # scan everything already in the index
gitleaks detect --source . --verbose # scan the full git history
```

[`.gitleaks.toml`](.gitleaks.toml) extends the default ruleset with two allowlists:

- `.env.example` and `README.md` — they intentionally contain placeholder credentials
- The literal `gsk_...` ellipsis token shown in the README is not a real Groq key — Groq keys are 56 characters, this one is 6

For deployments, also enable [GitHub Push Protection](https://docs.github.com/en/code-security/secret-scanning/push-protection-for-repositories-and-organizations) as a second line of defence — server-side scanning catches anyone who skipped `pre-commit install`.

---

## Roadmap

- BM25 hybrid retrieval (better for short, keyword-heavy queries)
- Per-document "delete" button (currently it's all-or-nothing reset)
- Streaming responses via Groq's SSE API
- OCR for scanned PDFs (Tesseract / PaddleOCR)

---

## License

MIT
