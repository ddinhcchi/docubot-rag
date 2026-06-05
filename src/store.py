from pathlib import Path

import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .ingest import Chunk

_COLLECTION = "docubot"
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Cheap whitespace + lowercase tokenizer. Works on Vietnamese diacritics
    because re.UNICODE keeps composed letters intact."""
    return _TOKEN_RE.findall(text.lower())


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return scores
    lo = min(scores.values())
    hi = max(scores.values())
    span = hi - lo
    if span < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / span for k, v in scores.items()}


class VectorStore:
    """Dense (sentence-transformer) + BM25 hybrid retrieval over ChromaDB.

    Dense alone misses exact-match queries ("find me the row where regex
    `\\w+` is mentioned"). BM25 alone misses paraphrases. Mixing them with
    a tunable α gives the best of both — α defaults to 0.3 (dense-leaning).
    """

    def __init__(self, persist_dir: str, embedding_model: str):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self.encoder = SentenceTransformer(embedding_model)
        # BM25 lives in-memory only — cheap to rebuild from ChromaDB on cold
        # start, no extra serialization to keep in sync.
        self._bm25_ids: list[str] = []
        self._bm25_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._rehydrate_bm25()

    def _rehydrate_bm25(self) -> None:
        if self.collection.count() == 0:
            self._bm25 = None
            return
        data = self.collection.get(include=["documents"])
        self._bm25_ids = list(data["ids"])
        self._bm25_tokens = [_tokenize(d or "") for d in data["documents"]]
        self._bm25 = BM25Okapi(self._bm25_tokens) if self._bm25_tokens else None

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        existing = set(self.collection.get(ids=[c.chunk_id for c in chunks])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            return 0

        embeddings = self.encoder.encode(
            [c.text for c in new_chunks],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        self.collection.add(
            ids=[c.chunk_id for c in new_chunks],
            documents=[c.text for c in new_chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source, "page": c.page} for c in new_chunks],
        )
        # Extend the in-memory BM25 corpus then rebuild — BM25Okapi has no
        # incremental fit; a 1k-chunk index rebuilds in <50 ms.
        for c in new_chunks:
            self._bm25_ids.append(c.chunk_id)
            self._bm25_tokens.append(_tokenize(c.text))
        self._bm25 = BM25Okapi(self._bm25_tokens)
        return len(new_chunks)

    def _search_dense(self, query: str, n: int) -> dict[str, float]:
        """Return {chunk_id: cosine_similarity} for the top-n dense hits."""
        if self.collection.count() == 0:
            return {}
        emb = self.encoder.encode([query], convert_to_numpy=True).tolist()
        result = self.collection.query(
            query_embeddings=emb,
            n_results=n,
            include=["distances"],
        )
        ids = result["ids"][0] if result["ids"] else []
        dists = result["distances"][0] if result.get("distances") else [0.0] * len(ids)
        return {cid: 1.0 - float(d) for cid, d in zip(ids, dists)}

    def _search_bm25(self, query: str, n: int) -> dict[str, float]:
        if not self._bm25 or not self._bm25_ids:
            return {}
        tokens = _tokenize(query)
        if not tokens:
            return {}
        scores = self._bm25.get_scores(tokens)
        # top-n indices
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return {self._bm25_ids[i]: float(scores[i]) for i in top_idx if scores[i] > 0}

    def search(self, query: str, top_k: int = 4, alpha: float = 0.0) -> list[Chunk]:
        """Hybrid retrieval.

        alpha=0.0 → pure dense (semantic)
        alpha=1.0 → pure BM25 (keyword)
        else      → linear combine with min-max normalized scores
        """
        if self.collection.count() == 0:
            return []

        alpha = max(0.0, min(1.0, alpha))
        if alpha == 0.0:
            picked = self._search_dense(query, top_k)
        elif alpha == 1.0:
            picked = self._search_bm25(query, top_k)
        else:
            # broader pool so both detectors get a fair shot at top-k
            pool = max(top_k * 5, 20)
            dense = _minmax(self._search_dense(query, pool))
            sparse = _minmax(self._search_bm25(query, pool))
            all_ids = set(dense) | set(sparse)
            picked = {
                cid: (1 - alpha) * dense.get(cid, 0.0) + alpha * sparse.get(cid, 0.0)
                for cid in all_ids
            }

        if not picked:
            return []

        top_ids = sorted(picked, key=picked.get, reverse=True)[:top_k]
        rows = self.collection.get(
            ids=top_ids, include=["documents", "metadatas"]
        )
        by_id: dict[str, tuple[str, dict]] = {
            rid: (doc, meta)
            for rid, doc, meta in zip(rows["ids"], rows["documents"], rows["metadatas"])
        }
        out: list[Chunk] = []
        for cid in top_ids:
            if cid not in by_id:
                continue
            doc, meta = by_id[cid]
            out.append(
                Chunk(
                    text=doc,
                    source=str(meta.get("source", "")),
                    page=int(meta.get("page", 0)),
                    chunk_id=cid,
                    score=round(picked[cid], 4),
                )
            )
        return out

    def reset(self) -> None:
        try:
            self.client.delete_collection(_COLLECTION)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25_ids = []
        self._bm25_tokens = []
        self._bm25 = None

    def count(self) -> int:
        return self.collection.count()

    def sources(self) -> list[str]:
        if self.collection.count() == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in all_meta if "source" in m})
