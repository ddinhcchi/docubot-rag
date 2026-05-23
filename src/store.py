from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .ingest import Chunk

_COLLECTION = "docubot"


class VectorStore:
    """ChromaDB persistent wrapper with a local sentence-transformer encoder."""

    def __init__(self, persist_dir: str, embedding_model: str):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self.encoder = SentenceTransformer(embedding_model)

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
        return len(new_chunks)

    def search(self, query: str, top_k: int = 4) -> list[Chunk]:
        if self.collection.count() == 0:
            return []
        emb = self.encoder.encode([query], convert_to_numpy=True).tolist()
        result = self.collection.query(
            query_embeddings=emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[Chunk] = []
        docs = result["documents"][0] if result["documents"] else []
        metas = result["metadatas"][0] if result["metadatas"] else []
        ids = result["ids"][0] if result["ids"] else []
        # ChromaDB returns cosine *distance* (0 = identical, 2 = opposite);
        # convert to similarity in [-1, 1] — for normalized embeddings this is
        # effectively bounded to [0, 1] for natural language queries.
        dists = result["distances"][0] if result.get("distances") else [None] * len(ids)
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            similarity = None if dist is None else round(1.0 - float(dist), 4)
            out.append(
                Chunk(
                    text=text,
                    source=str(meta.get("source", "")),
                    page=int(meta.get("page", 0)),
                    chunk_id=chunk_id,
                    score=similarity,
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

    def count(self) -> int:
        return self.collection.count()

    def sources(self) -> list[str]:
        if self.collection.count() == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in all_meta if "source" in m})
