"""Hybrid retrieval smoke tests using a tiny in-process index."""
import shutil
import tempfile
from pathlib import Path

import pytest

from src.ingest import Chunk
from src.store import VectorStore, _tokenize


@pytest.fixture
def store():
    tmp = tempfile.mkdtemp(prefix="docubot-test-")
    s = VectorStore(persist_dir=tmp, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    chunks = [
        Chunk(text="The YOLOv8 model has 3.2M parameters in the nano variant.",
              source="doc.txt", page=1, chunk_id="c1"),
        Chunk(text="Real-time object detection requires sub-100ms latency.",
              source="doc.txt", page=1, chunk_id="c2"),
        Chunk(text="ByteTrack assigns persistent IDs across frames.",
              source="doc.txt", page=2, chunk_id="c3"),
        Chunk(text="The cake recipe calls for three eggs and one cup of flour.",
              source="doc.txt", page=3, chunk_id="c4"),
    ]
    s.add(chunks)
    yield s
    shutil.rmtree(tmp, ignore_errors=True)


def test_tokenize_handles_punctuation_and_case():
    assert _tokenize("YOLO-v8, 3.2M parameters!") == ["yolo", "v8", "3", "2m", "parameters"]


def test_dense_only_finds_paraphrase(store):
    # paraphrase query — keyword match is weak
    hits = store.search("how fast can the detector be", top_k=2, alpha=0.0)
    ids = [h.chunk_id for h in hits]
    assert "c2" in ids, "dense should find the latency chunk on paraphrase"


def test_bm25_only_finds_exact_keyword(store):
    # an exact rare-keyword query — semantic embedding may miss it
    hits = store.search("ByteTrack", top_k=2, alpha=1.0)
    ids = [h.chunk_id for h in hits]
    assert "c3" in ids and ids[0] == "c3"


def test_hybrid_handles_both(store):
    hits = store.search("3.2M parameters", top_k=2, alpha=0.5)
    assert hits[0].chunk_id == "c1"


def test_alpha_zero_matches_pure_dense_top_id(store):
    a = store.search("real-time inference latency", top_k=1, alpha=0.0)
    b = store.search("real-time inference latency", top_k=1, alpha=0.0)
    assert a[0].chunk_id == b[0].chunk_id


def test_alpha_one_with_no_token_overlap_returns_empty(store):
    hits = store.search("xyzzy nonsense-token", top_k=2, alpha=1.0)
    assert hits == []


def test_reset_clears_bm25(store):
    assert store._bm25 is not None
    store.reset()
    assert store._bm25 is None
    assert store.count() == 0
