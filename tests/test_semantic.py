import numpy as np
import pytest

from app import db, semantic


class FakeModel:
    """Deterministic embeddings: a few keywords map to fixed directions,
    so nearest-neighbor behaves predictably without the real model."""

    KEYWORDS = ["budget", "zebra", "website"]

    def encode(self, texts, normalize_embeddings=True):
        out = []
        for text in texts:
            v = np.zeros(semantic.EMBEDDING_DIM, dtype=np.float32)
            v[0] = 0.05  # baseline so nothing is a zero vector
            for i, kw in enumerate(self.KEYWORDS):
                if kw in text.lower():
                    v[i + 1] = 1.0
            out.append(v / np.linalg.norm(v))
        return np.array(out)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic, "_model", FakeModel())
    c = db.connect(db_path=tmp_path / "test.db", key="ab" * 32)
    yield c
    c.close()


def add_transcript(conn, text, path):
    rec_id = db.insert_recording(conn, path, path)
    return db.insert_transcript(conn, rec_id, text, "en", "tiny", [])


def test_chunk_text_splits_long_text():
    text = "word " * 1000  # 5000 chars
    chunks = semantic.chunk_text(text)
    assert len(chunks) > 3
    assert all(len(c) <= semantic.CHUNK_CHARS for c in chunks)


def test_chunk_text_short_text_is_one_chunk():
    assert semantic.chunk_text("short meeting note") == ["short meeting note"]


def test_chunk_text_empty():
    assert semantic.chunk_text("   ") == []


def test_index_and_search_finds_relevant_transcript(conn):
    t1 = add_transcript(conn, "we argued about the budget for an hour", "/a")
    t2 = add_transcript(conn, "the zebra escaped from the zoo", "/b")
    semantic.index_transcript(conn, t1, "we argued about the budget for an hour")
    semantic.index_transcript(conn, t2, "the zebra escaped from the zoo")

    hits = semantic.search(conn, "what was the budget decision", k=2)
    assert hits[0]["transcript_id"] == t1


def test_reindex_replaces_old_chunks(conn):
    tid = add_transcript(conn, "the zebra escaped", "/a")
    semantic.index_transcript(conn, tid, "the zebra escaped")
    semantic.index_transcript(conn, tid, "now we discuss the website launch")

    n = conn.execute("SELECT count(*) FROM chunks WHERE transcript_id = ?",
                     (tid,)).fetchone()[0]
    assert n == 1
    hits = semantic.search(conn, "website plans", k=1)
    assert "website" in hits[0]["text"]


def test_ask_returns_sourced_answer(conn, monkeypatch):
    tid = add_transcript(conn, "we decided the budget is fifty dollars", "/a")
    semantic.index_transcript(conn, tid, "we decided the budget is fifty dollars")

    seen = {}

    def fake_llm(prompt, privileged=False):
        seen["prompt"] = prompt
        return "The budget is fifty dollars [1]."
    from app import summarize
    monkeypatch.setattr(summarize, "_call_llm", fake_llm)

    result = semantic.ask(conn, "what did we decide about the budget")
    assert "fifty dollars" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["n"] == 1
    assert "we decided the budget" in seen["prompt"]


def test_ask_with_empty_index(conn):
    result = semantic.ask(conn, "anything")
    assert result["sources"] == []
