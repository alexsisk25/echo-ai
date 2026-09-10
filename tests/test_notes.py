"""Hybrid notes: autosave, enhance (idempotent, privileged), FTS, export."""

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from app import config, db, main, notes, summarize

ENHANCED_JSON = json.dumps({
    "sections": [
        {"heading": "Budget decision",
         "bullets": ["Sam approved the 40k budget for Q3"]},
        {"heading": "Also discussed",
         "bullets": ["Ship date moved to Friday"]},
    ]
})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "notes.db")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)

    conn = db.connect()
    rec_id = db.insert_recording(conn, str(tmp_path / "standup.m4a"), "h1", 60.0)
    db.set_recording_status(conn, rec_id, "done")
    tid = db.insert_transcript(
        conn, rec_id, "sam approved the forty thousand budget and we ship "
        "friday", "en", "tiny",
        [{"start": 0.0, "end": 5.0, "text": "sam approved the budget",
          "speaker": "Sam"}],
    )
    bare_id = db.insert_recording(conn, str(tmp_path / "bare.m4a"), "h2", 10.0)
    conn.close()
    return TestClient(main.app), rec_id, tid, bare_id


def test_save_and_read_back(env):
    client, rec_id, _, _ = env
    resp = client.put(f"/api/recordings/{rec_id}/notes",
                      json={"text": "budget?? ask sam\nship date"})
    assert resp.json() == {"id": rec_id, "saved": True}
    detail = client.get(f"/api/recordings/{rec_id}").json()
    assert detail["notes"]["raw_text"] == "budget?? ask sam\nship date"
    assert detail["notes"]["enhanced"] is None

    # Autosave overwrites raw text in place: still one row.
    client.put(f"/api/recordings/{rec_id}/notes", json={"text": "budget!"})
    conn = db.connect()
    assert conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 1
    assert conn.execute(
        "SELECT raw_text FROM notes WHERE recording_id = ?", (rec_id,)
    ).fetchone()[0] == "budget!"
    conn.close()


def test_save_missing_recording_is_404(env):
    client, *_ = env
    assert client.put("/api/recordings/999/notes",
                      json={"text": "x"}).status_code == 404


def test_enhance_stores_separately_and_never_touches_raw(env, monkeypatch):
    client, rec_id, _, _ = env
    calls = []

    def fake_llm(prompt, privileged=False):
        calls.append((prompt, privileged))
        return ENHANCED_JSON

    monkeypatch.setattr(summarize, "_call_llm", fake_llm)
    client.put(f"/api/recordings/{rec_id}/notes",
               json={"text": "budget?? ask sam"})
    resp = client.post(f"/api/recordings/{rec_id}/notes/enhance")
    assert resp.status_code == 200
    body = resp.json()
    assert body["cached"] is False
    assert body["enhanced"]["sections"][0]["heading"] == "Budget decision"
    # The jottings and the transcript both reach the LLM.
    assert "budget?? ask sam" in calls[0][0]
    assert "ship" in calls[0][0]

    detail = client.get(f"/api/recordings/{rec_id}").json()
    assert detail["notes"]["raw_text"] == "budget?? ask sam"
    assert detail["notes"]["enhanced"]["sections"][0]["heading"] == \
        "Budget decision"
    assert detail["notes"]["enhanced_at"] is not None


def test_enhance_is_idempotent_until_notes_change(env, monkeypatch):
    client, rec_id, _, _ = env
    calls = []
    monkeypatch.setattr(
        summarize, "_call_llm",
        lambda prompt, privileged=False: calls.append(1) or ENHANCED_JSON)

    client.put(f"/api/recordings/{rec_id}/notes", json={"text": "budget"})
    assert client.post(
        f"/api/recordings/{rec_id}/notes/enhance").json()["cached"] is False
    assert client.post(
        f"/api/recordings/{rec_id}/notes/enhance").json()["cached"] is True
    assert len(calls) == 1

    # Changing the notes invalidates the cache and re-runs the LLM.
    client.put(f"/api/recordings/{rec_id}/notes",
               json={"text": "budget and ship date"})
    assert client.post(
        f"/api/recordings/{rec_id}/notes/enhance").json()["cached"] is False
    assert len(calls) == 2


def test_enhance_routes_privileged_to_local(env, monkeypatch):
    client, rec_id, _, _ = env
    seen = []
    monkeypatch.setattr(
        summarize, "_call_llm",
        lambda prompt, privileged=False: seen.append(privileged)
        or ENHANCED_JSON)
    client.post(f"/api/recordings/{rec_id}/privileged",
                json={"privileged": True})
    client.put(f"/api/recordings/{rec_id}/notes", json={"text": "budget"})
    client.post(f"/api/recordings/{rec_id}/notes/enhance")
    assert seen == [True]


def test_enhance_error_cases(env, monkeypatch):
    client, rec_id, _, bare_id = env
    # No notes yet.
    assert client.post(
        f"/api/recordings/{rec_id}/notes/enhance").status_code == 422
    # Notes but no transcript.
    client.put(f"/api/recordings/{bare_id}/notes", json={"text": "hello"})
    assert client.post(
        f"/api/recordings/{bare_id}/notes/enhance").status_code == 404


def test_notes_are_searchable(env):
    client, rec_id, _, _ = env
    client.put(f"/api/recordings/{rec_id}/notes",
               json={"text": "remember the xylophone invoice"})
    results = client.get("/api/search", params={"q": "xylophone"}).json()
    assert len(results) == 1
    assert results[0]["source"] == "notes"
    assert results[0]["recording_id"] == rec_id
    assert "<mark>xylophone</mark>" in results[0]["snippet"]

    # Transcript search still works alongside notes search.
    transcript_hits = client.get("/api/search",
                                 params={"q": "budget"}).json()
    assert len(transcript_hits) == 1
    assert "source" not in transcript_hits[0]
    assert transcript_hits[0]["recording_id"] == rec_id

    # Editing notes updates the index.
    client.put(f"/api/recordings/{rec_id}/notes", json={"text": "nothing"})
    assert client.get("/api/search", params={"q": "xylophone"}).json() == []


def test_export_includes_notes(env, monkeypatch, tmp_path):
    client, rec_id, _, _ = env
    (tmp_path / "standup.m4a").write_bytes(b"audio")
    monkeypatch.setattr(summarize, "_call_llm",
                        lambda prompt, privileged=False: ENHANCED_JSON)
    client.put(f"/api/recordings/{rec_id}/notes",
               json={"text": "budget?? ask sam"})
    client.post(f"/api/recordings/{rec_id}/notes/enhance")

    resp = client.get("/api/export", params={"ids": str(rec_id)})
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    assert "my-notes.md" in names
    assert "enhanced-notes.md" in names
    assert "budget?? ask sam" in zf.read("my-notes.md").decode()
    enhanced = zf.read("enhanced-notes.md").decode()
    assert "## Budget decision" in enhanced
    assert "- Sam approved the 40k budget for Q3" in enhanced


def test_delete_removes_notes(env):
    client, rec_id, _, _ = env
    client.put(f"/api/recordings/{rec_id}/notes",
               json={"text": "remember the xylophone invoice"})
    assert client.delete(f"/api/recordings/{rec_id}").status_code == 200
    conn = db.connect()
    assert conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 0
    conn.close()
    assert client.get("/api/search", params={"q": "xylophone"}).json() == []
