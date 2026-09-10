"""Provenance log: append-only events, every write path, seed, activity."""

import json

import pytest
from fastapi.testclient import TestClient

from app import config, db, main, provenance, summarize, sync


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "prov.db")
    monkeypatch.setattr(config, "INBOX_DIR", tmp_path / "inbox")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(sync, "LEDGER", tmp_path / "data" / "synced.txt")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)
    (tmp_path / "inbox").mkdir()
    return TestClient(main.app), tmp_path


def events(conn, **kw):
    return provenance.list_events(conn, **kw)


def test_append_only_and_listing(env):
    conn = db.connect()
    provenance.log_imported(conn, "memo-a", "memo-a.qta", ".qta")
    provenance.log_captured(conn, "20260101 090000-meeting", "x-meeting.m4a")
    all_ev = events(conn)
    assert [e["event_type"] for e in all_ev] == ["captured", "imported"] \
        or {e["event_type"] for e in all_ev} == {"captured", "imported"}
    imported = events(conn, event_type="imported")
    assert len(imported) == 1
    assert imported[0]["context"]["source_format"] == ".qta"
    conn.close()


def test_sync_writes_imported_events(env, monkeypatch):
    client, tmp_path = env
    source = tmp_path / "vm"
    source.mkdir()
    (source / "20260710 100000-AAA.m4a").write_bytes(b"AAAA")
    conn = db.connect()
    assert sync.sync_voice_memos(source, conn=conn) == 1
    ev = events(conn, event_type="imported")
    assert len(ev) == 1
    assert ev[0]["stem"] == "20260710 100000-AAA"
    assert ev[0]["context"]["original_filename"] == "20260710 100000-AAA.m4a"
    conn.close()


def test_delete_writes_deleted_event_with_title_and_how(env):
    client, tmp_path = env
    conn = db.connect()
    rec = db.insert_recording(conn, str(tmp_path / "inbox" / "m.m4a"), "h", 5.0)
    tid = db.insert_transcript(conn, rec, "hi", "en", "tiny",
                              [{"start": 0, "end": 1, "text": "hi",
                                "speaker": None}])
    db.upsert_summary(conn, tid, json.dumps({"title": "Budget Call"}),
                     model="t")
    conn.close()

    assert client.delete(f"/api/recordings/{rec}").status_code == 200
    conn = db.connect()
    ev = events(conn, event_type="deleted")
    assert len(ev) == 1
    assert ev[0]["context"]["title"] == "Budget Call"
    assert ev[0]["context"]["how"] == "single"
    assert ev[0]["context"]["recording_id"] == rec
    conn.close()


def test_bulk_delete_marks_how_bulk(env):
    client, tmp_path = env
    conn = db.connect()
    a = db.insert_recording(conn, str(tmp_path / "inbox" / "a.m4a"), "ha", 1.0)
    b = db.insert_recording(conn, str(tmp_path / "inbox" / "b.m4a"), "hb", 1.0)
    conn.close()
    client.post("/api/recordings/bulk-delete", json={"ids": [a, b]})
    conn = db.connect()
    ev = events(conn, event_type="deleted")
    assert len(ev) == 2
    assert all(e["context"]["how"] == "bulk" for e in ev)
    conn.close()


def test_sync_skipped_as_deleted_is_recorded(env):
    client, tmp_path = env
    source = tmp_path / "vm"
    source.mkdir()
    # Import, then delete (records a deleted event + ledger entry).
    (source / "20260710 100000-AAA.m4a").write_bytes(b"AAAA")
    conn = db.connect()
    sync.sync_voice_memos(source, conn=conn)
    rec = db.insert_recording(
        conn, str(tmp_path / "inbox" / "20260710 100000-AAA.m4a"),
        "hh", 1.0)
    conn.close()
    from app import manage
    conn = db.connect()
    manage.delete_recording(conn, rec, how="single")
    conn.close()
    # Apple renames it to .qta; a re-sync must be blocked AND recorded.
    (source / "20260710 100000-AAA.m4a").unlink()
    (source / "20260710 100000-AAA.qta").write_bytes(b"QQQ")
    conn = db.connect()
    assert sync.sync_voice_memos(source, conn=conn) == 0
    skipped = events(conn, event_type="sync-skipped-as-deleted")
    assert len(skipped) == 1
    assert skipped[0]["stem"] == "20260710 100000-AAA"
    conn.close()


def test_seed_is_idempotent_and_covers_recordings_and_cleanup(env):
    client, tmp_path = env
    conn = db.connect()
    db.insert_recording(conn, str(tmp_path / "inbox" / "20260101 090000-X.m4a"),
                       "h1", 1.0)
    db.insert_recording(conn, str(tmp_path / "inbox" / "20260202 100000-meeting.m4a"),
                       "h2", 1.0)
    n = provenance.seed(conn)
    # 2 recordings + 16 cleanup stems.
    assert n == 2 + len(provenance.QTA_CLEANUP_STEMS)
    assert provenance.seed(conn) == 0  # idempotent
    types = {e["event_type"] for e in events(conn)}
    assert types == {"imported", "captured", "deleted"}
    # The captured one is the meeting file.
    cap = events(conn, event_type="captured")
    assert cap[0]["stem"] == "20260202 100000-meeting"
    # A known cleanup stem is present as deleted with the incident note.
    deleted = events(conn, event_type="deleted")
    assert any(e["context"].get("context") == "qta incident cleanup"
               for e in deleted)
    conn.close()


def test_activity_endpoint_seeds_and_filters(env):
    client, tmp_path = env
    conn = db.connect()
    db.insert_recording(conn, str(tmp_path / "inbox" / "20260101 090000-X.m4a"),
                       "h1", 1.0)
    conn.close()
    # First hit seeds from the one recording + cleanup stems.
    body = client.get("/api/activity").json()
    assert any(e["event_type"] == "imported" for e in body)
    assert any(e["event_type"] == "deleted" for e in body)
    only_deleted = client.get("/api/activity", params={"type": "deleted"}).json()
    assert only_deleted and all(
        e["event_type"] == "deleted" for e in only_deleted)
