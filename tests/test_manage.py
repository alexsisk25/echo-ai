"""Recording management: delete (single and bulk), no-resync, export."""

import io
import json
import zipfile

import pytest
import sqlite_vec
from fastapi.testclient import TestClient

from app import config, db, main, sync

SUMMARY = {
    "title": "Walrus Standup",
    "date": "2026-07-16",
    "attendees": ["Alexa", "Sam"],
    "summary": "The team decided to ship the walrus project on Friday.",
    "decisions": ["Ship on Friday"],
    "action_items": [{"owner": "Alexa", "task": "ship the project",
                      "due_date": "2026-07-18", "priority": "high"}],
    "risks": ["Tight deadline"],
    "topics": ["walrus", "shipping"],
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "manage.db")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "INBOX_DIR", tmp_path / "inbox")
    monkeypatch.setattr(sync, "LEDGER", tmp_path / "data" / "synced_memos.txt")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)
    (tmp_path / "inbox").mkdir()
    return TestClient(main.app), tmp_path


def make_recording(conn, audio_path, file_hash):
    """A recording with every kind of derived data the app can attach."""
    audio_path.write_bytes(b"fake audio bytes")
    rec_id = db.insert_recording(conn, str(audio_path), file_hash, 5.0)
    db.set_recording_status(conn, rec_id, "done")
    tid = db.insert_transcript(
        conn, rec_id, "we decided to ship the walrus project on friday",
        "en", "tiny",
        [{"start": 0.0, "end": 3.0, "text": "we decided",
          "speaker": "Alexa"},
         {"start": 3.0, "end": 5.0,
          "text": "to ship the walrus project on friday",
          "speaker": "SPEAKER_01"}],
    )
    db.upsert_summary(conn, tid, json.dumps(SUMMARY), model="test")
    seg_id = conn.execute(
        "SELECT id FROM segments WHERE transcript_id = ?", (tid,)
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO commitments (transcript_id, owner, task, segment_id) "
        "VALUES (?, 'Alexa', 'ship the project', ?)", (tid, seg_id))
    cur = conn.execute(
        "INSERT INTO chunks (transcript_id, text) VALUES (?, ?)",
        (tid, "we decided to ship"))
    conn.execute(
        "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
        (cur.lastrowid, sqlite_vec.serialize_float32([0.0] * 384)))
    conn.execute(
        "INSERT INTO translations (transcript_id, lang, full_text) "
        "VALUES (?, 'es', 'decidimos')", (tid,))
    conn.execute(
        "INSERT INTO calendar_matches (recording_id, event_id, event_title, "
        "event_start, event_end) VALUES (?, 'ev1', 'Standup', "
        "'2026-07-16T09:00:00', '2026-07-16T09:30:00')", (rec_id,))
    conn.execute(
        "INSERT INTO rejected_tags (transcript_id, label, name) "
        "VALUES (?, 'SPEAKER_01', 'Sam')", (tid,))
    conn.execute(
        "INSERT OR REPLACE INTO dossiers (name, dossier_json, meetings_count) "
        "VALUES ('Alexa', '{}', 1)")
    conn.execute(
        "INSERT OR REPLACE INTO voiceprints (name, embedding, num_samples) "
        "VALUES ('Alexa', X'00', 1)")
    conn.commit()
    return rec_id, tid


def count(conn, table, where, args):
    return conn.execute(
        f"SELECT count(*) FROM {table} WHERE {where}", args
    ).fetchone()[0]


def test_delete_removes_everything(env):
    client, tmp_path = env
    conn = db.connect()
    audio = tmp_path / "inbox" / "memo.m4a"
    rec_id, tid = make_recording(conn, audio, "hash-a")

    resp = client.delete(f"/api/recordings/{rec_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == rec_id
    assert body["audio_deleted"] is True

    assert count(conn, "recordings", "id = ?", (rec_id,)) == 0
    assert count(conn, "transcripts", "recording_id = ?", (rec_id,)) == 0
    assert count(conn, "segments", "transcript_id = ?", (tid,)) == 0
    assert count(conn, "summaries", "transcript_id = ?", (tid,)) == 0
    assert count(conn, "commitments", "transcript_id = ?", (tid,)) == 0
    assert count(conn, "chunks", "transcript_id = ?", (tid,)) == 0
    assert count(conn, "chunks_vec", "rowid >= 0", ()) == 0
    assert count(conn, "translations", "transcript_id = ?", (tid,)) == 0
    assert count(conn, "rejected_tags", "transcript_id = ?", (tid,)) == 0
    assert count(conn, "calendar_matches", "recording_id = ?", (rec_id,)) == 0
    assert count(conn, "dossiers", "name = 'Alexa'", ()) == 0
    # FTS entry is gone: keyword search no longer finds the transcript.
    assert db.search_transcripts(conn, "walrus") == []
    # Voiceprint enrollments survive a delete.
    assert count(conn, "voiceprints", "name = 'Alexa'", ()) == 1
    assert not audio.exists()
    conn.close()


def test_delete_missing_recording_is_404(env):
    client, _ = env
    assert client.delete("/api/recordings/999").status_code == 404


def test_bulk_delete(env):
    client, tmp_path = env
    conn = db.connect()
    rec_a, _ = make_recording(conn, tmp_path / "inbox" / "a.m4a", "hash-a")
    rec_b, _ = make_recording(conn, tmp_path / "inbox" / "b.m4a", "hash-b")

    resp = client.post("/api/recordings/bulk-delete",
                       json={"ids": [rec_a, rec_b, 999]})
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 2}
    assert count(conn, "recordings", "1 = 1", ()) == 0
    conn.close()


def test_deleted_memo_never_resyncs(env):
    client, tmp_path = env
    source = tmp_path / "voice_memos"
    source.mkdir()
    (source / "memo.m4a").write_bytes(b"fake audio bytes")

    assert sync.sync_voice_memos(source_dir=source) == 1
    conn = db.connect()
    rec_id, _ = make_recording(
        conn, tmp_path / "inbox" / "memo.m4a", "hash-a")

    assert client.delete(f"/api/recordings/{rec_id}").status_code == 200
    assert not (tmp_path / "inbox" / "memo.m4a").exists()
    # The memo still sits in the Voice Memos folder, but the ledger
    # remembers it: a fresh sync copies nothing back.
    assert sync.sync_voice_memos(source_dir=source) == 0
    assert not (tmp_path / "inbox" / "memo.m4a").exists()
    conn.close()


def test_export_single_recording(env):
    client, tmp_path = env
    conn = db.connect()
    rec_id, _ = make_recording(conn, tmp_path / "inbox" / "memo.m4a", "hash-a")
    conn.close()

    resp = client.get("/api/export", params={"ids": str(rec_id)})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    # The recording carries a calendar match titled "Standup", which
    # names the export (manual > calendar > AI title).
    assert "Standup.zip" in resp.headers["content-disposition"]

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert sorted(zf.namelist()) == ["memo.m4a", "notes.md", "transcript.md"]
    assert zf.read("memo.m4a") == b"fake audio bytes"
    transcript = zf.read("transcript.md").decode()
    assert "# Standup" in transcript
    assert "[00:00:00] Alexa: we decided" in transcript
    assert "[00:00:03] SPEAKER_01: to ship the walrus project" in transcript
    notes = zf.read("notes.md").decode()
    assert "# Walrus Standup" in notes
    assert "Ship on Friday" in notes
    assert "- Alexa: ship the project (due 2026-07-18, priority high)" in notes
    assert "walrus, shipping" in notes


def test_export_bulk_uses_folders(env):
    client, tmp_path = env
    conn = db.connect()
    rec_a, _ = make_recording(conn, tmp_path / "inbox" / "a.m4a", "hash-a")
    rec_b, _ = make_recording(conn, tmp_path / "inbox" / "b.m4a", "hash-b")
    conn.close()

    resp = client.get("/api/export", params={"ids": f"{rec_a},{rec_b}"})
    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    prefix_a = f"{rec_a:03d}-Standup/"
    prefix_b = f"{rec_b:03d}-Standup/"
    assert prefix_a + "transcript.md" in names
    assert prefix_a + "notes.md" in names
    assert prefix_a + "a.m4a" in names
    assert prefix_b + "transcript.md" in names
    assert prefix_b + "b.m4a" in names


def test_export_missing_recording_is_404(env):
    client, _ = env
    assert client.get("/api/export", params={"ids": "999"}).status_code == 404
    assert client.get("/api/export", params={"ids": "abc"}).status_code == 422


def test_delete_keeps_audio_shared_with_another_recording(env):
    # Two rows can end up pointing at the same audio file (e.g. a junk
    # failed duplicate next to the healthy recording). Deleting one must
    # never take the survivor's audio with it.
    client, tmp_path = env
    conn = db.connect()
    audio = tmp_path / "inbox" / "shared.m4a"
    healthy, _ = make_recording(conn, audio, "hash-healthy")
    junk = db.insert_recording(conn, str(audio), "hash-junk-dup", 5.0)
    db.set_recording_status(conn, junk, "failed")
    conn.close()

    resp = client.delete(f"/api/recordings/{junk}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["file_kept_shared"] is True
    assert body["audio_deleted"] is False
    # The junk row is gone, the survivor and its audio are untouched.
    assert audio.exists()
    conn = db.connect()
    assert conn.execute(
        "SELECT 1 FROM recordings WHERE id = ?", (junk,)).fetchone() is None
    assert conn.execute(
        "SELECT status FROM recordings WHERE id = ?",
        (healthy,)).fetchone()[0] == "done"
    conn.close()
    # Deleting the last reference removes the file as before.
    resp = client.delete(f"/api/recordings/{healthy}")
    assert resp.json()["audio_deleted"] is True
    assert not audio.exists()


# Exports carried notes but never the folder digest, so a bulk export of
# a folder was missing the one artefact that describes the folder as a
# whole. Absent or stale says so in the file: an omitted file reads as
# "this folder has no digest", which is a different claim.

def _digest_for(conn, folder_id, covered):
    from app import digest as digest_mod

    d = digest_mod.FolderDigest(
        overview="The folder moved from pricing to equity over 2026.",
        themes=[digest_mod.Theme(theme="Pricing", first_appeared="2026-01-02",
                                 framing_then="an open question",
                                 framing_now="settled",
                                 how_it_changed="Resolved in March.")],
        shifts=[digest_mod.Shift(date="2026-03-01", shift="Pricing settled",
                                 note="after the March meeting")],
        open_threads=[digest_mod.OpenThread(thread="Who owns renewals",
                                            raised_on="2026-02-01")],
        commitments=[digest_mod.DigestCommitment(
            owner="Alexa", commitment="send the deck", status="open")],
        quotes=[digest_mod.Quote(quote="we should just ship it",
                                 speaker="Alexa", recording_id=covered[0],
                                 timestamp="00:00:12", why="the turn")],
    )
    digest_mod._store(conn, folder_id, d, covered, privileged=False)


def test_bulk_export_includes_the_folder_digest(env):
    from app import organize

    client, tmp_path = env
    conn = db.connect()
    rec_a, _ = make_recording(conn, tmp_path / "inbox" / "a.m4a", "hash-a")
    rec_b, _ = make_recording(conn, tmp_path / "inbox" / "b.m4a", "hash-b")
    folder = organize.create_folder(conn, "Business Planning")
    organize.assign_folder(conn, [rec_a, rec_b], folder["id"])
    _digest_for(conn, folder["id"], [rec_a, rec_b])
    conn.close()

    resp = client.get("/api/export", params={"ids": f"{rec_a},{rec_b}"})
    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert "Business-Planning/digest.md" in zf.namelist()
    body = zf.read("Business-Planning/digest.md").decode()
    assert "# Business Planning digest" in body
    assert "Generated from 2 recordings, last updated" in body
    assert "The folder moved from pricing to equity" in body
    assert "Pricing" in body and "Who owns renewals" in body
    assert "we should just ship it" in body
    assert "out of date" not in body


def test_export_says_so_when_the_digest_is_stale(env):
    from app import organize

    client, tmp_path = env
    conn = db.connect()
    rec_a, _ = make_recording(conn, tmp_path / "inbox" / "a.m4a", "hash-a")
    rec_b, _ = make_recording(conn, tmp_path / "inbox" / "b.m4a", "hash-b")
    folder = organize.create_folder(conn, "Therapy")
    organize.assign_folder(conn, [rec_a, rec_b], folder["id"])
    # Built from one of them, so the other counts as added since.
    _digest_for(conn, folder["id"], [rec_a])
    conn.close()

    resp = client.get("/api/export", params={"ids": f"{rec_a},{rec_b}"})
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    body = zf.read("Therapy/digest.md").decode()
    assert "This digest is out of date" in body
    assert "1 recording added since it was written" in body
    # The last version generated is still there, not dropped.
    assert "The folder moved from pricing to equity" in body


def test_export_says_so_when_there_is_no_digest(env):
    from app import organize

    client, tmp_path = env
    conn = db.connect()
    rec_a, _ = make_recording(conn, tmp_path / "inbox" / "a.m4a", "hash-a")
    rec_b, _ = make_recording(conn, tmp_path / "inbox" / "b.m4a", "hash-b")
    folder = organize.create_folder(conn, "Personal")
    organize.assign_folder(conn, [rec_a, rec_b], folder["id"])
    conn.close()

    resp = client.get("/api/export", params={"ids": f"{rec_a},{rec_b}"})
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert "Personal/digest.md" in zf.namelist()
    body = zf.read("Personal/digest.md").decode()
    assert "No digest has been generated for this folder yet." in body
    assert "2 recordings are filed here" in body


def test_unfiled_bulk_export_has_no_digest_file(env):
    client, tmp_path = env
    conn = db.connect()
    rec_a, _ = make_recording(conn, tmp_path / "inbox" / "a.m4a", "hash-a")
    rec_b, _ = make_recording(conn, tmp_path / "inbox" / "b.m4a", "hash-b")
    conn.close()

    resp = client.get("/api/export", params={"ids": f"{rec_a},{rec_b}"})
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert not any(n.endswith("digest.md") for n in zf.namelist())


def test_single_export_has_no_digest_file(env):
    from app import organize

    client, tmp_path = env
    conn = db.connect()
    rec_a, _ = make_recording(conn, tmp_path / "inbox" / "a.m4a", "hash-a")
    folder = organize.create_folder(conn, "Business Planning")
    organize.assign_folder(conn, [rec_a], folder["id"])
    _digest_for(conn, folder["id"], [rec_a])
    conn.close()

    resp = client.get("/api/export", params={"ids": str(rec_a)})
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert not any(n.endswith("digest.md") for n in zf.namelist())
