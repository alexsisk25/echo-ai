"""Recovery actions: retry, re-sync from source, reprocess scopes.

The load-bearing tests are the preserve-human-work guarantees: no
recovery path may clear manual titles, pinned segments, human-set
speaker names, tag rejections, notes, folders, commitment checkboxes,
or the privileged flag.
"""

import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from app import config, db, main, reprocess, summarize, sync, transcribe


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def make_audio(path, seconds=1, freq=440):
    """A small real AAC file every ffmpeg tool can read."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         f"sine=frequency={freq}:duration={seconds}", "-c:a", "aac",
         "-f", "mov", str(path)],
        check=True, capture_output=True, text=True,
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Shared DB (also for background threads) plus fake ML stack."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "INBOX_DIR", tmp_path / "inbox")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)
    monkeypatch.setattr(sync, "LEDGER", tmp_path / "data" / "ledger.txt")
    (tmp_path / "inbox").mkdir()

    def fake_transcribe(audio_path, model=None):
        return transcribe.TranscriptionResult(
            text="new transcript text",
            language="en", model="fake-model",
            segments=[
                {"start": 0.0, "end": 2.0, "text": "first new line"},
                {"start": 2.0, "end": 4.0, "text": "second new line"},
                {"start": 4.0, "end": 6.0, "text": "third new line"},
            ],
            duration_seconds=6.0,
        )
    monkeypatch.setattr(reprocess.transcribe, "transcribe", fake_transcribe)

    from app import calendar_sync, diarize
    monkeypatch.setattr(
        diarize, "diarize",
        lambda p, num_speakers=None, **kw: [
            {"start": 0.0, "end": 6.0, "speaker": "SPEAKER_00"}])
    # Guided splitting and the suspect check need no real models here.
    monkeypatch.setattr(reprocess.voices, "guided_split",
                        lambda *a, **k: {"split": {}})

    # Reprocessing must never re-match a recording that already has a
    # calendar row (a match or a dismissal is a human-visible decision).
    def guard_match(conn, rid):
        row = conn.execute(
            "SELECT 1 FROM calendar_matches WHERE recording_id = ?",
            (rid,)).fetchone()
        assert row is None, "match_recording ran despite an existing match"
        return None
    monkeypatch.setattr(calendar_sync, "match_recording", guard_match)
    monkeypatch.setattr(reprocess.voices, "auto_tag",
                        lambda conn, tid, path: {})
    monkeypatch.setattr(reprocess.semantic, "index_transcript",
                        lambda conn, tid, text: 1)
    # commitments.extract embeds the task to find its best source
    # segment; no test here may load the real embedding model.
    monkeypatch.setattr(reprocess.commitments, "_best_segment",
                        lambda conn, tid, task: None)
    monkeypatch.setattr(
        reprocess.summarize, "summarize",
        lambda text, privileged=False: summarize.MeetingSummary(
            title="Fresh AI Title", summary="A fresh summary.",
            action_items=[{"owner": "Ana", "task": "новый task",
                           "due_date": None, "priority": "low"}],
        ))

    reprocess.JOBS.clear()
    conn = db.connect()
    yield TestClient(main.app), conn, tmp_path
    conn.close()


def seed_recording(conn, tmp_path, status="done", with_human_work=True):
    """A recording rich in human work, to prove nothing clears it."""
    audio = config.INBOX_DIR / "20260721 122422-11F34D7B.m4a"
    make_audio(audio)
    rec_id = db.insert_recording(conn, str(audio), f"hash-{audio.name}")
    conn.execute("UPDATE recordings SET status = ? WHERE id = ?",
                 (status, rec_id))
    if status == "failed":
        conn.commit()
        return rec_id, None

    tid = db.insert_transcript(conn, rec_id, "old transcript text", "en",
                               "old-model", [
        {"start": 0.0, "end": 2.0, "text": "old pinned line",
         "speaker": "Alex Sisk", "pinned": True},
        {"start": 2.0, "end": 4.0, "text": "old human named line",
         "speaker": "Felix Vivanco"},
        {"start": 4.0, "end": 6.0, "text": "old machine line",
         "speaker": "SPEAKER_01"},
    ])
    if with_human_work:
        conn.execute("UPDATE recordings SET title = ?, privileged = 1 "
                     "WHERE id = ?", ("My Own Title", rec_id))
        conn.execute("INSERT INTO folders (name) VALUES ('Keep')")
        conn.execute(
            "UPDATE recordings SET folder_id = "
            "(SELECT id FROM folders WHERE name = 'Keep') WHERE id = ?",
            (rec_id,))
        conn.execute(
            "INSERT INTO notes (recording_id, raw_text, enhanced_json) "
            "VALUES (?, ?, ?)",
            (rec_id, "my precious jottings", '{"sections": ["enhanced"]}'))
        conn.execute(
            "INSERT INTO calendar_matches (recording_id, event_id, "
            "event_title, event_start, event_end, status) VALUES "
            "(?, 'ev1', 'Real Event', '2026-07-21T12:00:00', "
            "'2026-07-21T13:00:00', 'matched')",
            (rec_id,))
        conn.execute(
            "INSERT INTO rejected_tags (transcript_id, label, name) "
            "VALUES (?, 'SPEAKER_01', 'Wrong Guy')", (tid,))
        conn.execute(
            "INSERT INTO commitments (transcript_id, owner, task, status) "
            "VALUES (?, 'Alex', 'old done task', 'done')", (tid,))
        db.upsert_summary(conn, tid, '{"title": "Old AI Title"}', "m")
    conn.commit()
    return rec_id, tid


def human_work_intact(conn, rec_id, tid):
    row = conn.execute(
        "SELECT r.title, r.privileged, f.name FROM recordings r "
        "LEFT JOIN folders f ON f.id = r.folder_id WHERE r.id = ?",
        (rec_id,)).fetchone()
    assert row == ("My Own Title", 1, "Keep")
    assert conn.execute(
        "SELECT raw_text, enhanced_json FROM notes WHERE recording_id = ?",
        (rec_id,)).fetchone() == ("my precious jottings",
                                  '{"sections": ["enhanced"]}')
    assert conn.execute(
        "SELECT event_title, status FROM calendar_matches "
        "WHERE recording_id = ?",
        (rec_id,)).fetchone() == ("Real Event", "matched")
    assert conn.execute(
        "SELECT label, name FROM rejected_tags WHERE transcript_id = ?",
        (tid,)).fetchone() == ("SPEAKER_01", "Wrong Guy")
    assert conn.execute(
        "SELECT status FROM commitments WHERE transcript_id = ? "
        "AND task = 'old done task'", (tid,)).fetchone()[0] == "done"


def test_full_process_preserves_all_human_work(env):
    _client, conn, tmp_path = env
    rec_id, tid = seed_recording(conn, tmp_path)

    from pathlib import Path
    path = Path(conn.execute(
        "SELECT file_path FROM recordings WHERE id = ?",
        (rec_id,)).fetchone()[0])
    reprocess._full_process(conn, rec_id, path)

    human_work_intact(conn, rec_id, tid)
    # Same transcript row, new machine content.
    assert conn.execute(
        "SELECT full_text FROM transcripts WHERE id = ?",
        (tid,)).fetchone()[0] == "new transcript text"
    # Pinned line and human name re-attached by time overlap.
    segs = conn.execute(
        "SELECT text, speaker, pinned FROM segments WHERE transcript_id = ? "
        "ORDER BY start_seconds", (tid,)).fetchall()
    assert segs[0] == ("first new line", "Alex Sisk", 1)
    assert segs[1] == ("second new line", "Felix Vivanco", 0)
    # AI summary replaced; the summary title is not the manual title.
    assert "Fresh AI Title" in conn.execute(
        "SELECT summary_json FROM summaries WHERE transcript_id = ?",
        (tid,)).fetchone()[0]
    # New commitments added, old ones (and their state) kept.
    tasks = {r[0] for r in conn.execute(
        "SELECT task FROM commitments WHERE transcript_id = ?",
        (tid,)).fetchall()}
    assert "old done task" in tasks and "новый task" in tasks


def test_speakers_only_replaces_machine_labels_and_nothing_else(env,
                                                                monkeypatch):
    client, conn, tmp_path = env
    rec_id, tid = seed_recording(conn, tmp_path)

    def explode(*a, **k):
        raise AssertionError("speakers-only must not re-transcribe")
    monkeypatch.setattr(reprocess.transcribe, "transcribe", explode)

    resp = client.post(f"/api/recordings/{rec_id}/reprocess",
                       json={"scope": "speakers"})
    assert resp.status_code == 200
    assert wait_for(
        lambda: reprocess.job_status(rec_id)["state"] != "running")
    assert reprocess.job_status(rec_id)["state"] == "done"

    human_work_intact(conn, rec_id, tid)
    segs = conn.execute(
        "SELECT text, speaker, pinned FROM segments WHERE transcript_id = ? "
        "ORDER BY start_seconds", (tid,)).fetchall()
    # Human lines untouched, machine line freshly diarized, text intact.
    assert segs[0] == ("old pinned line", "Alex Sisk", 1)
    assert segs[1] == ("old human named line", "Felix Vivanco", 0)
    assert segs[2] == ("old machine line", "SPEAKER_00", 0)
    assert conn.execute(
        "SELECT full_text FROM transcripts WHERE id = ?",
        (tid,)).fetchone()[0] == "old transcript text"
    # Provenance carries the action and scope.
    ev = conn.execute(
        "SELECT context_json FROM provenance WHERE event_type = "
        "'reprocessed'").fetchone()
    assert ev and '"speakers"' in ev[0] and '"done"' in ev[0]


def test_notes_only_rewrites_summary_without_touching_audio_paths(
        env, monkeypatch):
    client, conn, tmp_path = env
    rec_id, tid = seed_recording(conn, tmp_path)

    def explode(*a, **k):
        raise AssertionError("notes-only must not run audio work")
    monkeypatch.setattr(reprocess.transcribe, "transcribe", explode)
    from app import diarize
    monkeypatch.setattr(diarize, "diarize", explode)

    resp = client.post(f"/api/recordings/{rec_id}/reprocess",
                       json={"scope": "notes"})
    assert resp.status_code == 200
    assert wait_for(
        lambda: reprocess.job_status(rec_id)["state"] != "running")
    assert reprocess.job_status(rec_id)["state"] == "done"

    human_work_intact(conn, rec_id, tid)
    assert "Fresh AI Title" in conn.execute(
        "SELECT summary_json FROM summaries WHERE transcript_id = ?",
        (tid,)).fetchone()[0]
    # Segments and transcript untouched.
    assert conn.execute(
        "SELECT COUNT(*) FROM segments WHERE transcript_id = ? "
        "AND text LIKE 'old%'", (tid,)).fetchone()[0] == 3


def test_retry_runs_pipeline_on_failed_recording(env):
    client, conn, tmp_path = env
    rec_id, _ = seed_recording(conn, tmp_path, status="failed")

    resp = client.post(f"/api/recordings/{rec_id}/retry")
    assert resp.status_code == 200
    assert wait_for(
        lambda: reprocess.job_status(rec_id)["state"] != "running")
    assert reprocess.job_status(rec_id)["state"] == "done"
    assert conn.execute(
        "SELECT status FROM recordings WHERE id = ?",
        (rec_id,)).fetchone()[0] == "done"
    assert conn.execute(
        "SELECT full_text FROM transcripts WHERE recording_id = ?",
        (rec_id,)).fetchone()[0] == "new transcript text"
    ev = conn.execute(
        "SELECT context_json FROM provenance WHERE event_type = 'retried'"
    ).fetchone()
    assert ev and '"done"' in ev[0]


def test_retry_refuses_unreadable_file_and_points_at_resync(env):
    client, conn, tmp_path = env
    rec_id, _ = seed_recording(conn, tmp_path, status="failed")
    # Corrupt the audio the way the real incident did: junk bytes with
    # no moov atom.
    from pathlib import Path
    path = Path(conn.execute(
        "SELECT file_path FROM recordings WHERE id = ?",
        (rec_id,)).fetchone()[0])
    path.write_bytes(b"\x00" * 4096)

    resp = client.post(f"/api/recordings/{rec_id}/retry")
    assert resp.status_code == 422
    assert "Re-sync from source" in resp.json()["detail"]
    # The blocked attempt is on the paper trail.
    ev = conn.execute(
        "SELECT context_json FROM provenance WHERE event_type = 'retried'"
    ).fetchone()
    assert ev and "blocked-unreadable" in ev[0]


def test_retry_recovers_a_recording_stuck_mid_transcription(env):
    """A worker that dies mid-run leaves the row on "transcribing"
    forever: the watcher skips it (its hash is already in the database)
    and Reprocess needs a transcript that was never written. Retry is
    the only way back, so it has to accept that state. Found live on
    recording 94, stuck since 2026-08-12."""
    client, conn, tmp_path = env
    # Exactly the real shape: a row that got as far as "transcribing"
    # and never wrote a transcript.
    audio = config.INBOX_DIR / "20260728 150621-3C7465DE.m4a"
    make_audio(audio)
    rec_id = db.insert_recording(conn, str(audio), "hash-stuck-row")
    conn.execute("UPDATE recordings SET status = 'transcribing' WHERE id = ?",
                 (rec_id,))
    conn.commit()

    resp = client.post(f"/api/recordings/{rec_id}/retry")
    assert resp.status_code == 200
    assert wait_for(
        lambda: reprocess.job_status(rec_id)["state"] != "running")
    assert conn.execute(
        "SELECT status FROM recordings WHERE id = ?",
        (rec_id,)).fetchone()[0] == "done"
    assert conn.execute(
        "SELECT full_text FROM transcripts WHERE recording_id = ?",
        (rec_id,)).fetchone()[0] == "new transcript text"


def test_retry_on_done_recording_is_409(env):
    client, conn, tmp_path = env
    rec_id, _ = seed_recording(conn, tmp_path)
    assert client.post(f"/api/recordings/{rec_id}/retry").status_code == 409


def test_reprocess_on_failed_recording_is_409(env):
    client, conn, tmp_path = env
    rec_id, _ = seed_recording(conn, tmp_path, status="failed")
    resp = client.post(f"/api/recordings/{rec_id}/reprocess",
                       json={"scope": "everything"})
    assert resp.status_code == 409


def test_resync_replaces_corrupt_local_copy_from_source(env, monkeypatch):
    client, conn, tmp_path = env
    rec_id, tid = seed_recording(conn, tmp_path)

    # The real case: fine in Voice Memos (as .qta), corrupt locally.
    memos = tmp_path / "voicememos"
    memos.mkdir()
    make_audio(memos / "20260721 122422-11F34D7B.qta", freq=880)
    monkeypatch.setattr(sync, "VOICE_MEMOS_DIR", memos)
    from pathlib import Path
    local = Path(conn.execute(
        "SELECT file_path FROM recordings WHERE id = ?",
        (rec_id,)).fetchone()[0])
    local.write_bytes(b"\x00" * 4096)
    sync.add_to_ledger(local.name)

    resp = client.post(f"/api/recordings/{rec_id}/resync")
    assert resp.status_code == 200
    assert wait_for(
        lambda: reprocess.job_status(rec_id)["state"] != "running")
    assert reprocess.job_status(rec_id)["state"] == "done", \
        reprocess.job_status(rec_id)["error"]

    # Fresh decodable audio under the same name and recording id.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(local)], capture_output=True, text=True)
    assert float(probe.stdout.strip()) > 0.5
    # The stem is back in the ledger so sync will not re-import it.
    assert local.stem in sync._load_ledger()
    # Same id, new hash, reprocessed content, human work intact.
    assert conn.execute(
        "SELECT file_hash FROM recordings WHERE id = ?",
        (rec_id,)).fetchone()[0] != f"hash-{local.name}"
    assert conn.execute(
        "SELECT full_text FROM transcripts WHERE id = ?",
        (tid,)).fetchone()[0] == "new transcript text"
    human_work_intact(conn, rec_id, tid)
    ev = conn.execute(
        "SELECT context_json FROM provenance WHERE event_type = 'resynced'"
    ).fetchone()
    assert ev and '"done"' in ev[0] and ".qta" in ev[0]


def test_resync_without_source_is_a_plain_404(env, monkeypatch):
    client, conn, tmp_path = env
    rec_id, _ = seed_recording(conn, tmp_path)
    memos = tmp_path / "empty-memos"
    memos.mkdir()
    monkeypatch.setattr(sync, "VOICE_MEMOS_DIR", memos)
    resp = client.post(f"/api/recordings/{rec_id}/resync")
    assert resp.status_code == 404
    assert "Voice Memos folder" in resp.json()["detail"]


def test_job_endpoint_reports_state(env):
    client, conn, tmp_path = env
    rec_id, _ = seed_recording(conn, tmp_path)
    body = client.get(f"/api/recordings/{rec_id}/job").json()
    assert body == {"recording_status": "done", "job": None}
    client.post(f"/api/recordings/{rec_id}/reprocess",
                json={"scope": "notes"})
    assert wait_for(
        lambda: client.get(f"/api/recordings/{rec_id}/job").json()["job"]
        is not None)
