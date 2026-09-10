"""Speaker fixes: per-line reassign, merge guard, reset speakers."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, db, main, voices


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "speakers.db")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)

    conn = db.connect()
    rec_id = db.insert_recording(conn, str(tmp_path / "meet.m4a"), "h1", 12.0)
    db.set_recording_status(conn, rec_id, "done")
    tid = db.insert_transcript(
        conn, rec_id, "full text", "en", "tiny",
        [{"start": 0.0, "end": 2.0, "text": "line a", "speaker": "SPEAKER_00"},
         {"start": 2.0, "end": 4.0, "text": "line b", "speaker": "SPEAKER_00"},
         {"start": 4.0, "end": 6.0, "text": "line c", "speaker": "SPEAKER_00"},
         {"start": 6.0, "end": 8.0, "text": "line d", "speaker": "SPEAKER_01"},
         {"start": 8.0, "end": 10.0, "text": "line e", "speaker": "SPEAKER_01"}],
    )
    conn.close()
    return TestClient(main.app), rec_id, tid


def seg_rows(tid):
    conn = db.connect()
    rows = conn.execute(
        "SELECT text, speaker, auto_original, pinned FROM segments "
        "WHERE transcript_id = ? ORDER BY start_seconds", (tid,)
    ).fetchall()
    conn.close()
    return rows


def test_per_line_reassign_touches_only_that_segment(env):
    client, rec_id, tid = env
    detail = client.get(f"/api/recordings/{rec_id}").json()
    target = detail["segments"][1]  # "line b" in the SPEAKER_00 cluster
    assert target["pinned"] is False

    resp = client.post(
        f"/api/transcripts/{tid}/segments/{target['id']}/speaker",
        json={"name": "Maria"})
    assert resp.status_code == 200
    assert resp.json() == {"segment_id": target["id"], "speaker": "Maria",
                           "pinned": True}

    rows = seg_rows(tid)
    assert rows[1] == ("line b", "Maria", None, 1)
    # The rest of the cluster did not move.
    assert rows[0] == ("line a", "SPEAKER_00", None, 0)
    assert rows[2] == ("line c", "SPEAKER_00", None, 0)

    detail = client.get(f"/api/recordings/{rec_id}").json()
    assert detail["segments"][1]["speaker"] == "Maria"
    assert detail["segments"][1]["pinned"] is True


def test_per_line_reassign_error_cases(env):
    client, _, tid = env
    assert client.post(
        f"/api/transcripts/{tid}/segments/99999/speaker",
        json={"name": "Maria"}).status_code == 404
    detail_seg = seg_rows(tid)
    assert all(r[3] == 0 for r in detail_seg)
    first_id = 1
    assert client.post(
        f"/api/transcripts/{tid}/segments/{first_id}/speaker",
        json={"name": "   "}).status_code == 422


def test_cluster_rename_skips_pinned_lines(env, monkeypatch):
    client, _, tid = env
    monkeypatch.setattr(main.voices, "enroll_from_recording",
                        lambda *a: 0)
    detail_rows = seg_rows(tid)
    conn = db.connect()
    seg_id = conn.execute(
        "SELECT id FROM segments WHERE transcript_id = ? AND text = 'line b'",
        (tid,)).fetchone()[0]
    conn.close()

    client.post(f"/api/transcripts/{tid}/segments/{seg_id}/speaker",
                json={"name": "Maria"})
    resp = client.post(f"/api/transcripts/{tid}/speakers",
                       json={"old_label": "SPEAKER_00", "new_label": "Sam"})
    assert resp.json()["changed"] == 2  # lines a and c, not the pinned b

    rows = seg_rows(tid)
    assert [r[1] for r in rows] == ["Sam", "Maria", "Sam",
                                    "SPEAKER_01", "SPEAKER_01"]

    # A cluster rename of the pinned line's own name does not move it
    # either: pinned lines only change via per-line reassign.
    resp = client.post(f"/api/transcripts/{tid}/speakers",
                       json={"old_label": "Maria", "new_label": "Bob"})
    assert resp.json()["changed"] == 0
    assert seg_rows(tid)[1][1] == "Maria"


def test_merge_guard_requires_force(env, monkeypatch):
    client, _, tid = env
    monkeypatch.setattr(main.voices, "enroll_from_recording",
                        lambda *a: 0)
    # First rename: target name is new, no guard.
    assert client.post(
        f"/api/transcripts/{tid}/speakers",
        json={"old_label": "SPEAKER_00", "new_label": "Sam"}
    ).status_code == 200

    # Renaming another cluster onto the same name is a merge: refused
    # without force, and nothing changes.
    resp = client.post(f"/api/transcripts/{tid}/speakers",
                       json={"old_label": "SPEAKER_01", "new_label": "Sam"})
    assert resp.status_code == 409
    assert "merge" in resp.json()["detail"].lower()
    assert "reassign" in resp.json()["detail"]
    assert [r[1] for r in seg_rows(tid)] == \
        ["Sam", "Sam", "Sam", "SPEAKER_01", "SPEAKER_01"]

    # Explicit confirmation goes through.
    resp = client.post(
        f"/api/transcripts/{tid}/speakers",
        json={"old_label": "SPEAKER_01", "new_label": "Sam", "force": True})
    assert resp.status_code == 200
    assert resp.json()["changed"] == 2
    assert [r[1] for r in seg_rows(tid)] == ["Sam"] * 5


def test_auto_tag_never_moves_pinned_lines(env, monkeypatch):
    client, _, tid = env
    conn = db.connect()
    # Pin "line b" while keeping its generic label: a human said this
    # line is SPEAKER_00, so machine tagging must leave it alone.
    seg_id = conn.execute(
        "SELECT id FROM segments WHERE transcript_id = ? AND text = 'line b'",
        (tid,)).fetchone()[0]
    db.reassign_segment(conn, tid, seg_id, "SPEAKER_00")
    conn.execute(
        "INSERT INTO voiceprints (name, embedding, num_samples) "
        "VALUES ('Sam', ?, 1)",
        (np.ones(192, dtype=np.float32).tobytes(),))
    conn.commit()

    monkeypatch.setattr(voices, "_load_audio",
                        lambda path: np.zeros(160000, dtype=np.float32))
    monkeypatch.setattr(voices, "embed_segments",
                        lambda audio, segs: np.ones((len(segs), 192)))
    monkeypatch.setattr(voices, "match_speaker", lambda emb, vp: "Sam")

    tagged = voices.auto_tag(conn, tid, "/fake.m4a")
    conn.close()
    assert tagged == {"SPEAKER_00": "Sam", "SPEAKER_01": "Sam"}
    rows = seg_rows(tid)
    assert rows[0][1] == "Sam" and rows[0][2] == "SPEAKER_00"
    assert rows[1] == ("line b", "SPEAKER_00", None, 1)  # pinned, untouched
    assert rows[3][1] == "Sam"


def test_reset_speakers_restores_diarization_output(env, monkeypatch):
    client, rec_id, tid = env
    conn = db.connect()
    # Mangle the recording the way the incident did: names everywhere,
    # one machine tag, one pinned line, one stored rejection.
    conn.execute("UPDATE segments SET speaker = 'Felix' "
                 "WHERE transcript_id = ? AND speaker = 'SPEAKER_00'", (tid,))
    conn.execute("UPDATE segments SET speaker = 'Alex', "
                 "auto_original = 'SPEAKER_01' "
                 "WHERE transcript_id = ? AND speaker = 'SPEAKER_01'", (tid,))
    seg_id = conn.execute(
        "SELECT id FROM segments WHERE transcript_id = ? AND text = 'line c'",
        (tid,)).fetchone()[0]
    db.reassign_segment(conn, tid, seg_id, "Maria")
    conn.execute("INSERT INTO rejected_tags (transcript_id, label, name) "
                 "VALUES (?, 'SPEAKER_00', 'Bob')", (tid,))
    conn.commit()

    # The audio file must exist; diarization itself is faked.
    audio = conn.execute("SELECT file_path FROM recordings WHERE id = ?",
                         (rec_id,)).fetchone()[0]
    from pathlib import Path
    Path(audio).write_bytes(b"fake audio")

    fake_turns = [
        {"start": 0.0, "end": 6.0, "speaker": "SPEAKER_00"},
        {"start": 6.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    from app import diarize
    monkeypatch.setattr(diarize, "diarize", lambda path: fake_turns)
    retag_calls = []
    monkeypatch.setattr(voices, "auto_tag",
                        lambda c, t, p, only_names=None:
                        retag_calls.append(t) or {})

    result = voices.reset_speakers(conn, rec_id)
    assert result["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    rows = seg_rows(tid)
    assert [r[1] for r in rows] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00",
                                    "SPEAKER_01", "SPEAKER_01"]
    assert all(r[2] is None and r[3] == 0 for r in rows)
    assert conn.execute("SELECT count(*) FROM rejected_tags "
                        "WHERE transcript_id = ?", (tid,)).fetchone()[0] == 0
    assert retag_calls == [tid]
    conn.close()


def test_reset_endpoint(env, monkeypatch, tmp_path):
    client, rec_id, _ = env
    # Missing audio: refused.
    assert client.post(
        f"/api/recordings/{rec_id}/speakers/reset").status_code == 404
    assert client.post(
        "/api/recordings/999/speakers/reset").status_code == 404

    (tmp_path / "meet.m4a").write_bytes(b"fake audio")
    started = []
    monkeypatch.setattr(main, "start_speaker_reset",
                        lambda rid: started.append(rid)
                        or main.RESET_STATUS.__setitem__(rid, "running"))
    resp = client.post(f"/api/recordings/{rec_id}/speakers/reset")
    assert resp.json() == {"status": "running"}
    assert started == [rec_id]
    # While running, a second click does not start another reset.
    client.post(f"/api/recordings/{rec_id}/speakers/reset")
    assert started == [rec_id]
    assert client.get(
        f"/api/recordings/{rec_id}").json()["reset_status"] == "running"
    main.RESET_STATUS.clear()
