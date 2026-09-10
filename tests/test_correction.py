"""Correction UX v1: every line action stays line-scoped; undo works."""

import pytest
from fastapi.testclient import TestClient

from app import config, db, main


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "corr.db")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)
    conn = db.connect()
    rec = db.insert_recording(conn, str(tmp_path / "m.m4a"), "h1", 12.0)
    db.set_recording_status(conn, rec, "done")
    tid = db.insert_transcript(
        conn, rec, "full", "en", "tiny",
        [{"start": 0, "end": 2, "text": "a", "speaker": "SPEAKER_00"},
         {"start": 2, "end": 4, "text": "b", "speaker": "SPEAKER_00"},
         {"start": 4, "end": 6, "text": "c", "speaker": "SPEAKER_00"}])
    # Machine-tag all three lines as "Sam" (auto_original set).
    conn.execute("UPDATE segments SET speaker = 'Sam', "
                 "auto_original = 'SPEAKER_00' WHERE transcript_id = ?", (tid,))
    conn.commit()
    conn.close()
    return TestClient(main.app), rec, tid


def segs(client, rec):
    return client.get(f"/api/recordings/{rec}").json()["segments"]


def test_line_confirm_touches_only_that_line(env):
    client, rec, tid = env
    ids = [s["id"] for s in segs(client, rec)]
    resp = client.post(
        f"/api/transcripts/{tid}/segments/{ids[1]}/confirm")
    assert resp.status_code == 200
    rows = segs(client, rec)
    # Only line 2 promoted to human (auto False); others stay auto.
    assert [s["auto"] for s in rows] == [True, False, True]
    assert [s["speaker"] for s in rows] == ["Sam", "Sam", "Sam"]
    # Confirming a line with no auto-tag is a 404.
    assert client.post(
        f"/api/transcripts/{tid}/segments/{ids[1]}/confirm").status_code == 404


def test_line_revert_touches_only_that_line(env):
    client, rec, tid = env
    ids = [s["id"] for s in segs(client, rec)]
    resp = client.post(f"/api/transcripts/{tid}/segments/{ids[0]}/revert")
    assert resp.status_code == 200
    rows = segs(client, rec)
    # Only line 1 reverted to its SPEAKER_00 label; the cluster stays.
    assert [s["speaker"] for s in rows] == ["SPEAKER_00", "Sam", "Sam"]
    assert [s["auto"] for s in rows] == [False, True, True]
    # No rejection was recorded (line revert is not a group reject).
    conn = db.connect()
    assert conn.execute("SELECT count(*) FROM rejected_tags").fetchone()[0] == 0
    conn.close()


def test_multi_select_assign_splits_a_cluster(env):
    client, rec, tid = env
    ids = [s["id"] for s in segs(client, rec)]
    # Pull lines 1 and 3 out to "Maria"; line 2 stays Sam.
    resp = client.post(f"/api/transcripts/{tid}/segments/assign",
                       json={"ids": [ids[0], ids[2]], "name": "Maria"})
    assert resp.json() == {"assigned": 2, "speaker": "Maria"}
    rows = segs(client, rec)
    assert [s["speaker"] for s in rows] == ["Maria", "Sam", "Maria"]
    assert [s["pinned"] for s in rows] == [True, False, True]


def test_undo_restores_exact_prior_state(env):
    client, rec, tid = env
    before = segs(client, rec)
    ids = [s["id"] for s in before]

    # Snapshot the line we're about to change, then revert it.
    snap = client.get(f"/api/transcripts/{tid}/segments/states",
                      params={"ids": str(ids[0])}).json()
    assert snap == [{"id": ids[0], "speaker": "Sam",
                     "auto_original": "SPEAKER_00", "pinned": 0}]
    client.post(f"/api/transcripts/{tid}/segments/{ids[0]}/revert")
    assert segs(client, rec)[0]["speaker"] == "SPEAKER_00"

    # Undo puts the machine tag back exactly.
    resp = client.post(f"/api/transcripts/{tid}/segments/restore",
                       json={"segments": snap})
    assert resp.json() == {"restored": 1}
    after = segs(client, rec)
    assert after[0]["speaker"] == "Sam"
    assert after[0]["auto"] is True
    assert [s["speaker"] for s in after] == [s["speaker"] for s in before]


def test_undo_of_multi_assign(env):
    client, rec, tid = env
    ids = [s["id"] for s in segs(client, rec)]
    snap = client.get(f"/api/transcripts/{tid}/segments/states",
                      params={"ids": f"{ids[0]},{ids[2]}"}).json()
    client.post(f"/api/transcripts/{tid}/segments/assign",
                json={"ids": [ids[0], ids[2]], "name": "Maria"})
    client.post(f"/api/transcripts/{tid}/segments/restore",
                json={"segments": snap})
    rows = segs(client, rec)
    assert [s["speaker"] for s in rows] == ["Sam", "Sam", "Sam"]
    assert all(s["auto"] for s in rows)
    assert all(not s["pinned"] for s in rows)


def test_line_menu_reassign_still_pins_one_line(env):
    client, rec, tid = env
    ids = [s["id"] for s in segs(client, rec)]
    # The line-scoped menu uses the existing reassign endpoint.
    client.post(f"/api/transcripts/{tid}/segments/{ids[1]}/speaker",
                json={"name": "Maria"})
    rows = segs(client, rec)
    assert [s["speaker"] for s in rows] == ["Sam", "Maria", "Sam"]
    assert rows[1]["pinned"] is True
