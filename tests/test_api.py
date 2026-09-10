import pytest
from fastapi.testclient import TestClient

from app import config, db, main, summarize


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "api.db")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)

    conn = db.connect()
    rec_id = db.insert_recording(conn, "/tmp/standup.m4a", "hash-a", 65.0)
    db.set_recording_status(conn, rec_id, "done")
    tid = db.insert_transcript(
        conn, rec_id, "we decided to ship the project on friday", "en",
        "tiny",
        [{"start": 0.0, "end": 3.0, "text": "we decided",
          "speaker": "SPEAKER_00"},
         {"start": 3.0, "end": 5.0, "text": "to ship friday",
          "speaker": "SPEAKER_01"}],
    )
    conn.close()

    return TestClient(main.app), rec_id, tid


def test_list_recordings(client):
    c, rec_id, _ = client
    resp = c.get("/api/recordings")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["id"] == rec_id
    assert rows[0]["file_name"] == "standup.m4a"
    assert rows[0]["title"] is None
    assert rows[0]["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert rows[0]["topics"] == []


def test_audio_endpoint_serves_file(client, tmp_path, monkeypatch):
    c, rec_id, _ = client
    audio = tmp_path / "standup.m4a"
    audio.write_bytes(b"fake audio bytes")
    conn = db.connect()
    conn.execute("UPDATE recordings SET file_path = ? WHERE id = ?",
                 (str(audio), rec_id))
    conn.commit()
    conn.close()

    resp = c.get(f"/api/recordings/{rec_id}/audio")
    assert resp.status_code == 200
    assert resp.content == b"fake audio bytes"
    assert resp.headers["content-type"] == "audio/mp4"


def test_audio_endpoint_missing_file_is_404(client):
    c, rec_id, _ = client
    # file_path points at /tmp/standup.m4a which does not exist
    assert c.get(f"/api/recordings/{rec_id}/audio").status_code == 404
    assert c.get("/api/recordings/999/audio").status_code == 404


def test_get_recording_detail(client):
    c, rec_id, _ = client
    resp = c.get(f"/api/recordings/{rec_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert "ship the project" in body["full_text"]
    assert body["segments"][0]["text"] == "we decided"
    assert body["summary"] is None


def test_get_missing_recording_is_404(client):
    c, _, _ = client
    assert c.get("/api/recordings/999").status_code == 404


def test_search_endpoint(client):
    c, rec_id, _ = client
    resp = c.get("/api/search", params={"q": "project"})
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1
    assert results[0]["recording_id"] == rec_id
    assert "<mark>project</mark>" in results[0]["snippet"]

    assert c.get("/api/search", params={"q": "walrus"}).json() == []


def test_detail_includes_speaker_labels(client):
    c, rec_id, _ = client
    segments = c.get(f"/api/recordings/{rec_id}").json()["segments"]
    assert [s["speaker"] for s in segments] == ["SPEAKER_00", "SPEAKER_01"]


def test_rename_speaker(client, monkeypatch):
    c, rec_id, tid = client
    monkeypatch.setattr(
        main.voices, "enroll_from_recording",
        lambda conn, t, path, old, new: 3,
    )
    resp = c.post(
        f"/api/transcripts/{tid}/speakers",
        json={"old_label": "SPEAKER_00", "new_label": "Alexa"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"changed": 1, "enrolled_samples": 3,
                           "pinned_skipped": 0}

    segments = c.get(f"/api/recordings/{rec_id}").json()["segments"]
    assert [s["speaker"] for s in segments] == ["Alexa", "SPEAKER_01"]


def test_rename_between_real_names_does_not_enroll(client, monkeypatch):
    c, _, tid = client
    called = []
    monkeypatch.setattr(
        main.voices, "enroll_from_recording",
        lambda *a: called.append(a) or 1,
    )
    resp = c.post(
        f"/api/transcripts/{tid}/speakers",
        json={"old_label": "SPEAKER_00", "new_label": "SPEAKER_05"},
    )
    assert resp.status_code == 200
    assert resp.json()["enrolled_samples"] == 0
    assert called == []


def test_rename_speaker_rejects_blank_label(client):
    c, _, tid = client
    resp = c.post(
        f"/api/transcripts/{tid}/speakers",
        json={"old_label": "SPEAKER_00", "new_label": "   "},
    )
    assert resp.status_code == 422


def test_rename_speaker_missing_transcript_is_404(client):
    c, _, _ = client
    resp = c.post(
        "/api/transcripts/999/speakers",
        json={"old_label": "SPEAKER_00", "new_label": "Alexa"},
    )
    assert resp.status_code == 404


def test_calendar_status_defaults_to_fake(client):
    c, _, _ = client
    body = c.get("/api/calendar/status").json()
    assert body["provider"] == "fake"
    assert body["google_connected"] is False


def test_calendar_match_title_and_unlink(client, monkeypatch):
    from app import calendar_sync
    c, rec_id, _ = client
    conn = db.connect()
    conn.execute(
        "UPDATE recordings SET file_path = ?, duration_seconds = 1800 "
        "WHERE id = ?", ("/tmp/20250817 210558-AAAA.m4a", rec_id),
    )
    conn.commit()
    conn.close()

    provider = calendar_sync.FakeCalendarProvider([
        {"id": "ev1", "title": "Board Meeting",
         "start": "2025-08-17T21:00:00", "end": "2025-08-17T22:00:00",
         "attendees": ["Zach"]},
    ])
    monkeypatch.setattr(calendar_sync, "get_provider", lambda: provider)

    assert c.post("/api/calendar/match").json() == {"matched": 1}

    detail = c.get(f"/api/recordings/{rec_id}").json()
    assert detail["calendar_event"]["title"] == "Board Meeting"
    assert detail["calendar_event"]["attendees"] == ["Zach"]

    # The event title wins in the list view (auto-title).
    listed = c.get("/api/recordings").json()
    assert listed[0]["title"] == "Board Meeting"

    # Unlink removes it and it stays gone.
    assert c.post(f"/api/recordings/{rec_id}/calendar/unlink").status_code == 200
    assert c.get(f"/api/recordings/{rec_id}").json()["calendar_event"] is None
    c.post("/api/calendar/match")
    assert c.get(f"/api/recordings/{rec_id}").json()["calendar_event"] is None


def test_calendar_connect_without_credentials_is_503(client, monkeypatch):
    from app import calendar_sync
    c, _, _ = client
    monkeypatch.setattr(calendar_sync, "google_connected", lambda: False)
    resp = c.post("/api/calendar/connect")
    assert resp.status_code == 503
    assert "google_credentials.json" in resp.json()["detail"]


def test_sync_endpoint_reports_copied_count(client, monkeypatch):
    c, _, _ = client
    monkeypatch.setattr(main.sync_memos, "sync_voice_memos",
                        lambda *a, **k: 2)
    resp = c.post("/api/sync")
    assert resp.status_code == 200
    assert resp.json() == {"copied": 2}


def test_sync_endpoint_permission_error_is_503(client, monkeypatch):
    c, _, _ = client

    def denied(*a, **k):
        raise PermissionError("no full disk access")
    monkeypatch.setattr(main.sync_memos, "sync_voice_memos", denied)
    resp = c.post("/api/sync")
    assert resp.status_code == 503
    assert "Full Disk Access" in resp.json()["detail"]


def test_summarize_endpoint_stores_and_returns_summary(client, monkeypatch):
    c, rec_id, tid = client
    fake = summarize.MeetingSummary(
        title="Ship Decision", summary="Team agreed to ship Friday.",
    )
    monkeypatch.setattr(main.summarize, "summarize", lambda text, privileged=False: fake)

    resp = c.post(f"/api/recordings/{rec_id}/summarize")
    assert resp.status_code == 200
    assert resp.json()["title"] == "Ship Decision"

    # The summary is persisted and shows up in the detail view.
    detail = c.get(f"/api/recordings/{rec_id}").json()
    assert detail["summary"]["title"] == "Ship Decision"


def test_rename_starts_background_retag_sweep(client, monkeypatch):
    c, _, tid = client
    monkeypatch.setattr(
        main.voices, "enroll_from_recording", lambda *a: 3,
    )
    swept = []
    monkeypatch.setattr(main, "start_retag_sweep", swept.append)
    c.post(
        f"/api/transcripts/{tid}/speakers",
        json={"old_label": "SPEAKER_00", "new_label": "Alexa"},
    )
    assert swept == ["Alexa"]

    # No enrollment (rename between real names) means no sweep.
    c.post(
        f"/api/transcripts/{tid}/speakers",
        json={"old_label": "Alexa", "new_label": "Alexa S"},
    )
    assert swept == ["Alexa"]


def _enroll_and_autotag(tid):
    import numpy as np
    from app import voices
    conn = db.connect()
    vec = np.zeros(192, dtype=np.float32)
    vec[0] = 1.0
    voices.enroll(conn, "Zach", np.array([vec]))
    conn.execute(
        "UPDATE segments SET speaker = 'Zach', auto_original = speaker "
        "WHERE transcript_id = ? AND speaker = 'SPEAKER_00'",
        (tid,),
    )
    conn.commit()
    conn.close()


def test_voices_endpoint_lists_enrollments(client):
    c, _, tid = client
    _enroll_and_autotag(tid)
    listed = c.get("/api/voices").json()
    assert len(listed) == 1
    assert listed[0]["name"] == "Zach"
    assert listed[0]["recordings"] == 1
    assert listed[0]["segments"] == 1
    assert listed[0]["auto_segments"] == 1


def test_detail_marks_auto_tagged_segments(client):
    c, rec_id, tid = client
    _enroll_and_autotag(tid)
    segments = c.get(f"/api/recordings/{rec_id}").json()["segments"]
    assert [(s["speaker"], s["auto"]) for s in segments] == [
        ("Zach", True), ("SPEAKER_01", False),
    ]


def test_confirm_endpoint_promotes_tag(client):
    c, rec_id, tid = client
    _enroll_and_autotag(tid)
    resp = c.post(f"/api/transcripts/{tid}/speakers/confirm",
                  json={"label": "Zach"})
    assert resp.status_code == 200
    assert resp.json() == {"confirmed": "Zach", "segments": 1}
    segments = c.get(f"/api/recordings/{rec_id}").json()["segments"]
    assert segments[0] == dict(segments[0], speaker="Zach", auto=False)
    # Nothing left to confirm.
    assert c.post(f"/api/transcripts/{tid}/speakers/confirm",
                  json={"label": "Zach"}).status_code == 404


def test_reject_endpoint_reverts_tag(client):
    c, rec_id, tid = client
    _enroll_and_autotag(tid)
    resp = c.post(f"/api/transcripts/{tid}/speakers/reject",
                  json={"label": "Zach"})
    assert resp.status_code == 200
    segments = c.get(f"/api/recordings/{rec_id}").json()["segments"]
    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[0]["auto"] is False


def test_voice_rename_endpoint(client):
    c, rec_id, tid = client
    _enroll_and_autotag(tid)
    resp = c.post("/api/voices/Zach/rename", json={"new_name": "Zachary"})
    assert resp.status_code == 200
    assert resp.json() == {"name": "Zachary", "segments_changed": 1}
    assert c.get("/api/voices").json()[0]["name"] == "Zachary"
    segments = c.get(f"/api/recordings/{rec_id}").json()["segments"]
    assert segments[0]["speaker"] == "Zachary"

    assert c.post("/api/voices/Nobody/rename",
                  json={"new_name": "X"}).status_code == 404
    assert c.post("/api/voices/Zachary/rename",
                  json={"new_name": "SPEAKER_09"}).status_code == 422


def test_voice_delete_endpoint_reverts_auto_tags(client):
    c, rec_id, tid = client
    _enroll_and_autotag(tid)
    resp = c.request("DELETE", "/api/voices/Zach")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": "Zach", "segments_reverted": 1}
    assert c.get("/api/voices").json() == []
    segments = c.get(f"/api/recordings/{rec_id}").json()["segments"]
    assert segments[0]["speaker"] == "SPEAKER_00"

    assert c.request("DELETE", "/api/voices/Zach").status_code == 404


def test_rename_skipping_pinned_lines_reports_them(client, monkeypatch):
    """The silent no-op Alex hit: a speaker whose every line is pinned.

    The rename still succeeds, but changes nothing, so the response has
    to say how many pinned lines were left behind or the UI has no way
    to explain itself.
    """
    c, rec_id, tid = client
    monkeypatch.setattr(main.voices, "enroll_from_recording",
                        lambda *a: 0)
    conn = db.connect()
    conn.execute(
        "UPDATE segments SET pinned = 1 WHERE transcript_id = ? "
        "AND speaker = 'SPEAKER_00'", (tid,))
    conn.commit()
    conn.close()

    resp = c.post(
        f"/api/transcripts/{tid}/speakers",
        json={"old_label": "SPEAKER_00", "new_label": "Alexa"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["changed"] == 0
    assert body["pinned_skipped"] == 1
    # Unchanged on disk: the pinned line kept its human label.
    segments = c.get(f"/api/recordings/{rec_id}").json()["segments"]
    assert [s["speaker"] for s in segments] == ["SPEAKER_00", "SPEAKER_01"]


def test_rename_can_include_pinned_lines_when_asked(client, monkeypatch):
    c, rec_id, tid = client
    monkeypatch.setattr(main.voices, "enroll_from_recording",
                        lambda *a: 0)
    conn = db.connect()
    conn.execute(
        "UPDATE segments SET pinned = 1 WHERE transcript_id = ? "
        "AND speaker = 'SPEAKER_00'", (tid,))
    conn.commit()
    conn.close()

    resp = c.post(
        f"/api/transcripts/{tid}/speakers",
        json={"old_label": "SPEAKER_00", "new_label": "Alexa",
              "include_pinned": True},
    )
    assert resp.status_code == 200
    assert resp.json()["changed"] == 1
    segments = c.get(f"/api/recordings/{rec_id}").json()["segments"]
    assert [s["speaker"] for s in segments] == ["Alexa", "SPEAKER_01"]


def test_rename_onto_existing_name_is_409_then_force_merges(client,
                                                            monkeypatch):
    """The merge guard and the force path the UI offers after it."""
    c, rec_id, tid = client
    monkeypatch.setattr(main.voices, "enroll_from_recording",
                        lambda *a: 0)
    resp = c.post(
        f"/api/transcripts/{tid}/speakers",
        json={"old_label": "SPEAKER_00", "new_label": "SPEAKER_01"},
    )
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]
    # Nothing moved while the guard was refusing.
    segments = c.get(f"/api/recordings/{rec_id}").json()["segments"]
    assert [s["speaker"] for s in segments] == ["SPEAKER_00", "SPEAKER_01"]

    resp = c.post(
        f"/api/transcripts/{tid}/speakers",
        json={"old_label": "SPEAKER_00", "new_label": "SPEAKER_01",
              "force": True},
    )
    assert resp.status_code == 200
    assert resp.json()["changed"] == 1
    segments = c.get(f"/api/recordings/{rec_id}").json()["segments"]
    assert [s["speaker"] for s in segments] == ["SPEAKER_01", "SPEAKER_01"]
