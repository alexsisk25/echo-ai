from datetime import datetime

import pytest

from app import calendar_sync, db


def fake_provider(events):
    return calendar_sync.FakeCalendarProvider([
        {"id": e[0], "title": e[1], "start": e[2], "end": e[3],
         "attendees": e[4] if len(e) > 4 else []}
        for e in events
    ])


@pytest.fixture
def conn(tmp_path):
    c = db.connect(db_path=tmp_path / "test.db", key="ab" * 32)
    yield c
    c.close()


def add_recording(conn, file_name="20250817 210558-AAAA.m4a",
                  duration=1800.0, digest=None):
    rec_id = db.insert_recording(conn, f"/tmp/{file_name}",
                                 digest or file_name, duration)
    db.set_recording_status(conn, rec_id, "done")
    return rec_id


def test_recording_window_from_filename():
    start, end = calendar_sync.recording_window(
        "/tmp/20250817 210558-AAAA.m4a", "2026-07-07 14:00:00", 600.0
    )
    assert start == datetime(2025, 8, 17, 21, 5, 58)
    assert (end - start).total_seconds() == 600.0


def test_recording_window_falls_back_to_created_at():
    start, _ = calendar_sync.recording_window(
        "/tmp/demo.m4a", "2026-07-07 14:00:00", 60.0
    )
    assert start == datetime(2026, 7, 7, 14, 0, 0)


def test_fake_provider_filters_by_window():
    p = fake_provider([
        ("e1", "Standup", "2025-08-17T21:00:00", "2025-08-17T21:30:00"),
        ("e2", "Lunch", "2025-08-17T12:00:00", "2025-08-17T13:00:00"),
    ])
    events = p.list_events(datetime(2025, 8, 17, 20, 50),
                           datetime(2025, 8, 17, 22, 0))
    assert [e["id"] for e in events] == ["e1"]


def test_match_picks_best_overlap_and_stores(conn):
    rec = add_recording(conn)  # 21:05:58 to 21:35:58
    p = fake_provider([
        ("brief", "Quick chat", "2025-08-17T21:00:00", "2025-08-17T21:10:00"),
        ("main", "Board Meeting", "2025-08-17T21:00:00",
         "2025-08-17T22:00:00", ["Zach", "Maria"]),
    ])
    match = calendar_sync.match_recording(conn, rec, provider=p)
    assert match["title"] == "Board Meeting"
    assert match["attendees"] == ["Zach", "Maria"]
    assert calendar_sync.get_match(conn, rec)["event_id"] == "main"


def test_no_overlap_means_no_match(conn):
    rec = add_recording(conn)
    p = fake_provider([
        ("e", "Earlier", "2025-08-17T08:00:00", "2025-08-17T09:00:00"),
    ])
    assert calendar_sync.match_recording(conn, rec, provider=p) is None
    assert calendar_sync.get_match(conn, rec) is None


def test_unlink_dismisses_and_prevents_rematch(conn):
    rec = add_recording(conn)
    p = fake_provider([
        ("main", "Board Meeting", "2025-08-17T21:00:00",
         "2025-08-17T22:00:00"),
    ])
    calendar_sync.match_recording(conn, rec, provider=p)
    assert calendar_sync.unlink(conn, rec) is True
    assert calendar_sync.get_match(conn, rec) is None
    # A rematch attempt must respect the dismissal.
    assert calendar_sync.match_recording(conn, rec, provider=p) is None
    assert calendar_sync.get_match(conn, rec) is None


def test_match_all_skips_already_matched(conn):
    rec1 = add_recording(conn, "20250817 210558-AAAA.m4a")
    rec2 = add_recording(conn, "20250818 100000-BBBB.m4a")
    p = fake_provider([
        ("m1", "Meeting One", "2025-08-17T21:00:00", "2025-08-17T22:00:00"),
        ("m2", "Meeting Two", "2025-08-18T10:00:00", "2025-08-18T11:00:00"),
    ])
    assert calendar_sync.match_all(conn, provider=p) == 2
    assert calendar_sync.match_all(conn, provider=p) == 0
    assert calendar_sync.get_match(conn, rec1)["title"] == "Meeting One"
    assert calendar_sync.get_match(conn, rec2)["title"] == "Meeting Two"


def test_timezone_aware_events_match_naive_recordings(conn):
    rec = add_recording(conn)
    p = fake_provider([
        ("g", "Google Event", "2025-08-17T21:00:00-06:00",
         "2025-08-17T22:00:00-06:00"),
    ])
    # Wall-clock overlap is what matters; the tz-aware event must not crash
    # or silently fail against the naive filename timestamp.
    match = calendar_sync.match_recording(conn, rec, provider=p)
    assert match is not None


def test_google_connected_means_authorized_not_configured(monkeypatch,
                                                          tmp_path):
    creds_file = tmp_path / "google_credentials.json"
    token_file = tmp_path / "google_token.json"
    monkeypatch.setattr(calendar_sync, "GOOGLE_CREDENTIALS", creds_file)
    monkeypatch.setattr(calendar_sync, "GOOGLE_TOKEN", token_file)

    # Nothing on disk: not connected.
    assert calendar_sync.google_connected() is False

    # Credentials file alone (the pre-fix bug): still NOT connected;
    # the OAuth flow has never run, so the Connect button must show.
    creds_file.write_text('{"installed": {"client_id": "x"}}')
    assert calendar_sync.google_connected() is False

    # A garbage token does not count as authorized.
    token_file.write_text("not json at all")
    assert calendar_sync.google_connected() is False

    # A real-shaped token with a refresh token means authorized.
    token_file.write_text(
        '{"token": "ya29.x", "refresh_token": "1//r", '
        '"client_id": "x", "client_secret": "y", '
        '"scopes": ["https://www.googleapis.com/auth/calendar.readonly"], '
        '"universe_domain": "googleapis.com", "account": "", '
        '"expiry": "2020-01-01T00:00:00Z"}'
    )
    assert calendar_sync.google_connected() is True


def test_connect_endpoint_guides_when_credentials_missing(monkeypatch,
                                                          tmp_path):
    from fastapi.testclient import TestClient

    from app import config, db, main

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "cal.db")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)
    client = TestClient(main.app)

    # No credentials file: 503 explains what to do (does NOT depend on
    # authorization state).
    resp = client.post("/api/calendar/connect")
    assert resp.status_code == 503
    assert "google_credentials.json" in resp.json()["detail"]

    # Status reflects authorization, so the UI keeps showing Connect.
    body = client.get("/api/calendar/status").json()
    assert body["google_connected"] is False
    assert body["provider"] == "fake"


def test_starting_event_finds_current_meeting_with_attendees():
    from datetime import timedelta
    now = datetime(2026, 7, 25, 9, 0, 0)
    p = fake_provider([
        # Starts in 2 minutes, has attendees: this is the one.
        ("m1", "Standup", "2026-07-25T09:02:00", "2026-07-25T09:30:00",
         ["Sam", "Alex"]),
        # No attendees: skipped.
        ("m2", "Focus block", "2026-07-25T09:01:00", "2026-07-25T10:00:00",
         []),
    ])
    ev = calendar_sync.starting_event(provider=p, now=now)
    assert ev is not None
    assert ev["id"] == "m1"
    assert ev["attendees"] == ["Sam", "Alex"]


def test_starting_event_none_when_nothing_imminent():
    now = datetime(2026, 7, 25, 9, 0, 0)
    p = fake_provider([
        # Starts in 2 hours: outside the window.
        ("m1", "Later", "2026-07-25T11:00:00", "2026-07-25T12:00:00",
         ["Sam"]),
        # Already ended.
        ("m2", "Past", "2026-07-25T08:00:00", "2026-07-25T08:30:00",
         ["Sam"]),
    ])
    assert calendar_sync.starting_event(provider=p, now=now) is None


def test_starting_event_picks_nearest_start():
    now = datetime(2026, 7, 25, 9, 0, 0)
    p = fake_provider([
        ("far", "Farther", "2026-07-25T09:04:00", "2026-07-25T09:30:00",
         ["Sam"]),
        ("near", "Nearer", "2026-07-25T09:01:00", "2026-07-25T09:30:00",
         ["Sam"]),
    ])
    assert calendar_sync.starting_event(provider=p, now=now)["id"] == "near"


def test_starting_endpoint(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app import config, main
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "s.db")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)
    monkeypatch.setattr(calendar_sync, "get_provider",
                        lambda: fake_provider([]))
    client = TestClient(main.app)
    assert client.get("/api/calendar/starting").json() == {"event": None}


# ---------- Ambiguity, moved events, and the chooser ----------
# PLAN.md Phase 3 asks the matcher to "code defensively for moved or
# duplicate events, and fall back to asking me when ambiguous". Until
# now it silently took the best overlap, whatever the runner-up looked
# like, and never looked at a stored match again.

def test_near_tie_asks_instead_of_guessing(conn):
    rec = add_recording(conn)  # 21:05:58 to 21:35:58
    p = fake_provider([
        ("a", "Board Meeting", "2025-08-17T21:00:00",
         "2025-08-17T21:30:00"),
        ("b", "Investor Call", "2025-08-17T21:05:00",
         "2025-08-17T21:33:00"),
    ])
    assert calendar_sync.match_recording(conn, rec, provider=p) is None
    assert calendar_sync.get_match(conn, rec) is None
    candidates = calendar_sync.get_candidates(conn, rec)
    assert {c["id"] for c in candidates} == {"a", "b"}
    # Nothing was auto-titled from a coin toss.
    assert conn.execute(
        "SELECT COUNT(*) FROM calendar_matches WHERE recording_id = ? "
        "AND status = 'matched'", (rec,)).fetchone()[0] == 0


def test_a_clear_winner_still_matches_without_asking(conn):
    rec = add_recording(conn)
    p = fake_provider([
        ("main", "Board Meeting", "2025-08-17T21:00:00",
         "2025-08-17T21:35:00"),
        ("sliver", "Quick chat", "2025-08-17T21:05:00",
         "2025-08-17T21:07:00"),
    ])
    match = calendar_sync.match_recording(conn, rec, provider=p)
    assert match["event_id"] == "main"
    assert calendar_sync.get_candidates(conn, rec) == []


def test_identical_titles_are_always_ambiguous(conn):
    """Two instances of one recurring meeting cannot be told apart by
    overlap, so even a clear overlap winner is not trusted."""
    rec = add_recording(conn)
    p = fake_provider([
        ("first", "Weekly 1:1", "2025-08-17T21:00:00",
         "2025-08-17T21:35:00"),
        ("second", "Weekly 1:1", "2025-08-17T21:30:00",
         "2025-08-17T21:32:00"),
    ])
    assert calendar_sync.match_recording(conn, rec, provider=p) is None
    assert {c["id"] for c in calendar_sync.get_candidates(conn, rec)} == {
        "first", "second"}


def test_choosing_one_candidate_stores_it_as_a_normal_match(conn):
    rec = add_recording(conn)
    p = fake_provider([
        ("a", "Board Meeting", "2025-08-17T21:00:00", "2025-08-17T21:30:00",
         ["Sam"]),
        ("b", "Investor Call", "2025-08-17T21:05:00", "2025-08-17T21:33:00",
         ["Dana"]),
    ])
    calendar_sync.match_recording(conn, rec, provider=p)
    match = calendar_sync.choose_event(conn, rec, "b")
    assert match["event_id"] == "b"
    assert match["title"] == "Investor Call"
    assert match["attendees"] == ["Dana"]
    assert calendar_sync.get_candidates(conn, rec) == []
    # And the answer sticks: matching again does not re-open the question.
    assert calendar_sync.match_recording(
        conn, rec, provider=p)["event_id"] == "b"


def test_choosing_neither_dismisses_and_is_not_asked_again(conn):
    rec = add_recording(conn)
    p = fake_provider([
        ("a", "Board Meeting", "2025-08-17T21:00:00", "2025-08-17T21:30:00"),
        ("b", "Investor Call", "2025-08-17T21:05:00", "2025-08-17T21:33:00"),
    ])
    calendar_sync.match_recording(conn, rec, provider=p)
    assert calendar_sync.choose_event(conn, rec, None) is None
    assert calendar_sync.get_candidates(conn, rec) == []
    assert calendar_sync.match_recording(conn, rec, provider=p) is None
    assert calendar_sync.get_candidates(conn, rec) == []


def test_choosing_needs_a_pending_question_and_a_real_candidate(conn):
    rec = add_recording(conn)
    with pytest.raises(LookupError):
        calendar_sync.choose_event(conn, rec, "a")
    p = fake_provider([
        ("a", "Board Meeting", "2025-08-17T21:00:00", "2025-08-17T21:30:00"),
        ("b", "Investor Call", "2025-08-17T21:05:00", "2025-08-17T21:33:00"),
    ])
    calendar_sync.match_recording(conn, rec, provider=p)
    with pytest.raises(LookupError):
        calendar_sync.choose_event(conn, rec, "not-a-candidate")


def test_a_moved_event_is_re_evaluated(conn):
    rec = add_recording(conn)
    before = fake_provider([
        ("main", "Board Meeting", "2025-08-17T21:00:00",
         "2025-08-17T21:35:00"),
    ])
    assert calendar_sync.match_recording(
        conn, rec, provider=before)["event_id"] == "main"

    # The organizer moved it out of the recording's window, and a
    # different meeting now covers that time.
    after = fake_provider([
        ("main", "Board Meeting", "2025-08-18T21:00:00",
         "2025-08-18T21:35:00"),
        ("other", "Vendor Sync", "2025-08-17T21:00:00",
         "2025-08-17T21:35:00"),
    ])
    match = calendar_sync.match_recording(conn, rec, provider=after)
    assert match["event_id"] == "other"
    assert match["title"] == "Vendor Sync"


def test_an_unchanged_event_is_left_alone(conn):
    rec = add_recording(conn)
    p = fake_provider([
        ("main", "Board Meeting", "2025-08-17T21:00:00",
         "2025-08-17T21:35:00"),
    ])
    calendar_sync.match_recording(conn, rec, provider=p)
    stored = conn.execute(
        "SELECT event_start, created_at FROM calendar_matches "
        "WHERE recording_id = ?", (rec,)).fetchone()
    assert calendar_sync.match_recording(
        conn, rec, provider=p)["event_id"] == "main"
    assert conn.execute(
        "SELECT event_start, created_at FROM calendar_matches "
        "WHERE recording_id = ?", (rec,)).fetchone() == stored


def test_an_unreadable_calendar_never_unpicks_a_good_match(conn):
    rec = add_recording(conn)
    p = fake_provider([
        ("main", "Board Meeting", "2025-08-17T21:00:00",
         "2025-08-17T21:35:00"),
    ])
    calendar_sync.match_recording(conn, rec, provider=p)

    class Broken:
        def list_events(self, start, end):
            raise RuntimeError("calendar unreachable")

    assert calendar_sync.match_recording(
        conn, rec, provider=Broken())["event_id"] == "main"


def test_starting_event_names_both_when_two_start_together():
    now = datetime(2026, 9, 9, 10, 0, 0)
    p = fake_provider([
        ("a", "Standup", "2026-09-09T10:00:00", "2026-09-09T10:30:00",
         ["Sam"]),
        ("b", "Design Review", "2026-09-09T10:00:30", "2026-09-09T10:45:00",
         ["Dana"]),
    ])
    out = calendar_sync.starting_event(provider=p, now=now)
    assert out["ambiguous"] is True
    assert [o["id"] for o in out["others"]] == ["b"]


def test_starting_event_stays_single_when_one_is_clearly_nearer():
    now = datetime(2026, 9, 9, 10, 0, 0)
    p = fake_provider([
        ("a", "Standup", "2026-09-09T10:00:00", "2026-09-09T10:30:00",
         ["Sam"]),
        ("b", "Design Review", "2026-09-09T10:04:00", "2026-09-09T10:45:00",
         ["Dana"]),
    ])
    out = calendar_sync.starting_event(provider=p, now=now)
    assert out["id"] == "a"
    assert "ambiguous" not in out
    assert "others" not in out


def test_choose_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app import main

    monkeypatch.setattr(dbmod.config, "DB_PATH", tmp_path / "cal.db")
    monkeypatch.setattr(dbmod, "get_or_create_key", lambda: "ab" * 32)
    c = dbmod.connect()
    rec = add_recording(c)
    p = fake_provider([
        ("a", "Board Meeting", "2025-08-17T21:00:00", "2025-08-17T21:30:00"),
        ("b", "Investor Call", "2025-08-17T21:05:00", "2025-08-17T21:33:00"),
    ])
    calendar_sync.match_recording(c, rec, provider=p)
    c.close()

    client = TestClient(main.app)
    detail = client.get(f"/api/recordings/{rec}").json()
    assert detail["calendar_event"] is None
    assert {x["id"] for x in detail["calendar_candidates"]} == {"a", "b"}

    resp = client.post(f"/api/recordings/{rec}/calendar/choose",
                       json={"event_id": "a"})
    assert resp.status_code == 200
    assert resp.json()["calendar_event"]["title"] == "Board Meeting"
    detail = client.get(f"/api/recordings/{rec}").json()
    assert detail["calendar_candidates"] == []

    # Nothing pending any more, so the endpoint says so rather than
    # inventing a match.
    assert client.post(f"/api/recordings/{rec}/calendar/choose",
                       json={"event_id": "b"}).status_code == 404
