"""Organization: manual titles, folders, suggestions, search, export."""

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from app import config, db, main, organize, summarize

SUMMARY = {"title": "Q3 Budget Review", "date": "2026-07-18",
           "attendees": [], "summary": "Reviewed the Q3 budget with Acme.",
           "decisions": [], "action_items": [], "risks": [],
           "topics": ["budget", "acme"]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "org.db")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)

    conn = db.connect()
    rec = db.insert_recording(conn, str(tmp_path / "a.m4a"), "h1", 60.0)
    db.set_recording_status(conn, rec, "done")
    tid = db.insert_transcript(conn, rec, "budget talk", "en", "tiny",
                               [{"start": 0, "end": 2, "text": "budget talk",
                                 "speaker": "SPEAKER_00"}])
    db.upsert_summary(conn, tid, json.dumps(SUMMARY), model="test")
    rec2 = db.insert_recording(conn, str(tmp_path / "b.m4a"), "h2", 30.0)
    db.set_recording_status(conn, rec2, "done")
    conn.close()
    return TestClient(main.app), rec, rec2


def test_manual_title_is_pinned_over_calendar_and_summary(env):
    client, rec, _ = env
    # AI title shows by default.
    rows = client.get("/api/recordings").json()
    row = next(r for r in rows if r["id"] == rec)
    assert row["title"] == "Q3 Budget Review"
    assert row["title_is_manual"] is False

    # A calendar match would normally win; store one.
    conn = db.connect()
    conn.execute(
        "INSERT INTO calendar_matches (recording_id, event_id, event_title, "
        "event_start, event_end) VALUES (?, 'ev', 'Calendar Title', "
        "'2026-07-18T09:00:00', '2026-07-18T10:00:00')", (rec,))
    conn.commit()
    conn.close()
    row = next(r for r in client.get("/api/recordings").json()
               if r["id"] == rec)
    assert row["title"] == "Calendar Title"

    # The manual title beats both, and survives a calendar re-match.
    assert client.put(f"/api/recordings/{rec}/title",
                      json={"title": "My Own Name"}).status_code == 200
    row = next(r for r in client.get("/api/recordings").json()
               if r["id"] == rec)
    assert row["title"] == "My Own Name"
    assert row["title_is_manual"] is True

    conn = db.connect()
    conn.execute("UPDATE calendar_matches SET event_title = 'Re-Matched' "
                 "WHERE recording_id = ?", (rec,))
    conn.commit()
    conn.close()
    row = next(r for r in client.get("/api/recordings").json()
               if r["id"] == rec)
    assert row["title"] == "My Own Name"

    # Clearing with an empty string falls back to the calendar title.
    client.put(f"/api/recordings/{rec}/title", json={"title": "  "})
    row = next(r for r in client.get("/api/recordings").json()
               if r["id"] == rec)
    assert row["title"] == "Re-Matched"
    assert client.put("/api/recordings/999/title",
                      json={"title": "x"}).status_code == 404


def test_folder_crud_and_assignment(env):
    client, rec, rec2 = env
    made = client.post("/api/folders", json={"name": "Client calls"}).json()
    assert made["existing"] is False
    # Same name again returns the existing folder.
    again = client.post("/api/folders", json={"name": "Client calls"}).json()
    assert again["id"] == made["id"] and again["existing"] is True
    assert client.post("/api/folders", json={"name": " "}).status_code == 422

    assert client.put(f"/api/recordings/{rec}/folder",
                      json={"folder_id": made["id"]}).status_code == 200
    row = next(r for r in client.get("/api/recordings").json()
               if r["id"] == rec)
    assert row["folder"] == "Client calls"

    folders = client.get("/api/folders").json()
    assert folders == [{"id": made["id"], "name": "Client calls", "count": 1,
                        "privileged": False}]

    # Rename, duplicate-name conflict, bulk assign, unfile.
    other = client.post("/api/folders", json={"name": "Internal"}).json()
    assert client.put(f"/api/folders/{other['id']}",
                      json={"name": "Client calls"}).status_code == 409
    assert client.put(f"/api/folders/{other['id']}",
                      json={"name": "Team"}).status_code == 200
    assert client.post("/api/recordings/bulk-folder",
                       json={"ids": [rec, rec2],
                             "folder_id": other["id"]}).json() == {"filed": 2}
    assert client.put(f"/api/recordings/{rec}/folder",
                      json={"folder_id": None}).status_code == 200
    row = next(r for r in client.get("/api/recordings").json()
               if r["id"] == rec)
    assert row["folder"] is None
    assert client.put(f"/api/recordings/{rec}/folder",
                      json={"folder_id": 999}).status_code == 404


def test_deleting_a_folder_unfiles_but_keeps_recordings(env):
    client, rec, rec2 = env
    made = client.post("/api/folders", json={"name": "Temp"}).json()
    client.post("/api/recordings/bulk-folder",
                json={"ids": [rec, rec2], "folder_id": made["id"]})
    resp = client.delete(f"/api/folders/{made['id']}")
    assert resp.json() == {"deleted": made["id"], "unfiled": 2}
    rows = client.get("/api/recordings").json()
    assert len(rows) == 2
    assert all(r["folder"] is None for r in rows)
    assert client.get("/api/folders").json() == []
    assert client.delete("/api/folders/999").status_code == 404


def test_suggestions_never_assign_without_accept(env, monkeypatch):
    client, rec, rec2 = env
    monkeypatch.setattr(
        summarize, "_call_llm",
        lambda prompt, privileged=False: '{"folder": "Client calls"}')

    got = client.post("/api/folders/suggest", json={}).json()
    # Only rec has a summary; rec2 cannot be classified.
    assert got == {"suggestions": {str(rec): "Client calls"}} or \
        got == {"suggestions": {rec: "Client calls"}}
    row = next(r for r in client.get("/api/recordings").json()
               if r["id"] == rec)
    assert row["suggested_folder"] == "Client calls"
    assert row["folder"] is None  # never auto-assigned

    # Accept files it, creating the folder on the fly.
    accepted = client.post(
        f"/api/recordings/{rec}/suggestion/accept").json()
    assert accepted["name"] == "Client calls"
    row = next(r for r in client.get("/api/recordings").json()
               if r["id"] == rec)
    assert row["folder"] == "Client calls"
    assert row["suggested_folder"] is None
    # No pending suggestion anymore.
    assert client.post(
        f"/api/recordings/{rec}/suggestion/accept").status_code == 404


def test_dismissed_suggestions_stay_dismissed(env, monkeypatch):
    client, rec, _ = env
    calls = []
    monkeypatch.setattr(
        summarize, "_call_llm",
        lambda prompt, privileged=False: calls.append(1)
        or '{"folder": "Internal"}')
    client.post("/api/folders/suggest", json={})
    assert client.post(
        f"/api/recordings/{rec}/suggestion/dismiss").status_code == 200
    row = next(r for r in client.get("/api/recordings").json()
               if r["id"] == rec)
    assert row["suggested_folder"] is None
    # A blanket re-run skips dismissed recordings.
    client.post("/api/folders/suggest", json={})
    assert len(calls) == 1
    # But an explicit request for this recording asks again.
    client.post("/api/folders/suggest", json={"ids": [rec]})
    assert len(calls) == 2


def test_suggestions_route_privileged_to_local(env, monkeypatch):
    client, rec, _ = env
    seen = []
    monkeypatch.setattr(
        summarize, "_call_llm",
        lambda prompt, privileged=False: seen.append(privileged)
        or '{"folder": "Personal"}')
    client.post(f"/api/recordings/{rec}/privileged",
                json={"privileged": True})
    client.post("/api/folders/suggest", json={"ids": [rec]})
    assert seen == [True]


def test_accept_all_suggestions(env, monkeypatch):
    client, rec, rec2 = env
    conn = db.connect()
    tid2 = db.insert_transcript(conn, rec2, "interview chat", "en", "tiny",
                                [{"start": 0, "end": 2, "text": "hi",
                                  "speaker": None}])
    db.upsert_summary(conn, tid2, json.dumps(
        {**SUMMARY, "title": "Interview"}), model="test")
    conn.close()
    answers = iter(['{"folder": "Clients"}', '{"folder": "Interviews"}'])
    monkeypatch.setattr(summarize, "_call_llm",
                        lambda prompt, privileged=False: next(answers))
    client.post("/api/folders/suggest", json={})
    assert client.post("/api/folders/accept-all").json() == {"filed": 2}
    rows = client.get("/api/recordings").json()
    assert sorted(r["folder"] for r in rows) == ["Clients", "Interviews"]


def test_titles_are_searchable(env):
    client, rec, _ = env
    client.put(f"/api/recordings/{rec}/title",
               json={"title": "Zephyr Kickoff"})
    hits = client.get("/api/search", params={"q": "Zephyr"}).json()
    assert any(h.get("source") == "title" and h["recording_id"] == rec
               for h in hits)
    # AI titles are searchable too (rec's summary title).
    hits = client.get("/api/search", params={"q": "Budget Review"}).json()
    assert not any(h.get("source") == "title" for h in hits)  # manual wins
    client.put(f"/api/recordings/{rec}/title", json={"title": ""})
    hits = client.get("/api/search", params={"q": "Budget Review"}).json()
    assert any(h.get("source") == "title" for h in hits)


def test_export_nests_by_folder_and_uses_manual_title(env, tmp_path):
    client, rec, rec2 = env
    (tmp_path / "a.m4a").write_bytes(b"audio-a")
    (tmp_path / "b.m4a").write_bytes(b"audio-b")
    made = client.post("/api/folders", json={"name": "Client calls"}).json()
    client.put(f"/api/recordings/{rec}/folder",
               json={"folder_id": made["id"]})
    client.put(f"/api/recordings/{rec}/title", json={"title": "Acme Sync"})

    resp = client.get("/api/export", params={"ids": f"{rec},{rec2}"})
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    assert f"Client-calls/{rec:03d}-Acme-Sync/transcript.md" in names
    assert f"Client-calls/{rec:03d}-Acme-Sync/a.m4a" in names
    # Unfiled recording sits at the zip root.
    assert f"{rec2:03d}-b/b.m4a" in names

    # Single export stays flat and uses the manual title for the name.
    single = client.get("/api/export", params={"ids": str(rec)})
    assert "Acme-Sync.zip" in single.headers["content-disposition"]


def test_export_name_falls_back_to_calendar_title(env, tmp_path):
    client, _, rec2 = env
    (tmp_path / "b.m4a").write_bytes(b"audio-b")
    # rec2 has no manual title and no summary: the calendar event
    # title should name its export, matching the list display.
    conn = db.connect()
    conn.execute(
        "INSERT INTO calendar_matches (recording_id, event_id, event_title, "
        "event_start, event_end) VALUES (?, 'ev', 'Standup With Acme', "
        "'2026-07-18T09:00:00', '2026-07-18T10:00:00')", (rec2,))
    conn.commit()
    conn.close()
    single = client.get("/api/export", params={"ids": str(rec2)})
    assert "Standup-With-Acme.zip" in single.headers["content-disposition"]


def test_delete_recording_leaves_folder_intact(env, tmp_path):
    client, rec, _ = env
    (tmp_path / "a.m4a").write_bytes(b"audio-a")
    made = client.post("/api/folders", json={"name": "Keepers"}).json()
    client.put(f"/api/recordings/{rec}/folder",
               json={"folder_id": made["id"]})
    assert client.delete(f"/api/recordings/{rec}").status_code == 200
    assert client.get("/api/folders").json() == \
        [{"id": made["id"], "name": "Keepers", "count": 0,
          "privileged": False}]
