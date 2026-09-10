import pytest

from app import db


@pytest.fixture
def conn(tmp_path):
    c = db.connect(db_path=tmp_path / "test.db", key="ab" * 32)
    yield c
    c.close()


def add_transcript(conn, text, path="/tmp/a.m4a", digest=None):
    rec_id = db.insert_recording(conn, path, digest or path)
    return db.insert_transcript(conn, rec_id, text, "en", "tiny", [])


def test_search_finds_matching_transcript(conn):
    add_transcript(conn, "we discussed the quarterly budget", "/tmp/a.m4a")
    add_transcript(conn, "kickoff for the new website", "/tmp/b.m4a")

    results = db.search_transcripts(conn, "budget")
    assert len(results) == 1
    assert "<mark>budget</mark>" in results[0]["snippet"]


def test_search_returns_empty_for_no_match(conn):
    add_transcript(conn, "we discussed the quarterly budget")
    assert db.search_transcripts(conn, "zebra") == []


def test_search_is_case_insensitive(conn):
    add_transcript(conn, "we discussed the Quarterly Budget")
    assert len(db.search_transcripts(conn, "budget")) == 1


def test_search_treats_operators_as_plain_words(conn):
    add_transcript(conn, "planning AND budgeting session")
    # AND must not be parsed as an FTS operator or raise a syntax error.
    results = db.search_transcripts(conn, "planning AND budgeting")
    assert len(results) == 1


def test_index_stays_in_sync_on_update_and_delete(conn):
    tid = add_transcript(conn, "original text about llamas")
    conn.execute(
        "UPDATE transcripts SET full_text = ? WHERE id = ?",
        ("revised text about alpacas", tid),
    )
    conn.commit()
    assert db.search_transcripts(conn, "llamas") == []
    assert len(db.search_transcripts(conn, "alpacas")) == 1

    conn.execute("DELETE FROM segments WHERE transcript_id = ?", (tid,))
    conn.execute("DELETE FROM transcripts WHERE id = ?", (tid,))
    conn.commit()
    assert db.search_transcripts(conn, "alpacas") == []


def test_backfill_indexes_preexisting_transcripts(tmp_path):
    """Transcripts written before the FTS table existed must become
    searchable the first time the new schema runs."""
    import sqlcipher3

    path = tmp_path / "old.db"
    key = "ab" * 32
    raw = sqlcipher3.connect(str(path))
    raw.execute("PRAGMA key = \"x'%s'\"" % key)
    raw.executescript(
        """
        CREATE TABLE recordings (
            id INTEGER PRIMARY KEY, file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE, duration_seconds REAL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            status TEXT NOT NULL DEFAULT 'pending'
        );
        CREATE TABLE transcripts (
            id INTEGER PRIMARY KEY,
            recording_id INTEGER NOT NULL REFERENCES recordings(id),
            full_text TEXT NOT NULL, language TEXT, model TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE segments (
            id INTEGER PRIMARY KEY,
            transcript_id INTEGER NOT NULL REFERENCES transcripts(id),
            start_seconds REAL NOT NULL, end_seconds REAL NOT NULL,
            text TEXT NOT NULL, speaker TEXT
        );
        INSERT INTO recordings (file_path, file_hash) VALUES ('/tmp/x.m4a', 'h');
        INSERT INTO transcripts (recording_id, full_text)
        VALUES (1, 'ancient meeting about giraffes');
        """
    )
    raw.commit()
    raw.close()

    conn = db.connect(db_path=path, key=key)
    assert len(db.search_transcripts(conn, "giraffes")) == 1
    conn.close()


def test_summary_upsert_and_get(conn):
    tid = add_transcript(conn, "some meeting")
    assert db.get_summary(conn, tid) is None
    db.upsert_summary(conn, tid, '{"title": "First"}', "model-a")
    assert "First" in db.get_summary(conn, tid)
    db.upsert_summary(conn, tid, '{"title": "Second"}', "model-b")
    assert "Second" in db.get_summary(conn, tid)
    count = conn.execute("SELECT count(*) FROM summaries").fetchone()[0]
    assert count == 1


# Folder-scoped search. Until now /api/search had no folder_id at all,
# so searching while the list was filtered to a folder silently searched
# the whole archive and the extra hits looked like a bug in the filter.

def _folder_with(conn, name, recording_ids):
    from app import organize
    folder = organize.create_folder(conn, name)
    organize.assign_folder(conn, recording_ids, folder["id"])
    return folder


def test_transcript_search_scopes_to_one_folder(conn):
    from app import organize

    a = db.insert_recording(conn, "/tmp/a.m4a", "ha")
    db.insert_transcript(conn, a, "the budget for therapy", "en", "tiny", [])
    b = db.insert_recording(conn, "/tmp/b.m4a", "hb")
    db.insert_transcript(conn, b, "the budget for the mortgage", "en",
                         "tiny", [])
    folder = organize.create_folder(conn, "Therapy")
    organize.assign_folder(conn, [a], folder["id"])

    assert len(db.search_transcripts(conn, "budget")) == 2
    scoped = db.search_transcripts(conn, "budget", folder_id=folder["id"])
    assert [r["recording_id"] for r in scoped] == [a]


def test_notes_and_title_search_scope_to_one_folder(conn):
    from app import notes as user_notes
    from app import organize

    a = db.insert_recording(conn, "/tmp/a.m4a", "ha")
    b = db.insert_recording(conn, "/tmp/b.m4a", "hb")
    user_notes.save(conn, a, "remember the pricing question")
    user_notes.save(conn, b, "remember the pricing question")
    conn.execute("UPDATE recordings SET title = ? WHERE id = ?",
                 ("Pricing review", a))
    conn.execute("UPDATE recordings SET title = ? WHERE id = ?",
                 ("Pricing review", b))
    conn.commit()
    folder = organize.create_folder(conn, "Slopes")
    organize.assign_folder(conn, [b], folder["id"])

    assert len(db.search_notes(conn, "pricing")) == 2
    scoped = db.search_notes(conn, "pricing", folder_id=folder["id"])
    assert [r["recording_id"] for r in scoped] == [b]

    assert len(organize.search_titles(conn, "Pricing")) == 2
    scoped = organize.search_titles(conn, "Pricing", folder_id=folder["id"])
    assert [r["recording_id"] for r in scoped] == [b]


def test_unscoped_search_is_unchanged(conn):
    from app import organize

    a = db.insert_recording(conn, "/tmp/a.m4a", "ha")
    db.insert_transcript(conn, a, "quarterly budget", "en", "tiny", [])
    b = db.insert_recording(conn, "/tmp/b.m4a", "hb")
    db.insert_transcript(conn, b, "quarterly budget again", "en", "tiny", [])
    folder = organize.create_folder(conn, "Anything")
    organize.assign_folder(conn, [a], folder["id"])

    assert len(db.search_transcripts(conn, "budget")) == 2
    assert len(db.search_transcripts(conn, "budget", folder_id=None)) == 2


def test_search_endpoint_takes_folder_id(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app import main, organize

    path = tmp_path / "api.db"
    monkeypatch.setattr(dbmod.config, "DB_PATH", path)
    monkeypatch.setattr(dbmod, "get_or_create_key", lambda: "ab" * 32)

    c = dbmod.connect()
    a = dbmod.insert_recording(c, "/tmp/a.m4a", "ha")
    dbmod.insert_transcript(c, a, "the equity split", "en", "tiny", [])
    b = dbmod.insert_recording(c, "/tmp/b.m4a", "hb")
    dbmod.insert_transcript(c, b, "the equity vesting", "en", "tiny", [])
    folder = organize.create_folder(c, "Business")
    organize.assign_folder(c, [a], folder["id"])
    c.close()

    client = TestClient(main.app)
    assert len(client.get("/api/search?q=equity").json()) == 2
    scoped = client.get(
        f"/api/search?q=equity&folder_id={folder['id']}").json()
    assert [r["recording_id"] for r in scoped] == [a]
    assert client.get("/api/search?q=equity&folder_id=999").status_code == 404
