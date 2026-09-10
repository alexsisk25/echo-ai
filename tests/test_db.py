import hashlib
import os
import sqlite3
import stat

import pytest

from app import db


@pytest.fixture
def conn(tmp_path):
    c = db.connect(db_path=tmp_path / "test.db", key="ab" * 32)
    yield c
    c.close()


def test_insert_and_read_back(conn):
    rec_id = db.insert_recording(conn, "/tmp/a.m4a", "hash1", 12.5)
    tid = db.insert_transcript(
        conn, rec_id, "hello world", "en", "tiny",
        [{"start": 0.0, "end": 1.2, "text": "hello"},
         {"start": 1.2, "end": 2.5, "text": "world"}],
    )
    text = conn.execute(
        "SELECT full_text FROM transcripts WHERE id = ?", (tid,)
    ).fetchone()[0]
    assert text == "hello world"
    segs = conn.execute(
        "SELECT text FROM segments WHERE transcript_id = ? "
        "ORDER BY start_seconds", (tid,)
    ).fetchall()
    assert [s[0] for s in segs] == ["hello", "world"]


def test_duplicate_detection(conn):
    db.insert_recording(conn, "/tmp/a.m4a", "samehash")
    assert db.recording_exists(conn, "samehash")
    assert not db.recording_exists(conn, "otherhash")


def test_status_transitions(conn):
    rec_id = db.insert_recording(conn, "/tmp/a.m4a", "h")
    db.set_recording_status(conn, rec_id, "done")
    status = conn.execute(
        "SELECT status FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()[0]
    assert status == "done"


def test_file_is_encrypted_and_owner_only(tmp_path):
    path = tmp_path / "enc.db"
    c = db.connect(db_path=path, key="cd" * 32)
    db.insert_recording(c, "/tmp/a.m4a", "h")
    c.close()

    # Plain sqlite3 without the key must fail to read it.
    plain = sqlite3.connect(str(path))
    with pytest.raises(sqlite3.DatabaseError):
        plain.execute("SELECT count(*) FROM recordings").fetchone()
    plain.close()

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600


def test_wrong_key_rejected(tmp_path):
    path = tmp_path / "enc.db"
    c = db.connect(db_path=path, key="ab" * 32)
    c.close()
    import sqlcipher3
    with pytest.raises(sqlcipher3.DatabaseError):
        bad = sqlcipher3.connect(str(path))
        bad.execute("PRAGMA key = \"x'%s'\"" % ("ff" * 32))
        bad.execute("SELECT count(*) FROM sqlite_master").fetchone()


# PLAN.md's security rules cover the journal files as well as the
# database, and a chmod that only ran at creation covered neither after
# the first run.

def test_journal_mode_is_wal(tmp_path):
    path = tmp_path / "wal.db"
    c = db.connect(db_path=path, key="ab" * 32)
    try:
        mode = c.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        c.close()
    assert mode.lower() == "wal"


def test_loosened_permissions_are_retightened_on_connect(tmp_path):
    path = tmp_path / "perm.db"
    # Held open on purpose: SQLite checkpoints and removes the -wal and
    # -shm siblings when the last connection closes, so the only way to
    # prove they get covered is to connect again while they exist.
    holder = db.connect(db_path=path, key="ab" * 32)
    db.insert_recording(holder, "/tmp/a.m4a", "h")
    try:
        siblings = [path] + [
            tmp_path / ("perm.db" + s) for s in db.JOURNAL_SUFFIXES
        ]
        present = [p for p in siblings if p.exists()]
        assert path in present
        assert any(str(p).endswith("-wal") for p in present)
        assert any(str(p).endswith("-shm") for p in present)

        # Loosened the way a stray umask or a file copy would leave them.
        for p in present:
            os.chmod(p, 0o644)
            assert stat.S_IMODE(p.stat().st_mode) == 0o644

        second = db.connect(db_path=path, key="ab" * 32)
        second.close()

        for p in present:
            assert stat.S_IMODE(p.stat().st_mode) == 0o600, f"{p} still loose"
    finally:
        holder.close()


def test_tighten_permissions_skips_missing_siblings(tmp_path):
    path = tmp_path / "only.db"
    path.write_bytes(b"")
    os.chmod(path, 0o644)
    touched = db.tighten_permissions(path)
    assert touched == [path]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def _seed_audio(folder, name, body: bytes):
    """Write a file and return it with the hash the recordings row holds."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(body)
    return path, hashlib.sha256(body).hexdigest()


def test_moved_project_repoints_recordings(conn, tmp_path, monkeypatch):
    """Moving the project folder must not leave every recording dangling.

    This is the bug the local-otter rename and the move off iCloud both
    caused: file_path is absolute, so after a move nothing can play,
    export, reprocess, enroll a voice, or delete its audio.
    """
    inbox = tmp_path / "inbox"
    moved, digest = _seed_audio(inbox, "20250626 140107.m4a", b"real audio")
    monkeypatch.setattr(db.config, "INBOX_DIR", inbox)
    monkeypatch.setattr(db.config, "BACKLOG_DIR", tmp_path / "backlog")

    stale = "/Users/someone/Old Place/inbox/20250626 140107.m4a"
    rec_id = db.insert_recording(conn, stale, digest, 10.0)

    assert db.relocate_moved_recordings(conn) == 1
    path = conn.execute(
        "SELECT file_path FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()[0]
    assert path == str(moved)


def test_relocation_requires_the_contents_to_match(conn, tmp_path,
                                                   monkeypatch):
    """A name is not an identity.

    Matching on filename alone attaches a recording to whatever now
    carries that name. Since delete unlinks file_path, a wrong match can
    destroy the wrong audio, so an unverifiable row is left dangling.
    """
    inbox = tmp_path / "inbox"
    _seed_audio(inbox, "20250626 140107.m4a", b"a completely different take")
    monkeypatch.setattr(db.config, "INBOX_DIR", inbox)
    monkeypatch.setattr(db.config, "BACKLOG_DIR", tmp_path / "backlog")

    stale = "/Users/someone/Old Place/inbox/20250626 140107.m4a"
    rec_id = db.insert_recording(conn, stale, "hash-of-the-real-audio", 10.0)

    assert db.relocate_moved_recordings(conn) == 0
    path = conn.execute(
        "SELECT file_path FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()[0]
    assert path == stale


def test_relocation_never_points_two_recordings_at_one_file(conn, tmp_path,
                                                            monkeypatch):
    """Two rows sharing a basename must not collapse onto one file.

    Both would then claim the same audio, and deleting either would take
    the other's recording with it.
    """
    inbox = tmp_path / "inbox"
    kept, digest = _seed_audio(inbox, "20250626 140107.m4a", b"the real one")
    monkeypatch.setattr(db.config, "INBOX_DIR", inbox)
    monkeypatch.setattr(db.config, "BACKLOG_DIR", tmp_path / "backlog")

    mine = db.insert_recording(
        conn, "/Users/someone/Old A/20250626 140107.m4a", digest, 10.0)
    other = db.insert_recording(
        conn, "/Users/someone/Old B/20250626 140107.m4a", "someone-else", 10.0)

    assert db.relocate_moved_recordings(conn) == 1
    rows = dict(conn.execute("SELECT id, file_path FROM recordings"))
    assert rows[mine] == str(kept)
    assert rows[other] == "/Users/someone/Old B/20250626 140107.m4a"
    assert len(set(rows.values())) == 2


def test_relocation_leaves_good_and_ambiguous_paths_alone(conn, tmp_path,
                                                          monkeypatch):
    inbox = tmp_path / "inbox"
    backlog = tmp_path / "backlog"
    inbox.mkdir()
    backlog.mkdir()
    monkeypatch.setattr(db.config, "INBOX_DIR", inbox)
    monkeypatch.setattr(db.config, "BACKLOG_DIR", backlog)

    # A path that still resolves is never touched.
    good, good_digest = _seed_audio(inbox, "here.m4a", b"audio")
    good_id = db.insert_recording(conn, str(good), good_digest, 10.0)

    # The same contents under the same name in both folders is genuinely
    # ambiguous, and the hash cannot break the tie.
    _seed_audio(inbox, "twice.m4a", b"same bytes")
    _, twice_digest = _seed_audio(backlog, "twice.m4a", b"same bytes")
    stale = "/Users/someone/Old Place/inbox/twice.m4a"
    amb_id = db.insert_recording(conn, stale, twice_digest, 10.0)

    assert db.relocate_moved_recordings(conn) == 0
    rows = dict(conn.execute("SELECT id, file_path FROM recordings"))
    assert rows[good_id] == str(good)
    assert rows[amb_id] == stale


def test_connect_runs_the_relocation(tmp_path, monkeypatch):
    """The wiring, not just the helper.

    Nothing else in the suite proves connect() actually performs the
    repair, which is the only thing that fixes a real archive.
    """
    inbox = tmp_path / "inbox"
    moved, digest = _seed_audio(inbox, "20250626 140107.m4a", b"real audio")
    monkeypatch.setattr(db.config, "INBOX_DIR", inbox)
    monkeypatch.setattr(db.config, "BACKLOG_DIR", tmp_path / "backlog")

    db_path = tmp_path / "wiring.db"
    first = db.connect(db_path=db_path, key="ab" * 32)
    stale = "/Users/someone/Old Place/inbox/20250626 140107.m4a"
    rec_id = db.insert_recording(first, stale, digest, 10.0)
    first.close()

    # The guard is per process, so a real second run is what a restart
    # looks like.
    monkeypatch.setattr(db, "_relocation_checked", False)
    second = db.connect(db_path=db_path, key="ab" * 32)
    try:
        path = second.execute(
            "SELECT file_path FROM recordings WHERE id = ?", (rec_id,)
        ).fetchone()[0]
        assert path == str(moved)
    finally:
        second.close()


def test_relocation_failure_cannot_break_connect(tmp_path, monkeypatch):
    """An opportunistic repair must not take down the app's chokepoint."""
    def boom(conn):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(db, "relocate_moved_recordings", boom)
    monkeypatch.setattr(db, "_relocation_checked", False)
    conn = db.connect(db_path=tmp_path / "still_works.db", key="ab" * 32)
    try:
        assert conn.execute("SELECT 1").fetchone()[0] == 1
    finally:
        conn.close()
