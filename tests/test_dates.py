"""recorded_at: filename parsing, mtime fallback, backfill, API order."""

import os
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app import config, db, main


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "dates.db")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)
    return tmp_path


def test_parser_reads_voice_memos_filename():
    assert db.recorded_at_for(
        "/x/20250817 210558-0B89FB69.m4a") == "2025-08-17 21:05:58"
    assert db.recorded_at_for(
        "20260708 144237.m4a") == "2026-07-08 14:42:37"


def test_parser_falls_back_to_file_mtime(tmp_path):
    f = tmp_path / "demo.m4a"
    f.write_bytes(b"x")
    stamp = datetime(2024, 2, 29, 8, 30, 15).timestamp()
    os.utime(f, (stamp, stamp))
    assert db.recorded_at_for(f) == "2024-02-29 08:30:15"


def test_parser_returns_none_when_nothing_to_go_on():
    assert db.recorded_at_for("/nowhere/untitled-memo.m4a") is None


def test_insert_recording_sets_recorded_at(env):
    conn = db.connect()
    rec = db.insert_recording(conn, "/x/20250817 210558-AB.m4a", "h1")
    got = conn.execute(
        "SELECT recorded_at FROM recordings WHERE id = ?", (rec,)
    ).fetchone()[0]
    assert got == "2025-08-17 21:05:58"

    # No filename timestamp and no file on disk: falls back to now,
    # so recorded_at is never NULL for new rows.
    rec2 = db.insert_recording(conn, "/nowhere/untitled.m4a", "h2")
    got2 = conn.execute(
        "SELECT recorded_at FROM recordings WHERE id = ?", (rec2,)
    ).fetchone()[0]
    assert got2 is not None
    conn.close()


def test_migration_backfills_existing_rows(env, tmp_path):
    # Simulate a database from before the column existed.
    conn = db.connect()
    conn.execute("ALTER TABLE recordings DROP COLUMN recorded_at")
    mtime_file = tmp_path / "mtime-memo.m4a"
    mtime_file.write_bytes(b"x")
    stamp = datetime(2023, 11, 2, 17, 5, 9).timestamp()
    os.utime(mtime_file, (stamp, stamp))
    conn.executemany(
        "INSERT INTO recordings (file_path, file_hash, created_at) "
        "VALUES (?, ?, ?)",
        [
            ("/x/20250817 210558-AB.m4a", "h1", "2026-07-07 10:00:00"),
            (str(mtime_file), "h2", "2026-07-07 11:00:00"),
            ("/gone/untitled.m4a", "h3", "2026-07-07 12:00:00"),
        ],
    )
    conn.commit()
    conn.close()

    conn = db.connect()  # migration runs here
    got = dict(conn.execute(
        "SELECT file_hash, recorded_at FROM recordings"
    ).fetchall())
    assert got["h1"] == "2025-08-17 21:05:58"
    assert got["h2"] == "2023-11-02 17:05:09"
    assert got["h3"] == "2026-07-07 12:00:00"
    conn.close()


def test_api_list_orders_by_recorded_at(env):
    conn = db.connect()
    # Processed in this order (created_at ascending), but recorded in
    # the opposite order: the list must follow recorded_at.
    old = db.insert_recording(conn, "/x/20200101 090000-AA.m4a", "h-old")
    new = db.insert_recording(conn, "/x/20250601 090000-BB.m4a", "h-new")
    conn.close()

    rows = TestClient(main.app).get("/api/recordings").json()
    assert [r["id"] for r in rows] == [new, old]
    assert rows[0]["recorded_at"] == "2025-06-01 09:00:00"
    assert rows[1]["recorded_at"] == "2020-01-01 09:00:00"
