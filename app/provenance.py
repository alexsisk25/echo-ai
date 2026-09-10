"""Append-only provenance log: a paper trail of every recording
lifecycle event, so a question like "what happened to this memo?" has
an answer instead of a guess (the .qta re-sync incident needed one).

Event types:
- imported: a memo copied/converted in from Voice Memos (context:
  original_filename, source_format).
- captured: an in-app meeting recording (context: file).
- deleted: a recording removed (context: recording_id, title, how =
  single | bulk | maintenance).
- sync-skipped-as-deleted: a folder memo the ledger blocked from
  re-importing because it had been deleted (a blocked resurrection).
- retried: the pipeline re-ran on a failed recording's existing file
  (context: recording_id, outcome = done | failed |
  blocked-unreadable).
- resynced: a corrupt local copy was replaced with a fresh copy from
  the Voice Memos folder and reprocessed (context: recording_id,
  outcome, source_format).
- reprocessed: a done recording re-ran everything, speakers only, or
  notes only (context: recording_id, scope, outcome).

Nothing in this module updates or deletes rows. Writes are best-effort
at the call sites: a logging failure must never break the real action.
"""

import json

IMPORTED = "imported"
CAPTURED = "captured"
DELETED = "deleted"
SYNC_SKIPPED = "sync-skipped-as-deleted"
RETRIED = "retried"
RESYNCED = "resynced"
REPROCESSED = "reprocessed"


def log_event(conn, event_type: str, stem: str, created_at: str | None = None,
              **context) -> None:
    """Append one event. created_at defaults to now; seeding passes a
    historical timestamp."""
    if created_at:
        conn.execute(
            "INSERT INTO provenance (created_at, event_type, stem, "
            "context_json) VALUES (?, ?, ?, ?)",
            (created_at, event_type, stem, json.dumps(context)),
        )
    else:
        conn.execute(
            "INSERT INTO provenance (event_type, stem, context_json) "
            "VALUES (?, ?, ?)",
            (event_type, stem, json.dumps(context)),
        )
    conn.commit()


def log_imported(conn, stem: str, original_filename: str,
                 source_format: str) -> None:
    log_event(conn, IMPORTED, stem, original_filename=original_filename,
              source_format=source_format)


def log_captured(conn, stem: str, file: str) -> None:
    log_event(conn, CAPTURED, stem, file=file)


def log_deleted(conn, stem: str, recording_id: int, title: str | None,
                how: str = "single") -> None:
    log_event(conn, DELETED, stem, recording_id=recording_id,
              title=title, how=how)


def log_sync_skipped(conn, stem: str) -> None:
    log_event(conn, SYNC_SKIPPED, stem)


def blocked_as_deleted(conn, stem: str) -> bool:
    """True when a file with this stem is a resurrection: its lifecycle
    story ends in a deletion and no live recording carries the stem.

    This is the import gate the folder scan uses. Ledger membership
    alone cannot be the signal there: every synced memo is in the
    ledger, including ones mid-first-processing. A deliberate re-import
    stays possible because sync writes a fresh 'imported' event when it
    copies a file, which makes the deletion no longer the last word.
    """
    from pathlib import Path
    last = conn.execute(
        "SELECT event_type FROM provenance WHERE stem = ? "
        "AND event_type IN (?, ?, ?) ORDER BY id DESC LIMIT 1",
        (stem, IMPORTED, CAPTURED, DELETED),
    ).fetchone()
    if last is None or last[0] != DELETED:
        return False
    live = any(
        Path(row[0]).stem == stem
        for row in conn.execute("SELECT file_path FROM recordings").fetchall()
    )
    return not live


def was_deleted(conn, stem: str) -> bool:
    """Has this memo ever been recorded as deleted? Used by sync to know
    a skip is a blocked resurrection rather than an ordinary dedupe."""
    return conn.execute(
        "SELECT 1 FROM provenance WHERE stem = ? AND event_type = ? LIMIT 1",
        (stem, DELETED),
    ).fetchone() is not None


def list_events(conn, event_type: str | None = None,
                limit: int = 200) -> list[dict]:
    """Recent events, newest first, optionally filtered by type."""
    where, params = "", []
    if event_type:
        where = "WHERE event_type = ?"
        params.append(event_type)
    rows = conn.execute(
        f"SELECT id, created_at, event_type, stem, context_json "
        f"FROM provenance {where} ORDER BY created_at DESC, id DESC LIMIT ?",
        (*params, limit),
    ).fetchall()
    return [
        {"id": r[0], "created_at": r[1], "event_type": r[2], "stem": r[3],
         "context": json.loads(r[4])}
        for r in rows
    ]


def seed(conn) -> int:
    """Populate the log from what is known today, once. Idempotent: does
    nothing if any event already exists. Returns rows written.

    Seeds current recordings as imported/captured events (timestamped by
    recorded_at), and the 16 memos removed in the 2026-07-26 .qta
    cleanup as deleted events.
    """
    from pathlib import Path

    if conn.execute("SELECT 1 FROM provenance LIMIT 1").fetchone():
        return 0

    written = 0
    rows = conn.execute(
        "SELECT id, file_path, recorded_at, created_at FROM recordings"
    ).fetchall()
    for rec_id, file_path, recorded_at, created_at in rows:
        name = Path(file_path).name
        stem = Path(file_path).stem
        when = recorded_at or created_at
        if name.endswith(("-meeting.m4a", "-inperson.m4a")):
            log_event(conn, CAPTURED, stem, created_at=when, file=name,
                      seeded=True)
        else:
            log_event(conn, IMPORTED, stem, created_at=when,
                      original_filename=name, source_format="unknown",
                      seeded=True)
        written += 1

    for stem in QTA_CLEANUP_STEMS:
        log_event(conn, DELETED, stem, created_at="2026-07-26 17:30:00",
                  recording_id=None, title=None, how="maintenance",
                  context="qta incident cleanup", seeded=True)
        written += 1
    return written


# The 16 memos removed in the 2026-07-26 cleanup (dated before July 8;
# see GOAL.md). Recorded here so the log reflects that history.
QTA_CLEANUP_STEMS = [
    "20260126 110850-2F57533A", "20260126 130033-1D4DEE93",
    "20260128 111517-3915FBD9", "20260128 123343-EE5BFF59",
    "20260401 140239-EC4FE956", "20260406 111031-5080FF2A",
    "20260408 112414-82666436", "20260413 111809-8589C690",
    "20260413 124046-CE4952F9", "20260519 143914-1A922C3C",
    "20260528 091652-9CCAF4AB", "20260604 122526-7FCF5BB8",
    "20260622 171152-B04AF74A", "20260623 111644-7C31D2D3",
    "20260701 104352-BAFD1114", "20260707 111829-F511EA24",
]
