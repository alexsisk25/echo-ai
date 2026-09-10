"""Encrypted SQLite storage: key management, schema, and write/read helpers.

The database is encrypted with SQLCipher (AES-256). The key is a random
hex string stored in the macOS Keychain, never in code or on disk.
"""

import hashlib
import logging
import os
import re
import secrets
import stat
import threading
from datetime import datetime
from pathlib import Path

import keyring
import sqlcipher3
import sqlite_vec

from app import config

log = logging.getLogger(__name__)

# Voice Memos names files like "20250817 210558-0B89FB69.m4a": the
# recording's local start time, then a random suffix. Same pattern as
# calendar_sync.FILENAME_TS.
VOICE_MEMO_TS = re.compile(r"^(\d{8}) (\d{6})")


def recorded_at_for(file_path) -> str | None:
    """When the audio was actually recorded, as "YYYY-MM-DD HH:MM:SS".

    The Voice Memos filename timestamp (local time) when present, else
    the audio file's modification time (copy2 preserves it from the
    original), else None so callers fall back to created_at.
    """
    path = Path(file_path)
    m = VOICE_MEMO_TS.match(path.name)
    if m:
        d, t = m.group(1), m.group(2)
        return (f"{d[:4]}-{d[4:6]}-{d[6:]} "
                f"{t[:2]}:{t[2:4]}:{t[4:]}")
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    duration_seconds REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    -- When the audio was recorded (local time), as opposed to
    -- created_at, which is when it was processed on this Mac.
    recorded_at TEXT,
    -- The user's own title. NULL means untitled by hand; display then
    -- falls back to the calendar event, then the AI summary. A manual
    -- title is pinned: no auto-titler ever writes this column.
    title TEXT,
    -- One folder or none. Suggestions are stored separately and are
    -- never applied without an explicit accept.
    folder_id INTEGER REFERENCES folders(id),
    suggested_folder TEXT,
    suggestion_dismissed INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'transcribing', 'done', 'failed'))
);

CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY,
    recording_id INTEGER NOT NULL REFERENCES recordings(id),
    full_text TEXT NOT NULL,
    language TEXT,
    model TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id),
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    text TEXT NOT NULL,
    speaker TEXT,
    -- Set only when voice matching renamed this segment's speaker:
    -- holds the diarized SPEAKER_XX label it replaced, so the tag can
    -- be reverted. NULL means a human (or diarization) set the label.
    auto_original TEXT,
    -- 1 when a human assigned THIS line's speaker directly. Pinned
    -- lines are never moved by cluster renames, auto-tagging, or
    -- retag sweeps; only another per-line reassign (or a full
    -- speaker reset) changes them.
    pinned INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY,
    transcript_id INTEGER NOT NULL UNIQUE REFERENCES transcripts(id),
    summary_json TEXT NOT NULL,
    model TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS calendar_matches (
    recording_id INTEGER PRIMARY KEY REFERENCES recordings(id),
    event_id TEXT NOT NULL,
    event_title TEXT NOT NULL,
    event_start TEXT NOT NULL,
    event_end TEXT NOT NULL,
    attendees_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'matched'
        CHECK (status IN ('matched', 'dismissed')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- When two calendar events fit a recording about equally well, the app
-- refuses to guess: it stores the candidates here instead of writing a
-- match, and the recording asks "Which meeting was this?". A row here
-- and a row in calendar_matches are mutually exclusive.
CREATE TABLE IF NOT EXISTS calendar_candidates (
    recording_id INTEGER PRIMARY KEY REFERENCES recordings(id),
    candidates_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id),
    lang TEXT NOT NULL,
    full_text TEXT NOT NULL,
    summary_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (transcript_id, lang)
);

CREATE TABLE IF NOT EXISTS commitments (
    id INTEGER PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id),
    owner TEXT NOT NULL,
    task TEXT NOT NULL,
    due_date TEXT,
    priority TEXT,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done')),
    segment_id INTEGER REFERENCES segments(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (transcript_id, owner, task)
);

CREATE TABLE IF NOT EXISTS dossiers (
    name TEXT PRIMARY KEY,
    dossier_json TEXT NOT NULL,
    meetings_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS voiceprints (
    name TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    num_samples INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- A person's chosen chip color (a named preset, never a raw value).
-- No row means Auto: the deterministic per-name hue.
CREATE TABLE IF NOT EXISTS person_colors (
    name TEXT PRIMARY KEY,
    color TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rejected_tags (
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id),
    label TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (transcript_id, label, name)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id),
    text TEXT NOT NULL
);

-- Append-only provenance log: one row per recording-lifecycle event.
-- Nothing ever updates or deletes rows here; it is a paper trail.
CREATE TABLE IF NOT EXISTS provenance (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    event_type TEXT NOT NULL,
    stem TEXT NOT NULL,
    context_json TEXT NOT NULL DEFAULT '{}'
);

-- Folders for organizing recordings. Deleting a folder unfiles its
-- recordings; it never deletes them. A privileged folder forces every
-- recording in it, and its digest, through the local model: new
-- arrivals are marked privileged on the way in.
CREATE TABLE IF NOT EXISTS folders (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    privileged INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One synthesized, longitudinal digest per folder, generated on demand.
-- covered_ids records exactly which recordings it was built from, so an
-- update sends only what is new; stale means the folder changed since.
CREATE TABLE IF NOT EXISTS folder_digests (
    folder_id INTEGER PRIMARY KEY REFERENCES folders(id),
    digest_json TEXT NOT NULL,
    covered_ids TEXT NOT NULL DEFAULT '[]',
    recordings_count INTEGER NOT NULL DEFAULT 0,
    stale INTEGER NOT NULL DEFAULT 0,
    model TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Hybrid notes: the user's own jottings per recording (raw_text is
-- theirs and is never overwritten by the app), plus the LLM-enhanced
-- version stored separately. enhanced_hash fingerprints the inputs so
-- re-enhancing unchanged notes returns the stored result.
CREATE TABLE IF NOT EXISTS notes (
    recording_id INTEGER PRIMARY KEY REFERENCES recordings(id),
    raw_text TEXT NOT NULL DEFAULT '',
    enhanced_json TEXT,
    enhanced_hash TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    enhanced_at TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    raw_text,
    content='notes',
    content_rowid='recording_id'
);

CREATE TRIGGER IF NOT EXISTS notes_fts_insert
AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts (rowid, raw_text)
    VALUES (new.recording_id, new.raw_text);
END;

CREATE TRIGGER IF NOT EXISTS notes_fts_delete
AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts (notes_fts, rowid, raw_text)
    VALUES ('delete', old.recording_id, old.raw_text);
END;

CREATE TRIGGER IF NOT EXISTS notes_fts_update
AFTER UPDATE OF raw_text ON notes BEGIN
    INSERT INTO notes_fts (notes_fts, rowid, raw_text)
    VALUES ('delete', old.recording_id, old.raw_text);
    INSERT INTO notes_fts (rowid, raw_text)
    VALUES (new.recording_id, new.raw_text);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    embedding float[384]
);

CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
    full_text,
    content='transcripts',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS transcripts_fts_insert
AFTER INSERT ON transcripts BEGIN
    INSERT INTO transcripts_fts (rowid, full_text)
    VALUES (new.id, new.full_text);
END;

CREATE TRIGGER IF NOT EXISTS transcripts_fts_delete
AFTER DELETE ON transcripts BEGIN
    INSERT INTO transcripts_fts (transcripts_fts, rowid, full_text)
    VALUES ('delete', old.id, old.full_text);
END;

CREATE TRIGGER IF NOT EXISTS transcripts_fts_update
AFTER UPDATE OF full_text ON transcripts BEGIN
    INSERT INTO transcripts_fts (transcripts_fts, rowid, full_text)
    VALUES ('delete', old.id, old.full_text);
    INSERT INTO transcripts_fts (rowid, full_text)
    VALUES (new.id, new.full_text);
END;
"""


def get_or_create_key() -> str:
    """Fetch the DB key from the Keychain, creating one on first run."""
    key = keyring.get_password(config.KEYCHAIN_SERVICE, config.KEYCHAIN_ACCOUNT)
    if key is None:
        key = secrets.token_hex(32)
        keyring.set_password(
            config.KEYCHAIN_SERVICE, config.KEYCHAIN_ACCOUNT, key
        )
        print(
            "NOTE: a new database encryption key was just created and\n"
            f"stored in your macOS Keychain (service '{config.KEYCHAIN_SERVICE}',\n"
            f"account '{config.KEYCHAIN_ACCOUNT}'). The database cannot be\n"
            "opened or recovered without it. If you ever migrate to a new\n"
            "Mac, this Keychain item must come along or the recordings\n"
            "database is lost. Do not delete it from Keychain Access."
        )
    return key


# PLAN.md's security rules ask for owner-only permissions on the database
# "and its journal files". SQLite writes those siblings itself, whenever
# it likes and with whatever the process umask allows, so a one-time
# chmod at creation is not enough: this runs on every connect.
JOURNAL_SUFFIXES = ("-wal", "-shm", "-journal")


def tighten_permissions(db_path: Path) -> list[Path]:
    """chmod 0600 the database and any journal siblings that exist.

    Returns the paths actually tightened, so callers (and tests) can see
    what was covered.
    """
    owner_only = stat.S_IRUSR | stat.S_IWUSR
    touched = []
    candidates = [Path(db_path)] + [
        Path(str(db_path) + suffix) for suffix in JOURNAL_SUFFIXES
    ]
    for path in candidates:
        try:
            if not path.exists():
                continue
            if stat.S_IMODE(path.stat().st_mode) != owner_only:
                os.chmod(path, owner_only)
            touched.append(path)
        except OSError:
            # A permission fix is best effort: never take the app down
            # over one, and never hide the rest of the files behind it.
            continue
    return touched


def connect(db_path=None, key=None) -> sqlcipher3.Connection:
    """Open the encrypted database, creating schema and file if needed.

    Tests pass an explicit db_path and key; normal use reads config
    and the Keychain.
    """
    db_path = db_path or config.DB_PATH
    key = key or get_or_create_key()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlcipher3.connect(str(db_path))
    # PRAGMA key must be the first statement on a SQLCipher connection.
    conn.execute("PRAGMA key = \"x'%s'\"" % key)
    conn.execute("PRAGMA foreign_keys = ON")
    # State the journal mode instead of inheriting the default, so the
    # sibling files this app has to protect are the ones it expects.
    conn.execute("PRAGMA journal_mode = WAL")

    # sqlite-vec powers semantic search; the vec0 table in the schema
    # needs the extension loaded on every connection.
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Queries against an external-content FTS table read the content table,
    # so a row-count comparison cannot tell whether the index is populated.
    # Instead: if the FTS table is about to be created for the first time,
    # rebuild afterwards to index transcripts that predate it.
    fts_existed = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name = 'transcripts_fts'"
    ).fetchone() is not None

    conn.executescript(SCHEMA)
    conn.commit()

    if not fts_existed:
        conn.execute(
            "INSERT INTO transcripts_fts (transcripts_fts) VALUES ('rebuild')"
        )
        conn.commit()

    # Migration: a long recording that diarizes to a single speaker is
    # usually a failure (a noisy room collapsing two voices), so the UI
    # offers a reprocess with a speaker-count hint. 0 = fine or already
    # dismissed, 1 = looks wrong.
    rec_cols = [r[1] for r in conn.execute(
        "PRAGMA table_info(recordings)"
    ).fetchall()]
    if "diarization_suspect" not in rec_cols:
        conn.execute(
            "ALTER TABLE recordings ADD COLUMN diarization_suspect "
            "INTEGER NOT NULL DEFAULT 0"
        )
        conn.commit()

    # Migration: two diarization clusters sitting under one speaker
    # name (a rename that merged two people). Holds the offending name,
    # NULL when clean; the dismissal is separate so a suggestion the
    # user waved away does not come back on every rescan.
    if "merged_name_suspect" not in rec_cols:
        conn.execute(
            "ALTER TABLE recordings ADD COLUMN merged_name_suspect TEXT")
        conn.execute(
            "ALTER TABLE recordings ADD COLUMN merged_name_dismissed "
            "INTEGER NOT NULL DEFAULT 0")
        conn.commit()

    # Migration: a folder can be privileged as a whole, which pins its
    # recordings and its digest to the local model.
    folder_cols = [r[1] for r in conn.execute(
        "PRAGMA table_info(folders)"
    ).fetchall()]
    if "privileged" not in folder_cols:
        conn.execute(
            "ALTER TABLE folders ADD COLUMN privileged INTEGER NOT NULL "
            "DEFAULT 0"
        )
        conn.commit()

    # Migration: chip colors briefly stored preset NAMES before moving
    # to hue numbers. Convert any legacy row to its historical hue.
    legacy_hues = {"red": 25, "orange": 55, "gold": 95, "green": 148,
                   "teal": 178, "cyan": 215, "blue": 260, "violet": 300,
                   "plum": 330, "pink": 355}
    for name, hue in legacy_hues.items():
        conn.execute("UPDATE person_colors SET color = ? WHERE color = ?",
                     (hue, name))
    conn.commit()

    # Migration for databases created before the privacy vault: the
    # privileged flag forces all analysis of a recording to local models.
    cols = [r[1] for r in conn.execute(
        "PRAGMA table_info(recordings)"
    ).fetchall()]
    if "privileged" not in cols:
        conn.execute(
            "ALTER TABLE recordings ADD COLUMN privileged INTEGER "
            "NOT NULL DEFAULT 0"
        )
        conn.commit()

    # Migration for databases created before tag provenance existed.
    seg_cols = [r[1] for r in conn.execute(
        "PRAGMA table_info(segments)"
    ).fetchall()]
    if "auto_original" not in seg_cols:
        conn.execute("ALTER TABLE segments ADD COLUMN auto_original TEXT")
        conn.commit()

    # Migration for databases created before per-line reassignment.
    if "pinned" not in seg_cols:
        conn.execute(
            "ALTER TABLE segments ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
        )
        conn.commit()

    # Migration for databases created before titles and folders.
    if "title" not in cols:
        conn.execute("ALTER TABLE recordings ADD COLUMN title TEXT")
        conn.execute(
            "ALTER TABLE recordings ADD COLUMN folder_id INTEGER "
            "REFERENCES folders(id)"
        )
        conn.execute(
            "ALTER TABLE recordings ADD COLUMN suggested_folder TEXT"
        )
        conn.execute(
            "ALTER TABLE recordings ADD COLUMN suggestion_dismissed "
            "INTEGER NOT NULL DEFAULT 0"
        )
        conn.commit()

    # Migration for databases created before recorded_at: backfill from
    # the Voice Memos filename, else the audio file's mtime, else keep
    # the processing timestamp.
    if "recorded_at" not in cols:
        conn.execute("ALTER TABLE recordings ADD COLUMN recorded_at TEXT")
        rows = conn.execute(
            "SELECT id, file_path, created_at FROM recordings"
        ).fetchall()
        for rec_id, file_path, created_at in rows:
            conn.execute(
                "UPDATE recordings SET recorded_at = ? WHERE id = ?",
                (recorded_at_for(file_path) or created_at, rec_id),
            )
        conn.commit()

    # Once per process: repoint recordings whose audio moved with the
    # project folder. Cheap when nothing moved (one stat per recording).
    # Wrapped, because this now gates every connect and an opportunistic
    # repair must never be able to take down the app's one chokepoint.
    global _relocation_checked
    with _relocation_lock:
        if not _relocation_checked:
            _relocation_checked = True
            try:
                relocate_moved_recordings(conn)
            except Exception:
                log.exception("Could not check for relocated recordings")

    # Every connect, not just the first: WAL and shm files appear after
    # creation, and an earlier run under a looser umask leaves them
    # readable until something tightens them again.
    tighten_permissions(db_path)
    return conn


# Set once per process so the scan below does not run on every connect.
# Locked because the startup warmup thread and the first request can
# otherwise both pass the check and hash the archive twice over.
_relocation_checked = False
_relocation_lock = threading.Lock()


def _sha256(path) -> str:
    """Same digest as watcher.file_hash, which is what file_hash holds.

    Duplicated rather than imported because watcher imports this module.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relocate_moved_recordings(conn) -> int:
    """Repoint recordings whose audio moved with the project folder.

    file_path is absolute, so moving the project leaves every row
    dangling and silently breaks playback, export, retry, reprocess,
    voice enrollment, and delete (unlink no-ops on a stale path, so the
    audio survives a delete and can be re-imported). That has now
    happened twice: the local-otter rename in July and the move off
    iCloud in September, each repaired by hand afterwards.

    A candidate is accepted only when its contents hash to the
    file_hash already stored on the row. Matching on filename alone
    would attach a recording to whatever now carries that name, and
    since delete unlinks file_path, a wrong match can destroy the wrong
    audio. Anything ambiguous or unverifiable is left dangling, which is
    visible and recoverable, rather than guessed.
    """
    rows = conn.execute(
        "SELECT id, file_path, file_hash FROM recordings"
    ).fetchall()
    fixed = 0
    skipped = 0
    for rec_id, file_path, file_hash in rows:
        if not file_path or os.path.exists(file_path):
            continue
        if not file_hash:
            skipped += 1
            continue
        name = os.path.basename(file_path)
        matches = []
        for folder in (config.INBOX_DIR, config.BACKLOG_DIR):
            candidate = folder / name
            try:
                if candidate.exists() and _sha256(candidate) == file_hash:
                    matches.append(candidate)
            except OSError:
                continue
        if len(matches) != 1:
            skipped += 1
            continue
        conn.execute(
            "UPDATE recordings SET file_path = ? WHERE id = ?",
            (str(matches[0]), rec_id),
        )
        fixed += 1
    if fixed:
        conn.commit()
        log.info("Repointed %d recording(s) after a folder move", fixed)
    if skipped:
        # Silence here would read as "nothing moved", which is a
        # different claim from "these could not be matched safely".
        log.warning(
            "%d recording(s) have missing audio that could not be "
            "matched by content and were left alone", skipped
        )
    return fixed


def insert_recording(conn, file_path: str, file_hash: str,
                     duration_seconds: float | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO recordings "
        "(file_path, file_hash, duration_seconds, recorded_at) "
        "VALUES (?, ?, ?, COALESCE(?, datetime('now')))",
        (file_path, file_hash, duration_seconds, recorded_at_for(file_path)),
    )
    conn.commit()
    return cur.lastrowid


def recording_exists(conn, file_hash: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM recordings WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    return row is not None


def set_recording_status(conn, recording_id: int, status: str) -> None:
    conn.execute(
        "UPDATE recordings SET status = ? WHERE id = ?", (status, recording_id)
    )
    conn.commit()


def insert_transcript(conn, recording_id: int, full_text: str,
                      language: str | None, model: str,
                      segments: list[dict]) -> int:
    cur = conn.execute(
        "INSERT INTO transcripts (recording_id, full_text, language, model) "
        "VALUES (?, ?, ?, ?)",
        (recording_id, full_text, language, model),
    )
    transcript_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO segments "
        "(transcript_id, start_seconds, end_seconds, text, speaker, pinned) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (transcript_id, s["start"], s["end"], s["text"], s.get("speaker"),
             1 if s.get("pinned") else 0)
            for s in segments
        ],
    )
    conn.commit()
    return transcript_id


def rename_speaker(conn, transcript_id: int, old_label: str,
                   new_label: str, include_pinned: bool = False) -> int:
    """Rename a speaker across one transcript. Returns rows changed.

    Lines a human assigned individually (pinned) stay where they are,
    unless include_pinned is set: the caller must have asked for that
    explicitly, because a pinned line is a human decision.
    """
    sql = ("UPDATE segments SET speaker = ? WHERE transcript_id = ? "
           "AND speaker = ?")
    if not include_pinned:
        sql += " AND pinned = 0"
    cur = conn.execute(sql, (new_label, transcript_id, old_label))
    conn.commit()
    return cur.rowcount


SUSPECT_MIN_SECONDS = 600.0


def refresh_diarization_suspect(conn, recording_id: int) -> bool:
    """Flag a long recording whose diarization produced one speaker.

    Ten minutes of conversation almost never is one voice, so this is
    the shape of a failed diarization. It only marks; labels are never
    changed automatically.
    """
    row = conn.execute(
        "SELECT duration_seconds FROM recordings WHERE id = ?",
        (recording_id,),
    ).fetchone()
    duration = (row[0] if row else None) or 0
    n_speakers = conn.execute(
        "SELECT COUNT(DISTINCT speaker) FROM segments s "
        "JOIN transcripts t ON t.id = s.transcript_id "
        "WHERE t.recording_id = ? AND s.speaker IS NOT NULL",
        (recording_id,),
    ).fetchone()[0]
    suspect = duration >= SUSPECT_MIN_SECONDS and n_speakers == 1
    conn.execute(
        "UPDATE recordings SET diarization_suspect = ? WHERE id = ?",
        (1 if suspect else 0, recording_id),
    )
    conn.commit()
    return suspect


def count_pinned_for_speaker(conn, transcript_id: int, label: str) -> int:
    """How many of this speaker's lines are pinned, and so would be left
    behind by a normal rename. Lets the UI explain a no-op rename."""
    return conn.execute(
        "SELECT COUNT(*) FROM segments WHERE transcript_id = ? "
        "AND speaker = ? AND pinned = 1",
        (transcript_id, label),
    ).fetchone()[0]


def speaker_exists(conn, transcript_id: int, label: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM segments WHERE transcript_id = ? AND speaker = ? "
        "LIMIT 1",
        (transcript_id, label),
    ).fetchone() is not None


def reassign_segment(conn, transcript_id: int, segment_id: int,
                     new_label: str) -> bool:
    """Assign one line to a speaker, leaving its old cluster alone.
    The line is pinned: cluster renames and auto-tagging skip it from
    now on. Returns False when the segment is not in this transcript."""
    cur = conn.execute(
        "UPDATE segments SET speaker = ?, auto_original = NULL, pinned = 1 "
        "WHERE id = ? AND transcript_id = ?",
        (new_label, segment_id, transcript_id),
    )
    conn.commit()
    return cur.rowcount == 1


def assign_segments(conn, transcript_id: int, segment_ids: list[int],
                    new_label: str) -> int:
    """Assign several lines to one speaker and pin them. The manual
    cluster-split path. Returns rows changed."""
    changed = 0
    for sid in segment_ids:
        changed += conn.execute(
            "UPDATE segments SET speaker = ?, auto_original = NULL, "
            "pinned = 1 WHERE id = ? AND transcript_id = ?",
            (new_label, sid, transcript_id),
        ).rowcount
    conn.commit()
    return changed


def confirm_segment(conn, transcript_id: int, segment_id: int) -> bool:
    """Confirm one auto-tagged line: promote it to human-set (clear
    auto_original) without changing the name. Line-scoped."""
    cur = conn.execute(
        "UPDATE segments SET auto_original = NULL "
        "WHERE id = ? AND transcript_id = ? AND auto_original IS NOT NULL",
        (segment_id, transcript_id),
    )
    conn.commit()
    return cur.rowcount == 1


def revert_segment(conn, transcript_id: int, segment_id: int) -> bool:
    """Revert one auto-tagged line back to its SPEAKER_XX label. Only
    this line; no rejection is recorded and no other line moves."""
    cur = conn.execute(
        "UPDATE segments SET speaker = auto_original, auto_original = NULL "
        "WHERE id = ? AND transcript_id = ? AND auto_original IS NOT NULL",
        (segment_id, transcript_id),
    )
    conn.commit()
    return cur.rowcount == 1


def segment_states(conn, transcript_id: int,
                   segment_ids: list[int]) -> list[dict]:
    """Current (speaker, auto_original, pinned) for the given lines, so
    the caller can snapshot before a change and undo it exactly."""
    if not segment_ids:
        return []
    marks = ",".join("?" * len(segment_ids))
    rows = conn.execute(
        f"SELECT id, speaker, auto_original, pinned FROM segments "
        f"WHERE transcript_id = ? AND id IN ({marks})",
        (transcript_id, *segment_ids),
    ).fetchall()
    return [{"id": r[0], "speaker": r[1], "auto_original": r[2],
             "pinned": r[3]} for r in rows]


def restore_segments(conn, transcript_id: int, states: list[dict]) -> int:
    """Set lines back to exact prior states. The undo primitive: every
    line-labeling action is reversed by restoring its snapshot."""
    changed = 0
    for s in states:
        changed += conn.execute(
            "UPDATE segments SET speaker = ?, auto_original = ?, pinned = ? "
            "WHERE id = ? AND transcript_id = ?",
            (s.get("speaker"), s.get("auto_original"),
             1 if s.get("pinned") else 0, s["id"], transcript_id),
        ).rowcount
    conn.commit()
    return changed


def search_transcripts(conn, query: str, limit: int = 20,
                       folder_id: int | None = None) -> list[dict]:
    """Keyword search over all transcripts via FTS5, best matches first.

    The query is passed as a quoted string so FTS5 operators like AND
    and NEAR are treated as plain words, not syntax a user must know.
    folder_id narrows the search to one folder, so a search made while
    the list is filtered means what it looks like it means.
    """
    quoted = '"%s"' % query.replace('"', '""')
    scope = "" if folder_id is None else " AND r.folder_id = ?"
    params = ([quoted] if folder_id is None else [quoted, folder_id]) + [limit]
    rows = conn.execute(
        """
        SELECT t.id, t.recording_id, r.file_path, r.created_at,
               snippet(transcripts_fts, 0, '<mark>', '</mark>', ' ... ', 12)
        FROM transcripts_fts
        JOIN transcripts t ON t.id = transcripts_fts.rowid
        JOIN recordings r ON r.id = t.recording_id
        WHERE transcripts_fts MATCH ?""" + scope + """
        ORDER BY rank
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        {
            "transcript_id": r[0],
            "recording_id": r[1],
            "file_path": r[2],
            "created_at": r[3],
            "snippet": r[4],
        }
        for r in rows
    ]


def search_notes(conn, query: str, limit: int = 20,
                 folder_id: int | None = None) -> list[dict]:
    """Keyword search over the user's own notes, best matches first."""
    quoted = '"%s"' % query.replace('"', '""')
    scope = "" if folder_id is None else " AND r.folder_id = ?"
    params = ([quoted] if folder_id is None else [quoted, folder_id]) + [limit]
    rows = conn.execute(
        """
        SELECT n.recording_id, r.file_path, r.created_at,
               snippet(notes_fts, 0, '<mark>', '</mark>', ' ... ', 12)
        FROM notes_fts
        JOIN notes n ON n.recording_id = notes_fts.rowid
        JOIN recordings r ON r.id = n.recording_id
        WHERE notes_fts MATCH ?""" + scope + """
        ORDER BY rank
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        {
            "transcript_id": None,
            "recording_id": r[0],
            "file_path": r[1],
            "created_at": r[2],
            "snippet": r[3],
            "source": "notes",
        }
        for r in rows
    ]


def upsert_summary(conn, transcript_id: int, summary_json: str,
                   model: str) -> None:
    conn.execute(
        """
        INSERT INTO summaries (transcript_id, summary_json, model)
        VALUES (?, ?, ?)
        ON CONFLICT (transcript_id)
        DO UPDATE SET summary_json = excluded.summary_json,
                      model = excluded.model,
                      created_at = datetime('now')
        """,
        (transcript_id, summary_json, model),
    )
    conn.commit()
    # A folder digest is built from these summaries, so rewriting one
    # (a reprocess, a re-summarize) means the stored digest describes a
    # summary that no longer exists. Marked here, at the single place
    # every writer goes through, rather than at each call site.
    from app import digest
    digest.mark_stale_for_recording(conn, transcript_id)


def is_privileged(conn, recording_id: int) -> bool:
    row = conn.execute(
        "SELECT privileged FROM recordings WHERE id = ?", (recording_id,)
    ).fetchone()
    return bool(row and row[0])


def transcript_privileged(conn, transcript_id: int) -> bool:
    row = conn.execute(
        "SELECT r.privileged FROM recordings r "
        "JOIN transcripts t ON t.recording_id = r.id WHERE t.id = ?",
        (transcript_id,),
    ).fetchone()
    return bool(row and row[0])


def get_summary(conn, transcript_id: int) -> str | None:
    row = conn.execute(
        "SELECT summary_json FROM summaries WHERE transcript_id = ?",
        (transcript_id,),
    ).fetchone()
    return row[0] if row else None
