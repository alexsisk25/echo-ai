"""Recording management: permanent delete and export-to-zip.

Delete removes everything a recording contributed: the audio file, its
transcript, segments, summary, translations, commitments, embedding
chunks, FTS entries (via trigger), calendar match, the user's notes,
and any cached dossier
for a person who spoke in it (the dossier rebuilds without this meeting
on next view). The filename is kept in the sync ledger so a deleted memo
never re-syncs from the iPhone. Voiceprint enrollments survive: they are
averaged across recordings, and speaker labels elsewhere are untouched.

Export builds a zip per recording holding the audio file, the transcript
as markdown (speakers and timestamps), and the AI notes as markdown.
A bulk export nests one folder per recording, and each folder
represented in it also gets that folder's digest as digest.md.
"""

import json
import re
import tempfile
import zipfile
from pathlib import Path

from app import sync


def delete_recording(conn, recording_id: int,
                     how: str = "single") -> dict | None:
    """Permanently delete one recording and all data derived from it.

    Writes a provenance 'deleted' event (how = single | bulk |
    maintenance). Returns a dict of what was removed, or None if the id
    is unknown.
    """
    from app import provenance

    row = conn.execute(
        """
        SELECT r.file_path, r.folder_id, COALESCE(r.title,
                 (SELECT event_title FROM calendar_matches
                  WHERE recording_id = r.id AND status = 'matched'),
                 (SELECT json_extract(s.summary_json, '$.title')
                  FROM transcripts t JOIN summaries s
                    ON s.transcript_id = t.id
                  WHERE t.recording_id = r.id LIMIT 1))
        FROM recordings r WHERE r.id = ?
        """,
        (recording_id,),
    ).fetchone()
    if row is None:
        return None
    file_path = Path(row[0])
    folder_id = row[1]
    title = row[2]

    transcript_ids = [r[0] for r in conn.execute(
        "SELECT id FROM transcripts WHERE recording_id = ?", (recording_id,)
    ).fetchall()]

    # Names must be collected before segments go: any person who spoke
    # here has a stale cached dossier once this meeting disappears.
    named_speakers = set()
    counts = {"transcripts": len(transcript_ids), "segments": 0,
              "chunks": 0, "commitments": 0}
    for tid in transcript_ids:
        named_speakers.update(r[0] for r in conn.execute(
            "SELECT DISTINCT speaker FROM segments WHERE transcript_id = ? "
            "AND speaker IS NOT NULL "
            "AND speaker NOT LIKE 'SPEAKER^_%' ESCAPE '^'",
            (tid,),
        ).fetchall())
        counts["commitments"] += conn.execute(
            "DELETE FROM commitments WHERE transcript_id = ?", (tid,)
        ).rowcount
        conn.execute(
            "DELETE FROM chunks_vec WHERE rowid IN "
            "(SELECT id FROM chunks WHERE transcript_id = ?)", (tid,)
        )
        counts["chunks"] += conn.execute(
            "DELETE FROM chunks WHERE transcript_id = ?", (tid,)
        ).rowcount
        conn.execute(
            "DELETE FROM rejected_tags WHERE transcript_id = ?", (tid,)
        )
        conn.execute(
            "DELETE FROM translations WHERE transcript_id = ?", (tid,)
        )
        conn.execute(
            "DELETE FROM summaries WHERE transcript_id = ?", (tid,)
        )
        counts["segments"] += conn.execute(
            "DELETE FROM segments WHERE transcript_id = ?", (tid,)
        ).rowcount
        # The transcripts_fts_delete trigger drops the FTS entry here.
        conn.execute("DELETE FROM transcripts WHERE id = ?", (tid,))

    conn.execute(
        "DELETE FROM calendar_matches WHERE recording_id = ?", (recording_id,)
    )
    conn.execute(
        "DELETE FROM calendar_candidates WHERE recording_id = ?",
        (recording_id,)
    )
    # The user's notes for this recording (the notes_fts trigger cleans
    # the search index).
    conn.execute("DELETE FROM notes WHERE recording_id = ?", (recording_id,))
    counts["dossiers"] = 0
    for name in named_speakers:
        counts["dossiers"] += conn.execute(
            "DELETE FROM dossiers WHERE name = ?", (name,)
        ).rowcount
    conn.execute("DELETE FROM recordings WHERE id = ?", (recording_id,))
    conn.commit()

    # Shared-file guard: if another recording still references this same
    # audio file (e.g. a junk duplicate row next to a healthy one), only
    # the database rows go; the file stays for the survivor.
    shared = conn.execute(
        "SELECT 1 FROM recordings WHERE file_path = ? LIMIT 1",
        (str(file_path),),
    ).fetchone()
    counts["audio_deleted"] = file_path.exists() and not shared
    counts["file_kept_shared"] = bool(shared)
    if not shared:
        file_path.unlink(missing_ok=True)
    # Deleted means gone for good: the ledger entry stops the sync
    # from ever copying this memo back in from the iPhone.
    sync.add_to_ledger(file_path.name)
    provenance.log_deleted(conn, file_path.stem, recording_id, title, how)
    # The folder's digest described this recording; it no longer holds.
    from app import digest
    digest.mark_stale(conn, folder_id)
    counts["id"] = recording_id
    return counts


def _timestamp(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _slug(text: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:60]
    return slug or fallback


def _transcript_markdown(title: str, segments: list, full_text: str) -> str:
    lines = [f"# {title}", ""]
    if segments:
        for start, _end, text, speaker in segments:
            lines.append(
                f"[{_timestamp(start)}] {speaker or 'unknown'}: {text.strip()}"
            )
    else:
        lines.append(full_text or "No transcript.")
    return "\n".join(lines) + "\n"


def _notes_markdown(summary: dict) -> str:
    lines = [f"# {summary.get('title') or 'Untitled'}", ""]
    if summary.get("date"):
        lines.append(f"Date: {summary['date']}")
    if summary.get("attendees"):
        lines.append(f"Attendees: {', '.join(summary['attendees'])}")
    lines += ["", "## Summary", "", summary.get("summary", "")]
    for heading, key in (("Decisions", "decisions"), ("Risks", "risks")):
        if summary.get(key):
            lines += ["", f"## {heading}", ""]
            lines += [f"- {item}" for item in summary[key]]
    if summary.get("action_items"):
        lines += ["", "## Action items", ""]
        for item in summary["action_items"]:
            extra = []
            if item.get("due_date"):
                extra.append(f"due {item['due_date']}")
            if item.get("priority"):
                extra.append(f"priority {item['priority']}")
            suffix = f" ({', '.join(extra)})" if extra else ""
            lines.append(f"- {item.get('owner', 'unassigned')}: "
                         f"{item.get('task', '')}{suffix}")
    if summary.get("topics"):
        lines += ["", "## Topics", "", ", ".join(summary["topics"])]
    return "\n".join(lines) + "\n"


def _recording_bundle(conn, recording_id: int) -> dict | None:
    from app import notes as user_notes

    row = conn.execute(
        """
        SELECT r.file_path, t.id, t.full_text, s.summary_json,
               r.title, f.name, r.folder_id,
               (SELECT event_title FROM calendar_matches
                WHERE recording_id = r.id AND status = 'matched')
        FROM recordings r
        LEFT JOIN transcripts t ON t.recording_id = r.id
        LEFT JOIN summaries s ON s.transcript_id = t.id
        LEFT JOIN folders f ON f.id = r.folder_id
        WHERE r.id = ?
        """,
        (recording_id,),
    ).fetchone()
    if row is None:
        return None
    (file_path, tid, full_text, summary_json, manual_title, folder_name,
     folder_id, event_title) = row
    segments = []
    if tid is not None:
        segments = conn.execute(
            "SELECT start_seconds, end_seconds, text, speaker FROM segments "
            "WHERE transcript_id = ? ORDER BY start_seconds", (tid,)
        ).fetchall()
    summary = json.loads(summary_json) if summary_json else None
    # Same precedence as the recording list: manual > calendar > AI.
    title = (manual_title or event_title or (summary or {}).get("title")
             or Path(file_path).stem)
    jotted = user_notes.get(conn, recording_id)
    return {
        "file_path": Path(file_path),
        "title": title,
        "folder": folder_name,
        "folder_id": folder_id,
        "transcript_md": _transcript_markdown(title, segments, full_text),
        "notes_md": _notes_markdown(summary) if summary else None,
        "my_notes_md": (f"# My notes\n\n{jotted['raw_text']}\n"
                        if jotted["raw_text"].strip() else None),
        "enhanced_md": (user_notes.enhanced_markdown(jotted["enhanced"])
                        if jotted["enhanced"] else None),
    }


def _folders_in(bundles) -> list[tuple[int, str]]:
    """Distinct (folder_id, folder_name) pairs across a set of bundles,
    in the order they first appear."""
    seen, out = set(), []
    for _rid, b in bundles:
        fid, fname = b.get("folder_id"), b.get("folder")
        if fid is None or fid in seen:
            continue
        seen.add(fid)
        out.append((fid, fname))
    return out


def export_zip(conn, recording_ids: list[int]) -> tuple[Path, str] | None:
    """Build a zip on disk for one or more recordings.

    Returns (zip_path, download_name), or None when no id exists. The
    caller owns the temp file and must delete it after sending. A single
    recording exports flat; several export as one folder per recording.
    """
    from app import digest

    bundles = [(rid, _recording_bundle(conn, rid)) for rid in recording_ids]
    bundles = [(rid, b) for rid, b in bundles if b is not None]
    if not bundles:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for rid, b in bundles:
            # Bulk exports mirror the folder structure: recordings in
            # a folder nest under it, unfiled ones sit at the root.
            parent = (f"{_slug(b['folder'], 'folder')}/"
                      if len(bundles) > 1 and b["folder"] else "")
            folder = ("" if len(bundles) == 1 else
                      parent
                      + f"{rid:03d}-{_slug(b['title'], f'recording-{rid}')}/")
            if b["file_path"].exists():
                zf.write(b["file_path"], folder + b["file_path"].name)
            zf.writestr(folder + "transcript.md", b["transcript_md"])
            if b["notes_md"]:
                zf.writestr(folder + "notes.md", b["notes_md"])
            if b["my_notes_md"]:
                zf.writestr(folder + "my-notes.md", b["my_notes_md"])
            if b["enhanced_md"]:
                zf.writestr(folder + "enhanced-notes.md", b["enhanced_md"])

        # One digest.md per folder represented in a bulk export, written
        # next to that folder's recordings. digest.markdown always
        # returns text: a missing or stale digest says so in the file,
        # because an omitted file reads as "this folder has none".
        if len(bundles) > 1:
            for fid, fname in _folders_in(bundles):
                try:
                    body = digest.markdown(conn, fid)
                except LookupError:
                    continue
                zf.writestr(f"{_slug(fname, 'folder')}/digest.md", body)
    tmp.close()

    if len(bundles) == 1:
        rid, b = bundles[0]
        name = f"{_slug(b['title'], f'recording-{rid}')}.zip"
    else:
        name = f"echo-export-{len(bundles)}-recordings.zip"
    return Path(tmp.name), name
