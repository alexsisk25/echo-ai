"""Organization: manual titles, folders, and LLM folder suggestions.

A manual title is pinned: display resolves manual > calendar event >
AI summary, and no auto-titler ever writes the manual column. Folders
hold at most one folder per recording; deleting a folder unfiles its
recordings. Suggestions are stored next to the recording and are never
applied without an explicit accept (privileged recordings classify on
the local model).
"""

import json

from pydantic import BaseModel, ValidationError

from app import db

DEFAULT_FOLDER_IDEAS = "Client calls, Internal, Personal, Interviews"


def list_folders(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT f.id, f.name, count(r.id), COALESCE(f.privileged, 0)
        FROM folders f
        LEFT JOIN recordings r ON r.folder_id = f.id
        GROUP BY f.id ORDER BY f.name
        """
    ).fetchall()
    return [{"id": r[0], "name": r[1], "count": r[2],
             "privileged": bool(r[3])} for r in rows]


def create_folder(conn, name: str) -> dict:
    """Create (or return the existing) folder with this name."""
    name = name.strip()
    if not name:
        raise ValueError("Folder name is empty")
    row = conn.execute(
        "SELECT id, COALESCE(privileged, 0) FROM folders WHERE name = ?",
        (name,),
    ).fetchone()
    if row:
        return {"id": row[0], "name": name, "existing": True,
                "privileged": bool(row[1])}
    cur = conn.execute("INSERT INTO folders (name) VALUES (?)", (name,))
    conn.commit()
    return {"id": cur.lastrowid, "name": name, "existing": False,
            "privileged": False}


def rename_folder(conn, folder_id: int, name: str) -> bool:
    name = name.strip()
    if not name:
        raise ValueError("Folder name is empty")
    dup = conn.execute(
        "SELECT 1 FROM folders WHERE name = ? AND id != ?", (name, folder_id)
    ).fetchone()
    if dup:
        raise FileExistsError(f'A folder named "{name}" already exists')
    cur = conn.execute(
        "UPDATE folders SET name = ? WHERE id = ?", (name, folder_id)
    )
    conn.commit()
    return cur.rowcount == 1


def delete_folder(conn, folder_id: int) -> int | None:
    """Delete a folder; its recordings become unfiled. Returns how many
    were unfiled, or None when the folder does not exist."""
    if conn.execute("SELECT 1 FROM folders WHERE id = ?",
                    (folder_id,)).fetchone() is None:
        return None
    cur = conn.execute(
        "UPDATE recordings SET folder_id = NULL WHERE folder_id = ?",
        (folder_id,),
    )
    conn.execute("DELETE FROM folder_digests WHERE folder_id = ?",
                 (folder_id,))
    conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
    conn.commit()
    return cur.rowcount


def assign_folder(conn, recording_ids: list[int],
                  folder_id: int | None) -> int:
    """File recordings into a folder (or unfile with None). Assigning
    always clears any pending suggestion. Returns rows changed.

    Two side effects the folder owns: every digest whose contents just
    changed (the folder gained and the folders that lost) is marked
    stale, and a recording landing in a privileged folder is marked
    privileged itself, so it can never be analyzed in the cloud.
    """
    from app import digest

    target = None
    if folder_id is not None:
        target = conn.execute(
            "SELECT COALESCE(privileged, 0) FROM folders WHERE id = ?",
            (folder_id,),
        ).fetchone()
        if target is None:
            raise LookupError("Folder not found")
    changed = 0
    touched: set[int] = set()
    for rid in recording_ids:
        previous = conn.execute(
            "SELECT folder_id FROM recordings WHERE id = ?", (rid,)
        ).fetchone()
        rows = conn.execute(
            "UPDATE recordings SET folder_id = ?, suggested_folder = NULL "
            "WHERE id = ?",
            (folder_id, rid),
        ).rowcount
        changed += rows
        if rows and previous and previous[0] is not None \
                and previous[0] != folder_id:
            touched.add(previous[0])
        if rows and folder_id is not None:
            touched.add(folder_id)
            if target[0]:
                conn.execute(
                    "UPDATE recordings SET privileged = 1 WHERE id = ?",
                    (rid,),
                )
    conn.commit()
    for fid in touched:
        digest.mark_stale(conn, fid)
    return changed


def set_folder_privileged(conn, folder_id: int, privileged: bool) -> dict:
    """Mark a whole folder privileged (or stop marking new arrivals).

    Turning it on marks every recording already in the folder privileged
    too, so nothing in it can reach the cloud, and stales the digest
    because its routing just changed. Turning it off only stops future
    arrivals from being marked: recordings already flagged stay flagged,
    since un-privileging someone's recordings behind their back is not
    this switch's job.
    """
    from app import digest

    if conn.execute("SELECT 1 FROM folders WHERE id = ?",
                    (folder_id,)).fetchone() is None:
        raise LookupError("Folder not found")
    conn.execute("UPDATE folders SET privileged = ? WHERE id = ?",
                 (1 if privileged else 0, folder_id))
    marked = 0
    if privileged:
        marked = conn.execute(
            "UPDATE recordings SET privileged = 1 WHERE folder_id = ? "
            "AND privileged = 0",
            (folder_id,),
        ).rowcount
    conn.commit()
    digest.mark_stale(conn, folder_id)
    return {"folder_id": folder_id, "privileged": privileged,
            "recordings_marked": marked}


class FolderSuggestion(BaseModel):
    folder: str


SUGGEST_PROMPT = """You organize meeting recordings into folders. Given a meeting's title, summary, and topics, pick the single best folder name for it.

Prefer one of the user's existing folders when it fits: {folders}
If none fits well, propose one short new folder name in the same spirit (like {ideas}).

Return ONLY a JSON object, no prose and no code fences: {{"folder": "name"}}

Meeting title: {title}
Summary: {summary}
Topics: {topics}"""


def suggest_folder(conn, recording_id: int) -> str | None:
    """Ask the LLM for a folder suggestion and store it. Returns the
    suggestion, or None when the recording has no summary to classify.
    Never assigns; accepting is a separate explicit step."""
    from app import summarize

    row = conn.execute(
        """
        SELECT r.title, r.privileged, s.summary_json,
               (SELECT event_title FROM calendar_matches
                WHERE recording_id = r.id AND status = 'matched')
        FROM recordings r
        LEFT JOIN transcripts t ON t.recording_id = r.id
        LEFT JOIN summaries s ON s.transcript_id = t.id
        WHERE r.id = ?
        """,
        (recording_id,),
    ).fetchone()
    if row is None:
        raise LookupError("Recording not found")
    manual_title, privileged, summary_json, event_title = row
    if not summary_json:
        return None
    summary = json.loads(summary_json)
    folders = [f["name"] for f in list_folders(conn)]

    prompt = SUGGEST_PROMPT.format(
        folders=", ".join(folders) if folders else "(none yet)",
        ideas=DEFAULT_FOLDER_IDEAS,
        title=manual_title or event_title or summary.get("title")
        or "Untitled",
        summary=summary.get("summary", ""),
        topics=", ".join(summary.get("topics", [])),
    )
    raw = summarize._call_llm(prompt, privileged=bool(privileged))
    try:
        suggestion = FolderSuggestion.model_validate_json(
            summarize._strip_fences(raw)
        )
    except (ValidationError, ValueError) as err:
        fixed = summarize._call_llm(
            summarize.CLEANUP_PROMPT.format(error=err, text=raw),
            privileged=bool(privileged),
        )
        suggestion = FolderSuggestion.model_validate_json(
            summarize._strip_fences(fixed)
        )
    name = suggestion.folder.strip()
    if not name:
        return None
    conn.execute(
        "UPDATE recordings SET suggested_folder = ?, "
        "suggestion_dismissed = 0 WHERE id = ?",
        (name, recording_id),
    )
    conn.commit()
    return name


def suggest_for_unfiled(conn, recording_ids: list[int] | None = None) -> dict:
    """Suggest folders for the given recordings, or for every done,
    unfiled recording without a pending or dismissed suggestion."""
    if recording_ids is None:
        recording_ids = [r[0] for r in conn.execute(
            """
            SELECT id FROM recordings
            WHERE status = 'done' AND folder_id IS NULL
              AND suggested_folder IS NULL AND suggestion_dismissed = 0
            """
        ).fetchall()]
    out = {}
    for rid in recording_ids:
        try:
            name = suggest_folder(conn, rid)
        except LookupError:
            continue
        if name:
            out[rid] = name
    return out


def dismiss_suggestion(conn, recording_id: int) -> bool:
    """Clear a suggestion and remember the dismissal so bulk runs do
    not bring it back."""
    cur = conn.execute(
        "UPDATE recordings SET suggested_folder = NULL, "
        "suggestion_dismissed = 1 WHERE id = ?",
        (recording_id,),
    )
    conn.commit()
    return cur.rowcount == 1


def accept_suggestion(conn, recording_id: int) -> dict | None:
    """File the recording into its suggested folder, creating the
    folder if needed. Returns {folder_id, name} or None when there is
    no pending suggestion."""
    row = conn.execute(
        "SELECT suggested_folder FROM recordings WHERE id = ?",
        (recording_id,),
    ).fetchone()
    if row is None or not row[0]:
        return None
    folder = create_folder(conn, row[0])
    assign_folder(conn, [recording_id], folder["id"])
    return {"folder_id": folder["id"], "name": folder["name"]}


def accept_all_suggestions(conn) -> int:
    """Accept every pending suggestion. Returns how many were filed."""
    rids = [r[0] for r in conn.execute(
        "SELECT id FROM recordings WHERE suggested_folder IS NOT NULL"
    ).fetchall()]
    return sum(1 for rid in rids if accept_suggestion(conn, rid))


def search_titles(conn, query: str, limit: int = 20,
                  folder_id: int | None = None) -> list[dict]:
    """Recordings whose resolved title matches the query (manual title,
    calendar event, or AI title)."""
    like = f"%{query}%"
    scope = "" if folder_id is None else " AND r.folder_id = ?"
    params = ([like] if folder_id is None else [like, folder_id]) + [limit]
    rows = conn.execute(
        """
        SELECT DISTINCT r.id, r.file_path, r.created_at,
               COALESCE(r.title,
                        (SELECT event_title FROM calendar_matches
                         WHERE recording_id = r.id AND status = 'matched'),
                        json_extract(s.summary_json, '$.title'))
        FROM recordings r
        LEFT JOIN transcripts t ON t.recording_id = r.id
        LEFT JOIN summaries s ON s.transcript_id = t.id
        WHERE COALESCE(r.title,
                       (SELECT event_title FROM calendar_matches
                        WHERE recording_id = r.id AND status = 'matched'),
                       json_extract(s.summary_json, '$.title'))
              LIKE ?""" + scope + """
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
            "source": "title",
        }
        for r in rows
    ]
