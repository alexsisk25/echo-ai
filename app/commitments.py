"""Commitment tracker: every promise from every meeting in one list.

Commitments come from the action items the summarizer already extracts
(owner, task, due date). Each is linked to the transcript segment whose
text best matches the task, found with the local embedding model, so the
UI can jump straight to the moment it was said. Status is open or done.

PLAN.md asks for overdue alerts, and the due date was stored and shown
as text but never compared to today. due_state does that comparison in
one place so the list, the sort, and the filters agree.
"""

import json
from datetime import date, datetime

import numpy as np

from app import config, semantic

MIN_SEGMENT_CHARS = 15
# "Due soon" is the next week, matching how far ahead a person can
# usefully act on a promise made in a meeting.
DUE_SOON_DAYS = 7


def _parse_due(due_date: str | None) -> date | None:
    """A due date the model wrote, or None when there is nothing usable.

    The summarizer is asked for YYYY-MM-DD but is a language model, so
    anything else is treated as no date rather than as an error.
    """
    if not due_date:
        return None
    text = str(due_date).strip()[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def due_state(due_date: str | None, status: str = "open",
              today: date | None = None) -> tuple[str, int | None]:
    """(state, days_until) for one commitment.

    States: "overdue", "due-soon", "later", "none". A commitment that is
    already done is never overdue, whatever its date says: the point of
    the state is what still needs doing.
    """
    parsed = _parse_due(due_date)
    if parsed is None:
        return "none", None
    days = (parsed - (today or date.today())).days
    if status == "done":
        return "later", days
    if days < 0:
        return "overdue", days
    if days <= DUE_SOON_DAYS:
        return "due-soon", days
    return "later", days


def is_mine(owner: str | None, user_name: str | None = None) -> bool:
    """Whether a commitment is the user's own. OTTER_USER_NAME sets the
    name; "Me" is the default the capture pipeline already pins to the
    user's own channel, so it counts either way."""
    name = (user_name or config.USER_NAME or "Me").strip().lower()
    owned = (owner or "").strip().lower()
    if not owned:
        return False
    return owned == name or owned == "me"


# Overdue first and most overdue at the top, then due soon, then dated
# but not urgent, then undated. Done drops below all of it.
_URGENCY = {"overdue": 0, "due-soon": 1, "later": 2, "none": 3}


def _sort_key(item: dict) -> tuple:
    return (
        1 if item["status"] == "done" else 0,
        _URGENCY[item["due_state"]],
        item["days_until"] if item["days_until"] is not None else 10**6,
        -item["id"],
    )


def _best_segment(conn, transcript_id: int, task: str) -> int | None:
    """The segment whose text is semantically closest to the task."""
    rows = conn.execute(
        "SELECT id, text FROM segments WHERE transcript_id = ? "
        "AND length(text) >= ?",
        (transcript_id, MIN_SEGMENT_CHARS),
    ).fetchall()
    if not rows:
        return None
    model = semantic._get_model()
    task_vec = model.encode([task], normalize_embeddings=True)[0]
    seg_vecs = model.encode([r[1] for r in rows], normalize_embeddings=True)
    return rows[int(np.argmax(seg_vecs @ task_vec))][0]


def extract(conn, transcript_id: int) -> int:
    """Store commitments from one transcript's summary. Returns how many
    are new. Safe to rerun; existing rows and their status are kept."""
    row = conn.execute(
        "SELECT summary_json FROM summaries WHERE transcript_id = ?",
        (transcript_id,),
    ).fetchone()
    if row is None:
        return 0
    items = json.loads(row[0]).get("action_items", [])
    added = 0
    for item in items:
        owner = (item.get("owner") or "unassigned").strip()
        task = (item.get("task") or "").strip()
        if not task:
            continue
        exists = conn.execute(
            "SELECT 1 FROM commitments WHERE transcript_id = ? "
            "AND owner = ? AND task = ?",
            (transcript_id, owner, task),
        ).fetchone()
        if exists:
            continue
        segment_id = _best_segment(conn, transcript_id, task)
        conn.execute(
            "INSERT INTO commitments (transcript_id, owner, task, due_date, "
            "priority, segment_id) VALUES (?, ?, ?, ?, ?, ?)",
            (transcript_id, owner, task, item.get("due_date"),
             item.get("priority"), segment_id),
        )
        added += 1
    conn.commit()
    return added


def backfill(conn) -> int:
    """Extract commitments from every summarized transcript."""
    tids = [r[0] for r in conn.execute(
        "SELECT transcript_id FROM summaries"
    ).fetchall()]
    return sum(extract(conn, tid) for tid in tids)


def list_commitments(conn, status: str | None = None,
                     owner: str | None = None, due: str | None = None,
                     mine: bool | None = None,
                     today: date | None = None) -> list[dict]:
    """Every commitment, most urgent first.

    due filters to "overdue" or "due-soon"; mine filters to (or away
    from) commitments owned by OTTER_USER_NAME. Both are applied in
    Python rather than SQL because urgency depends on today's date and
    on the status, not on a stored column.
    """
    where, params = [], []
    if status:
        where.append("c.status = ?")
        params.append(status)
    if owner:
        where.append("c.owner = ?")
        params.append(owner)
    rows = conn.execute(
        f"""
        SELECT c.id, c.owner, c.task, c.due_date, c.priority, c.status,
               c.transcript_id, t.recording_id, su.summary_json,
               seg.start_seconds
        FROM commitments c
        JOIN transcripts t ON t.id = c.transcript_id
        LEFT JOIN summaries su ON su.transcript_id = t.id
        LEFT JOIN segments seg ON seg.id = c.segment_id
        {'WHERE ' + ' AND '.join(where) if where else ''}
        """,
        params,
    ).fetchall()
    out = []
    for r in rows:
        summary = json.loads(r[8]) if r[8] else {}
        state, days = due_state(r[3], r[5], today=today)
        out.append({
            "id": r[0], "owner": r[1], "task": r[2], "due_date": r[3],
            "priority": r[4], "status": r[5], "transcript_id": r[6],
            "recording_id": r[7], "meeting_title": summary.get("title"),
            "meeting_date": summary.get("date"),
            "segment_start": r[9],
            "due_state": state,
            "days_until": days,
            "mine": is_mine(r[1]),
        })
    if due in ("overdue", "due-soon"):
        out = [i for i in out if i["due_state"] == due]
    if mine is True:
        out = [i for i in out if i["mine"]]
    elif mine is False:
        out = [i for i in out if not i["mine"]]
    out.sort(key=_sort_key)
    return out


def set_status(conn, commitment_id: int, status: str) -> bool:
    if status not in ("open", "done"):
        raise ValueError("status must be open or done")
    cur = conn.execute(
        "UPDATE commitments SET status = ? WHERE id = ?",
        (status, commitment_id),
    )
    conn.commit()
    return cur.rowcount == 1
