"""Person dossiers: an aggregated profile per named speaker.

A "person" is any speaker label that is not an auto-generated SPEAKER_XX.
The dossier is built by the LLM from that person's meetings (titles,
summaries) and the action items assigned to them, then cached until the
set of meetings changes.
"""

import json

from pydantic import BaseModel, ValidationError

from app import config, db


class PersonDossier(BaseModel):
    name: str
    summary: str
    cares_about: list[str] = []
    commitments_made: list[str] = []
    follow_ups_owed: list[str] = []


def list_people(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT s.speaker, count(DISTINCT t.recording_id) AS n_meetings,
               max(r.created_at)
        FROM segments s
        JOIN transcripts t ON t.id = s.transcript_id
        JOIN recordings r ON r.id = t.recording_id
        WHERE s.speaker IS NOT NULL
          AND s.speaker NOT LIKE 'SPEAKER^_%' ESCAPE '^'
        GROUP BY s.speaker
        ORDER BY n_meetings DESC, s.speaker
        """
    ).fetchall()
    return [
        {"name": r[0], "meetings": r[1], "last_seen": r[2]} for r in rows
    ]


def _owner_matches(owner: str, name: str) -> bool:
    """Loose match: 'Sam' owns items for 'Sam', 'Sam Riley', 'sam'."""
    owner, name = owner.lower().strip(), name.lower().strip()
    if not owner or not name:
        return False
    return owner in name or name in owner or \
        owner.split()[0] == name.split()[0]


def person_data(conn, name: str) -> dict | None:
    """Everything known about one named speaker, from existing data."""
    meetings = conn.execute(
        """
        SELECT DISTINCT r.id, r.created_at, t.id, su.summary_json
        FROM segments s
        JOIN transcripts t ON t.id = s.transcript_id
        JOIN recordings r ON r.id = t.recording_id
        LEFT JOIN summaries su ON su.transcript_id = t.id
        WHERE s.speaker = ?
        ORDER BY r.created_at DESC
        """,
        (name,),
    ).fetchall()
    if not meetings:
        return None

    out_meetings, action_items, topics = [], [], []
    for rec_id, created_at, tid, summary_json in meetings:
        summary = json.loads(summary_json) if summary_json else {}
        out_meetings.append({
            "recording_id": rec_id,
            "created_at": created_at,
            "title": summary.get("title"),
            "date": summary.get("date"),
            "summary": summary.get("summary"),
        })
        topics.extend(summary.get("topics", []))
        for item in summary.get("action_items", []):
            if _owner_matches(item.get("owner", ""), name):
                action_items.append({**item, "recording_id": rec_id,
                                     "title": summary.get("title")})
    return {
        "name": name,
        "meetings": out_meetings,
        "action_items": action_items,
        "topics": sorted(set(topics)),
    }


DOSSIER_PROMPT = """You maintain a profile of a person based on meeting records. Return ONLY a JSON object, no prose and no code fences, with exactly these keys:
- "name": the person's name as given
- "summary": 2-4 sentences on who this person appears to be and their relationship to the user, based only on these meetings
- "cares_about": list of things this person cares about or focuses on, [] if unclear
- "commitments_made": list of concrete things this person committed to do, [] if none
- "follow_ups_owed": list of things others owe this person or open threads with them, [] if none

Do not invent facts that are not in the meeting records.

Person: {name}

Their meetings:
{meetings}"""


def build_dossier(conn, name: str) -> PersonDossier:
    """Generate and cache a dossier for one person via the LLM route.
    If any of their meetings is privileged, the whole dossier is built
    by the local model so privileged text never reaches the cloud."""
    from app import summarize

    data = person_data(conn, name)
    if data is None:
        raise ValueError(f"No meetings found for {name}")

    privileged = any(
        db.is_privileged(conn, m["recording_id"]) for m in data["meetings"]
    )

    lines = []
    for m in data["meetings"]:
        lines.append(
            f"- {m['title'] or 'Untitled'}"
            f"{' (' + (m['date'] or m['created_at'][:10]) + ')'}: "
            f"{m['summary'] or 'no summary'}"
        )
    for a in data["action_items"]:
        lines.append(
            f"- Action item for {name} from {a['title'] or 'a meeting'}: "
            f"{a['task']}"
            f"{' due ' + a['due_date'] if a.get('due_date') else ''}"
        )
    raw = summarize._call_llm(
        DOSSIER_PROMPT.format(name=name, meetings="\n".join(lines)),
        privileged=privileged,
    )
    try:
        dossier = PersonDossier.model_validate_json(
            summarize._strip_fences(raw)
        )
    except (ValidationError, ValueError) as err:
        fixed = summarize._call_llm(
            summarize.CLEANUP_PROMPT.format(error=err, text=raw),
            privileged=privileged,
        )
        dossier = PersonDossier.model_validate_json(
            summarize._strip_fences(fixed)
        )

    conn.execute(
        """
        INSERT INTO dossiers (name, dossier_json, meetings_count)
        VALUES (?, ?, ?)
        ON CONFLICT (name) DO UPDATE SET
            dossier_json = excluded.dossier_json,
            meetings_count = excluded.meetings_count,
            updated_at = datetime('now')
        """,
        (name, dossier.model_dump_json(), len(data["meetings"])),
    )
    conn.commit()
    return dossier


def get_dossier(conn, name: str) -> dict | None:
    """Cached dossier, or None when missing or stale (meeting set grew)."""
    row = conn.execute(
        "SELECT dossier_json, meetings_count FROM dossiers WHERE name = ?",
        (name,),
    ).fetchone()
    if row is None:
        return None
    data = person_data(conn, name)
    if data is None or len(data["meetings"]) != row[1]:
        return None
    return json.loads(row[0])
