"""Hybrid notes: the user's own jottings merged with the transcript.

The raw notes belong to the user and are never overwritten by the app.
Enhance sends the jottings plus the transcript through the existing LLM
gateway (privileged recordings use the local model) and stores the
structured result separately. Re-enhancing with unchanged inputs
returns the stored result without another LLM call.
"""

import hashlib
import json

from pydantic import BaseModel, ValidationError

from app import db


class NoteSection(BaseModel):
    heading: str
    bullets: list[str] = []


class EnhancedNotes(BaseModel):
    sections: list[NoteSection]


ENHANCE_PROMPT = """You turn a person's rough meeting jottings into structured notes, using the meeting transcript as evidence. The jottings are what the person flagged as important, so every jotted fragment must appear, expanded with specifics from the transcript (who said what, decisions, numbers, dates). After covering the jottings, you may add one short final section titled "Also discussed" with other notable points from the transcript. Do not invent anything that is in neither the jottings nor the transcript.

Return ONLY a JSON object, no prose and no code fences, shaped exactly like:
{{"sections": [{{"heading": "short heading", "bullets": ["point", "point"]}}]}}

Their jottings:
{notes}

Transcript:
{transcript}"""


def get(conn, recording_id: int) -> dict:
    row = conn.execute(
        "SELECT raw_text, enhanced_json, enhanced_at FROM notes "
        "WHERE recording_id = ?",
        (recording_id,),
    ).fetchone()
    if row is None:
        return {"raw_text": "", "enhanced": None, "enhanced_at": None}
    return {
        "raw_text": row[0],
        "enhanced": json.loads(row[1]) if row[1] else None,
        "enhanced_at": row[2],
    }


def save(conn, recording_id: int, text: str) -> None:
    """Upsert the user's raw notes. The enhanced version is kept as-is
    (possibly stale) until the user enhances again."""
    conn.execute(
        """
        INSERT INTO notes (recording_id, raw_text) VALUES (?, ?)
        ON CONFLICT (recording_id) DO UPDATE SET
            raw_text = excluded.raw_text,
            updated_at = datetime('now')
        """,
        (recording_id, text),
    )
    conn.commit()


def _input_hash(raw_text: str, transcript: str) -> str:
    h = hashlib.sha256()
    h.update(raw_text.encode())
    h.update(b"\0")
    h.update(transcript.encode())
    return h.hexdigest()


def enhance(conn, recording_id: int) -> dict:
    """Merge the raw notes with the transcript into structured notes.

    Raises LookupError when the recording has no transcript and
    ValueError when there are no notes to enhance.
    """
    from app import summarize

    current = conn.execute(
        "SELECT raw_text, enhanced_json, enhanced_hash FROM notes "
        "WHERE recording_id = ?",
        (recording_id,),
    ).fetchone()
    raw_text = current[0] if current else ""
    if not raw_text.strip():
        raise ValueError("There are no notes to enhance yet.")

    t_row = conn.execute(
        "SELECT full_text FROM transcripts WHERE recording_id = ?",
        (recording_id,),
    ).fetchone()
    if t_row is None:
        raise LookupError("No transcript for this recording.")
    transcript = t_row[0]

    fingerprint = _input_hash(raw_text, transcript)
    if current and current[1] and current[2] == fingerprint:
        return {"enhanced": json.loads(current[1]), "cached": True}

    privileged = db.is_privileged(conn, recording_id)
    prompt = ENHANCE_PROMPT.format(notes=raw_text, transcript=transcript)
    raw_out = summarize._call_llm(prompt, privileged=privileged)
    try:
        enhanced = EnhancedNotes.model_validate_json(
            summarize._strip_fences(raw_out)
        )
    except (ValidationError, ValueError) as err:
        fixed = summarize._call_llm(
            summarize.CLEANUP_PROMPT.format(error=err, text=raw_out),
            privileged=privileged,
        )
        enhanced = EnhancedNotes.model_validate_json(
            summarize._strip_fences(fixed)
        )

    conn.execute(
        """
        UPDATE notes SET enhanced_json = ?, enhanced_hash = ?,
            enhanced_at = datetime('now')
        WHERE recording_id = ?
        """,
        (enhanced.model_dump_json(), fingerprint, recording_id),
    )
    conn.commit()
    return {"enhanced": enhanced.model_dump(), "cached": False}


def enhanced_markdown(enhanced: dict) -> str:
    """The enhanced notes as markdown, for export."""
    lines = ["# Enhanced notes", ""]
    for section in enhanced.get("sections", []):
        lines += [f"## {section.get('heading', '')}", ""]
        lines += [f"- {b}" for b in section.get("bullets", [])]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
