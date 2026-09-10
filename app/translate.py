"""Spanish/English translation of transcripts and AI notes.

Translations go through the llm gateway, so privileged recordings are
translated by the local model only. Results are cached per transcript
and language.
"""

import json
import logging

from pydantic import ValidationError

from app import summarize
from app.summarize import MeetingSummary

log = logging.getLogger("otter.translate")

LANG_NAMES = {"en": "English", "es": "Spanish"}
CHUNK_CHARS = 4000

TEXT_PROMPT = """Translate the following meeting transcript text to {language}. Return ONLY the translation, no preamble and no notes. Keep the speaker's tone and meaning; do not summarize or omit anything.

Text:
{text}"""

SUMMARY_PROMPT = """Translate every string value in this JSON object to {language}. Return ONLY the translated JSON object with exactly the same keys and structure, no prose and no code fences. Keep dates, numbers, and the JSON structure unchanged.

JSON:
{summary}"""


def _chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    text = text.strip()
    out = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        out.append(text[start:end])
        start = end
    return out


def translate_recording(conn, recording_id: int, target: str,
                        privileged: bool = False) -> dict:
    """Translate a recording's transcript and notes. Cached per language."""
    if target not in LANG_NAMES:
        raise ValueError("target must be en or es")
    row = conn.execute(
        """
        SELECT t.id, t.full_text, s.summary_json FROM transcripts t
        LEFT JOIN summaries s ON s.transcript_id = t.id
        WHERE t.recording_id = ?
        """,
        (recording_id,),
    ).fetchone()
    if row is None:
        raise LookupError("No transcript for this recording")
    transcript_id, full_text, summary_json = row

    cached = get_translation(conn, transcript_id, target)
    if cached:
        return cached

    language = LANG_NAMES[target]
    translated_text = "\n".join(
        summarize._call_llm(
            TEXT_PROMPT.format(language=language, text=chunk),
            privileged=privileged,
        ).strip()
        for chunk in _chunks(full_text)
    )

    translated_summary = _translate_summary(summary_json, language,
                                            privileged)

    conn.execute(
        "INSERT OR REPLACE INTO translations "
        "(transcript_id, lang, full_text, summary_json) VALUES (?, ?, ?, ?)",
        (transcript_id, target, translated_text, translated_summary),
    )
    conn.commit()
    return get_translation(conn, transcript_id, target)


def _translate_summary(summary_json: str | None, language: str,
                       privileged: bool) -> str | None:
    """The AI notes in the target language, or None when the model will
    not produce valid JSON.

    Malformed JSON gets one cleanup pass, the same as summaries and
    enhanced notes. If it still will not parse, the translation is
    delivered without translated notes rather than failing outright:
    the transcript is the thing the user asked for, and losing it to a
    bad JSON response would be the worse outcome.
    """
    if not summary_json:
        return None
    raw = summarize._call_llm(
        SUMMARY_PROMPT.format(language=language, summary=summary_json),
        privileged=privileged,
    )
    try:
        return MeetingSummary.model_validate_json(
            summarize._strip_fences(raw)).model_dump_json()
    except (ValidationError, ValueError) as first_error:
        try:
            fixed = summarize._call_llm(
                summarize.CLEANUP_PROMPT.format(error=first_error, text=raw),
                privileged=privileged,
            )
            return MeetingSummary.model_validate_json(
                summarize._strip_fences(fixed)).model_dump_json()
        except (ValidationError, ValueError):
            log.warning("Translated notes could not be parsed for %s; the "
                        "transcript translation is kept without them",
                        language)
            return None


def get_translation(conn, transcript_id: int, lang: str) -> dict | None:
    row = conn.execute(
        "SELECT full_text, summary_json FROM translations "
        "WHERE transcript_id = ? AND lang = ?",
        (transcript_id, lang),
    ).fetchone()
    if row is None:
        return None
    return {
        "lang": lang,
        "full_text": row[0],
        "summary": json.loads(row[1]) if row[1] else None,
    }
