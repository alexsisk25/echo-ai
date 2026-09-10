"""LLM meeting notes: transcript text in, validated summary JSON out.

Only plain text goes to the cloud model through LiteLLM. Audio never
leaves the machine. The response must match the MeetingSummary schema;
a malformed response gets one cleanup pass before we give up.
"""

import json

from pydantic import BaseModel, ValidationError

from app import config


class ActionItem(BaseModel):
    owner: str
    task: str
    due_date: str | None = None
    priority: str | None = None


class MeetingSummary(BaseModel):
    title: str
    date: str | None = None
    attendees: list[str] = []
    summary: str
    decisions: list[str] = []
    action_items: list[ActionItem] = []
    risks: list[str] = []
    topics: list[str] = []


PROMPT = """You are a meeting analyst. Read the transcript below and return ONLY a JSON object, no prose and no code fences, with exactly these keys:
- "title": a short descriptive title for the meeting
- "date": the meeting date in YYYY-MM-DD if stated in the transcript, else null
- "attendees": list of participant names mentioned, [] if none
- "summary": 2-5 sentence summary of what was discussed
- "decisions": list of decisions that were made, [] if none
- "action_items": list of objects with keys "owner", "task", "due_date" (YYYY-MM-DD or null), "priority" ("high", "medium", "low", or null). If a task has no clear owner, use "unassigned".
- "risks": list of risks or concerns raised, [] if none
- "topics": list of short topic labels, [] if none

Do not invent facts that are not in the transcript.

Transcript:
{transcript}"""

CLEANUP_PROMPT = """The following text was supposed to be a single valid JSON object but failed to parse or validate. Fix it and return ONLY the corrected JSON object, no prose and no code fences. The validation error was: {error}

Text:
{text}"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _call_llm(prompt: str, privileged: bool = False) -> str:
    """All calls go through the llm gateway, which enforces the privacy
    vault. Kept as a module function so tests can fake it.

    Newer Claude models reject the temperature parameter, so determinism
    comes from the strict prompt plus schema validation instead.
    """
    from app import llm

    return llm.complete(prompt, privileged=privileged)


def summarize(transcript_text: str,
              privileged: bool = False) -> MeetingSummary:
    """Summarize a transcript, with one cleanup pass on malformed JSON."""
    raw = _call_llm(PROMPT.format(transcript=transcript_text),
                    privileged=privileged)
    try:
        return MeetingSummary.model_validate_json(_strip_fences(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as first_error:
        fixed = _call_llm(CLEANUP_PROMPT.format(error=first_error, text=raw),
                          privileged=privileged)
        return MeetingSummary.model_validate_json(_strip_fences(fixed))
