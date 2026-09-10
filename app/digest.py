"""Folder digests: a longitudinal view of everything in one folder.

Built hierarchically so cost stays low and long folders are possible.
Level 1 is free: each recording contributes its existing summary (never
the raw transcript), plus a few real quotable lines with timestamps. A
recording with no summary falls back to chunked transcript retrieval.
Level 2 only exists for long folders: chronological batches are
condensed into period briefs. Level 3 synthesizes the digest.

Updates are incremental. Adding a recording marks the digest stale; the
update pass sees the stored digest plus only the new recordings, so a
folder of a hundred meetings does not get rebuilt to absorb one.

Privacy: a folder can be privileged as a whole, and a folder holding any
privileged recording is treated as privileged. Every LLM call for such a
folder goes to the local model through the gateway, which fails closed
rather than falling back to the cloud.

Tone is deliberately descriptive: the prompts report themes, changes and
open threads, and are told not to assess the user or interpret them.
"""

import json
import re

from pydantic import BaseModel, ValidationError

from app import config

# Recordings per intermediate pass. Above this a folder is condensed
# into chronological period briefs before the final synthesis.
BATCH_SIZE = 12
# Characters of transcript used for a recording that has no summary.
FALLBACK_CHARS = 1800
# Real lines offered to the model as quote material, per recording.
QUOTE_CANDIDATES = 3
QUOTE_MAX_CHARS = 240


class Theme(BaseModel):
    theme: str
    first_appeared: str | None = None
    framing_then: str = ""
    framing_now: str = ""
    how_it_changed: str = ""


class Shift(BaseModel):
    date: str | None = None
    shift: str
    note: str = ""


class OpenThread(BaseModel):
    thread: str
    raised_on: str | None = None
    last_mentioned: str | None = None
    note: str = ""


class DigestCommitment(BaseModel):
    owner: str
    commitment: str
    made_on: str | None = None
    status: str = "unclear"
    note: str = ""


class Quote(BaseModel):
    quote: str
    speaker: str | None = None
    recording_id: int | None = None
    timestamp: str | None = None
    start_seconds: float | None = None
    why: str = ""


class FolderDigest(BaseModel):
    overview: str
    themes: list[Theme] = []
    shifts: list[Shift] = []
    open_threads: list[OpenThread] = []
    commitments: list[DigestCommitment] = []
    quotes: list[Quote] = []


TONE = """Be descriptive, not diagnostic. Report what was said, when it changed, and what is unresolved. Do not assess, diagnose, or psychoanalyse the people in these recordings, do not offer advice or clinical interpretation, and do not judge whether anything was a good idea. Describe the record, nothing more."""

DIGEST_PROMPT = """You write a longitudinal digest of one folder of a person's own meeting recordings. The material below is in chronological order, oldest first, and each block carries its date.

{tone}

Your job is TRAJECTORY, not aggregation. A list of what each meeting was about is a failure. Show how the folder moved: what keeps coming back, when each thing first appeared, how the framing of it changed, what was raised and never resolved, and what people committed to.

Return ONLY a JSON object, no prose and no code fences, with exactly these keys:
- "overview": 3-6 sentences describing the arc of this folder over time, with dates.
- "themes": list of objects {{"theme", "first_appeared" (YYYY-MM-DD of the recording it first appears in, or null), "framing_then" (how it was talked about then), "framing_now" (how it is talked about in the most recent material), "how_it_changed" (one or two sentences; say "unchanged" if the framing held steady)}}. Only include something that appears in more than one recording.
- "shifts": list of objects {{"date" (YYYY-MM-DD), "shift" (what changed on that date), "note" (what it followed from)}}. A shift is a notable change of direction, position, plan, or tone that can be pinned to a date.
- "open_threads": list of objects {{"thread", "raised_on" (YYYY-MM-DD or null), "last_mentioned" (YYYY-MM-DD or null), "note"}}. These are questions, decisions, or follow-ups that were raised and never resolved in later material. Leave the list empty if everything was resolved.
- "commitments": list of objects {{"owner", "commitment", "made_on" (YYYY-MM-DD or null), "status" ("open", "done", or "unclear"), "note" (the evidence for that status)}}. Mark done only when later material shows it happened.
- "quotes": list of objects {{"quote", "speaker", "recording_id" (integer), "timestamp" ("HH:MM:SS"), "why" (one short line on what it represents)}}. Pick 3-6 lines that represent the folder. Copy each quote VERBATIM from a "Quotable lines" entry and keep its recording id and timestamp exactly as given. Never invent a quote, a recording id, or a timestamp; an unquoted point belongs in a theme, not here.

Do not invent facts, dates, or people that are not in the material.

Folder: {folder}
Recordings covered: {count}

Material:
{material}"""

BATCH_PROMPT = """You are condensing part of a person's meeting folder so a later pass can trace how the folder changed over time. The material below is chronological, oldest first.

{tone}

Write a compact brief of this period in plain prose, at most 350 words, covering: what was recurring, the dates things first came up, how framing changed within the period, anything raised and left unresolved, and who committed to what. Keep every date you use.

Then, under a final line reading "Quotable lines:", copy up to 4 of the most representative lines from the "Quotable lines" entries VERBATIM, each on its own line and in exactly the form they were given, including the [rec N @ HH:MM:SS] marker. Do not write any other section.

Period {period}

Material:
{material}"""

UPDATE_PROMPT = """You are updating an existing digest of a person's meeting folder with recordings added since it was written. You do NOT have the earlier recordings, only the digest built from them, so preserve what it says unless the new material changes it.

{tone}

Update in place:
- Keep every existing theme unless the new material contradicts it. When a theme comes up again, update "framing_now" and "how_it_changed" and leave "first_appeared" and "framing_then" alone. Add a theme only when the new material shows it recurring.
- Add a shift for anything in the new material that changed direction, with its date.
- Carry the open threads forward. Remove one only when the new material resolves it, and when it does, note the resolution in the overview. Add threads the new material leaves open.
- Carry commitments forward and update their status when the new material shows progress. Add new commitments.
- Keep the existing quotes. Add one only from a "Quotable lines" entry in the new material, copied VERBATIM with its recording id and timestamp exactly as given.
- Rewrite the overview so it describes the whole arc including the new material.

Return ONLY the complete updated JSON object in the same shape as the digest below, no prose and no code fences. Do not invent facts, dates, or people.

Folder: {folder}

The digest as it stands (built from {covered} earlier recording(s)):
{existing}

New material, chronological, oldest first:
{material}"""


def _hms(seconds: float) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def folder_info(conn, folder_id: int) -> dict:
    """The folder itself: id, name, and whether it is privileged."""
    row = conn.execute(
        "SELECT id, name, COALESCE(privileged, 0) FROM folders WHERE id = ?",
        (folder_id,),
    ).fetchone()
    if row is None:
        raise LookupError("Folder not found")
    return {"id": row[0], "name": row[1], "privileged": bool(row[2])}


def folder_is_privileged(conn, folder_id: int) -> bool:
    """True when the folder itself is marked privileged, or when any
    recording in it is. Either way every LLM call for this folder, and
    its digest, must run on the local model."""
    if folder_info(conn, folder_id)["privileged"]:
        return True
    row = conn.execute(
        "SELECT 1 FROM recordings WHERE folder_id = ? AND privileged = 1 "
        "LIMIT 1",
        (folder_id,),
    ).fetchone()
    return row is not None


def folder_recordings(conn, folder_id: int) -> list[dict]:
    """Every processed recording in the folder, oldest first."""
    rows = conn.execute(
        """
        SELECT r.id,
               COALESCE(r.recorded_at, r.created_at) AS happened_at,
               COALESCE(r.title,
                        (SELECT event_title FROM calendar_matches
                         WHERE recording_id = r.id AND status = 'matched'),
                        json_extract(s.summary_json, '$.title')),
               s.summary_json, t.id
        FROM recordings r
        LEFT JOIN transcripts t ON t.recording_id = r.id
        LEFT JOIN summaries s ON s.transcript_id = t.id
        WHERE r.folder_id = ? AND r.status = 'done'
        ORDER BY happened_at, r.id
        """,
        (folder_id,),
    ).fetchall()
    out: dict[int, dict] = {}
    for rec_id, happened_at, title, summary_json, transcript_id in rows:
        # A recording with several transcripts contributes once; prefer
        # the row that actually carries a summary.
        if rec_id in out and not summary_json:
            continue
        out[rec_id] = {
            "recording_id": rec_id,
            "date": (happened_at or "")[:10],
            "title": title or "Untitled",
            "summary": json.loads(summary_json) if summary_json else None,
            "transcript_id": transcript_id,
        }
    return list(out.values())


def _quote_candidates(conn, rec: dict) -> list[str]:
    """A few real lines from the recording, with speaker and timestamp,
    so the model can quote without ever being handed the transcript."""
    if rec["transcript_id"] is None:
        return []
    rows = conn.execute(
        """
        SELECT start_seconds, speaker, text FROM segments
        WHERE transcript_id = ? AND length(text) > 40
        ORDER BY length(text) DESC LIMIT ?
        """,
        (rec["transcript_id"], QUOTE_CANDIDATES),
    ).fetchall()
    lines = []
    for start, speaker, text in sorted(rows, key=lambda r: r[0]):
        clean = " ".join((text or "").split())[:QUOTE_MAX_CHARS]
        who = speaker or "Unknown"
        lines.append(f'  [rec {rec["recording_id"]} @ {_hms(start)}] '
                     f'{who}: "{clean}"')
    return lines


def _transcript_excerpt(conn, transcript_id: int) -> str:
    """Fallback for a recording with no summary: its stored chunks,
    sampled across the whole transcript so the excerpt is not just the
    opening minutes. Falls back to the raw text when it is unindexed."""
    chunks = [r[0] for r in conn.execute(
        "SELECT text FROM chunks WHERE transcript_id = ? ORDER BY id",
        (transcript_id,),
    ).fetchall()]
    if not chunks:
        row = conn.execute(
            "SELECT full_text FROM transcripts WHERE id = ?",
            (transcript_id,),
        ).fetchone()
        return (row[0] or "")[:FALLBACK_CHARS].strip() if row else ""
    keep = max(1, FALLBACK_CHARS // max(len(chunks[0]), 1))
    if len(chunks) > keep:
        step = len(chunks) / keep
        chunks = [chunks[int(i * step)] for i in range(keep)]
    text = "\n…\n".join(chunks)
    return text[:FALLBACK_CHARS].strip()


def _source_block(conn, rec: dict) -> str:
    """One recording's contribution: its summary (never the transcript),
    or a chunked excerpt when no summary exists, plus quote material."""
    head = (f'[recording {rec["recording_id"]}] {rec["date"] or "undated"} '
            f'- {rec["title"]}')
    lines = [head]
    summary = rec["summary"]
    if summary:
        lines.append(f'Summary: {summary.get("summary", "")}')
        if summary.get("decisions"):
            lines.append("Decisions: "
                         + "; ".join(summary["decisions"]))
        items = summary.get("action_items") or []
        if items:
            lines.append("Action items: " + "; ".join(
                f'{i.get("owner", "unassigned")} - {i.get("task", "")}'
                + (f' (due {i["due_date"]})' if i.get("due_date") else "")
                for i in items))
        if summary.get("risks"):
            lines.append("Risks: " + "; ".join(summary["risks"]))
        if summary.get("topics"):
            lines.append("Topics: " + ", ".join(summary["topics"]))
    elif rec["transcript_id"] is not None:
        excerpt = _transcript_excerpt(conn, rec["transcript_id"])
        if excerpt:
            lines.append("No summary for this recording. Transcript "
                         f"excerpt:\n{excerpt}")
    quotes = _quote_candidates(conn, rec)
    if quotes:
        lines.append("Quotable lines:")
        lines.extend(quotes)
    return "\n".join(lines)


def _call(prompt: str, privileged: bool) -> str:
    from app import summarize

    return summarize._call_llm(prompt, privileged=privileged)


def _validate(raw: str, privileged: bool) -> FolderDigest:
    """Parse the model's JSON, with one cleanup pass, same as summaries."""
    from app import summarize

    try:
        return FolderDigest.model_validate_json(summarize._strip_fences(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as err:
        fixed = _call(
            summarize.CLEANUP_PROMPT.format(error=err, text=raw), privileged)
        return FolderDigest.model_validate_json(
            summarize._strip_fences(fixed))


def _period_label(batch: list[dict], index: int, total: int) -> str:
    first = batch[0]["date"] or "undated"
    last = batch[-1]["date"] or "undated"
    return f"{index} of {total}, {first} to {last}"


def _condense(conn, recs: list[dict], privileged: bool) -> list[str]:
    """Level 2: chronological batches condensed into period briefs, so a
    long folder never goes to the model as one giant prompt."""
    batches = [recs[i:i + BATCH_SIZE]
               for i in range(0, len(recs), BATCH_SIZE)]
    briefs = []
    for n, batch in enumerate(batches, start=1):
        material = "\n\n".join(_source_block(conn, r) for r in batch)
        brief = _call(BATCH_PROMPT.format(
            tone=TONE,
            period=_period_label(batch, n, len(batches)),
            material=material,
        ), privileged)
        briefs.append(f"Period {_period_label(batch, n, len(batches))}:\n"
                      f"{brief.strip()}")
    return briefs


def _material(conn, recs: list[dict], privileged: bool) -> str:
    """Chronological material for a synthesis pass, condensed first when
    the folder is long enough to need it."""
    if len(recs) > BATCH_SIZE:
        return "\n\n".join(_condense(conn, recs, privileged))
    return "\n\n".join(_source_block(conn, r) for r in recs)


def _full_pass(conn, folder: dict, recs: list[dict],
               privileged: bool) -> FolderDigest:
    raw = _call(DIGEST_PROMPT.format(
        tone=TONE, folder=folder["name"], count=len(recs),
        material=_material(conn, recs, privileged),
    ), privileged)
    return _validate(raw, privileged)


def _update_pass(conn, folder: dict, existing: dict, new_recs: list[dict],
                 covered: int, privileged: bool) -> FolderDigest:
    raw = _call(UPDATE_PROMPT.format(
        tone=TONE, folder=folder["name"], covered=covered,
        existing=json.dumps(existing, indent=2),
        material=_material(conn, new_recs, privileged),
    ), privileged)
    return _validate(raw, privileged)


_PUNCT = re.compile(r"[^a-z0-9 ]+")


def _normalize(text: str) -> str:
    return _PUNCT.sub("", " ".join((text or "").lower().split()))


def _link_quotes(conn, digest: FolderDigest, recs: list[dict]) -> FolderDigest:
    """Anchor every quote to a real line. A quote is kept only when it is
    found in a segment of a recording in this folder, and its recording,
    speaker and timestamp then come from that segment rather than from
    the model. Anything unlocatable is dropped: an invented quote with a
    plausible timestamp would be worse than no quote."""
    by_id = {r["recording_id"]: r for r in recs
             if r["transcript_id"] is not None}
    segments: dict[int, list] = {}

    def lines_for(rec_id: int) -> list:
        if rec_id not in segments:
            segments[rec_id] = [
                (start, speaker, _normalize(text))
                for start, speaker, text in conn.execute(
                    "SELECT start_seconds, speaker, text FROM segments "
                    "WHERE transcript_id = ? ORDER BY start_seconds",
                    (by_id[rec_id]["transcript_id"],),
                ).fetchall()
            ]
        return segments[rec_id]

    kept = []
    for quote in digest.quotes:
        needle = _normalize(quote.quote)
        if len(needle) < 12:
            continue
        # The quote must be contained in a line that was actually said.
        # Containment the other way round (a real line sitting inside a
        # longer "quote") is refused on purpose: that is how a true
        # sentence with invented words appended would pass.
        if quote.recording_id in by_id:
            matches = [(quote.recording_id, start, speaker)
                       for start, speaker, hay in
                       lines_for(quote.recording_id) if needle in hay]
        else:
            # No usable recording id, so the folder is searched. If more
            # than one recording said it, there is no way to know who is
            # being quoted, and a confident wrong attribution is worse
            # than no quote.
            matches = [(rec_id, start, speaker) for rec_id in by_id
                       for start, speaker, hay in lines_for(rec_id)
                       if needle in hay]
            if len({m[0] for m in matches}) > 1:
                continue
        if not matches:
            continue
        rec_id, start, speaker = matches[0]
        quote.recording_id = rec_id
        quote.start_seconds = float(start)
        quote.timestamp = _hms(start)
        quote.speaker = speaker or quote.speaker
        kept.append(quote)
    digest.quotes = kept
    return digest


def get(conn, folder_id: int) -> dict | None:
    """The stored digest for a folder, or None when it has never been
    generated. Carries how many recordings it covers and how many have
    landed in the folder since."""
    row = conn.execute(
        "SELECT digest_json, covered_ids, recordings_count, stale, model, "
        "updated_at FROM folder_digests WHERE folder_id = ?",
        (folder_id,),
    ).fetchone()
    if row is None:
        return None
    covered = json.loads(row[1])
    current = [r["recording_id"] for r in folder_recordings(conn, folder_id)]
    new_since = [rid for rid in current if rid not in set(covered)]
    # A recording can leave the folder's set without any of the callers
    # that mark staleness running (it can also stop being 'done', for
    # instance while it reprocesses), so removals are recomputed here
    # rather than trusted to the flag.
    dropped = [rid for rid in covered if rid not in set(current)]
    return {
        "folder_id": folder_id,
        "digest": json.loads(row[0]),
        "covered_ids": covered,
        "recordings_count": row[2],
        "stale": bool(row[3]) or bool(new_since) or bool(dropped),
        "model": row[4],
        "updated_at": row[5],
        "new_since": len(new_since),
        "dropped_since": len(dropped),
    }


def mark_stale(conn, folder_id: int | None) -> bool:
    """Flag a folder's digest as out of date. Called whenever the folder's
    contents change; the digest itself is left in place so the view keeps
    showing the last one until it is updated."""
    if folder_id is None:
        return False
    cur = conn.execute(
        "UPDATE folder_digests SET stale = 1 WHERE folder_id = ?",
        (folder_id,),
    )
    conn.commit()
    return cur.rowcount > 0


def mark_stale_for_recording(conn, transcript_id: int) -> bool:
    """Stale the digest of whatever folder this transcript's recording
    is filed in, if any. Used when the material a digest was built from
    changes underneath it."""
    row = conn.execute(
        "SELECT r.folder_id FROM recordings r "
        "JOIN transcripts t ON t.recording_id = r.id WHERE t.id = ?",
        (transcript_id,),
    ).fetchone()
    return mark_stale(conn, row[0] if row else None)


def delete(conn, folder_id: int) -> None:
    conn.execute("DELETE FROM folder_digests WHERE folder_id = ?",
                 (folder_id,))
    conn.commit()


def _store(conn, folder_id: int, digest: FolderDigest, covered: list[int],
           privileged: bool) -> None:
    conn.execute(
        """
        INSERT INTO folder_digests
            (folder_id, digest_json, covered_ids, recordings_count, stale,
             model, updated_at)
        VALUES (?, ?, ?, ?, 0, ?, datetime('now'))
        ON CONFLICT (folder_id) DO UPDATE SET
            digest_json = excluded.digest_json,
            covered_ids = excluded.covered_ids,
            recordings_count = excluded.recordings_count,
            stale = 0,
            model = excluded.model,
            updated_at = datetime('now')
        """,
        (folder_id, digest.model_dump_json(), json.dumps(covered),
         len(covered), "local" if privileged else config.LLM_MODEL),
    )
    conn.commit()


def generate(conn, folder_id: int, force: bool = False) -> dict:
    """Build the folder's digest, or update it with what is new.

    Returns the stored digest exactly as get() would, plus "cached" (the
    stored digest was still current) and "incremental" (only the new
    recordings went to the model). Raises LookupError for an unknown
    folder and ValueError when there is nothing processed to digest.
    """
    folder = folder_info(conn, folder_id)
    recs = folder_recordings(conn, folder_id)
    if not recs:
        raise ValueError("This folder has no processed recordings yet.")

    privileged = folder_is_privileged(conn, folder_id)
    existing = get(conn, folder_id)
    covered = set(existing["covered_ids"]) if existing else set()
    current_ids = [r["recording_id"] for r in recs]
    new_recs = [r for r in recs if r["recording_id"] not in covered]
    removed = covered - set(current_ids)

    if existing and not force and not new_recs and not removed \
            and not existing["stale"]:
        return {**existing, "cached": True, "incremental": False}

    # Incremental is only safe while the digest still describes every
    # recording it was built from: something removed means the stored
    # text may describe material that is gone, so that rebuilds.
    incremental = bool(existing and not force and new_recs and not removed)
    if incremental:
        digest = _update_pass(conn, folder, existing["digest"], new_recs,
                              len(covered), privileged)
    else:
        digest = _full_pass(conn, folder, recs, privileged)

    digest = _link_quotes(conn, digest, recs)
    _store(conn, folder_id, digest, current_ids, privileged)
    return {**get(conn, folder_id), "cached": False,
            "incremental": incremental}


def markdown(conn, folder_id: int) -> str:
    """The folder digest as a Markdown file for exports.

    Always returns something. A folder with no digest, or one whose
    digest is out of date, says so in the file: an export that silently
    dropped the digest would read as "this folder has no digest", which
    is a different and worse claim.
    """
    folder = folder_info(conn, folder_id)
    stored = get(conn, folder_id)
    out = [f"# {folder['name']} digest", ""]

    if stored is None:
        in_folder = len(folder_recordings(conn, folder_id))
        out += [
            "No digest has been generated for this folder yet.",
            "",
            f"{in_folder} recording{'' if in_folder == 1 else 's'} are filed "
            "here. Open the folder in Echo AI and choose \"Generate digest\" "
            "to build one.",
            "",
        ]
        return "\n".join(out)

    count = stored["recordings_count"]
    line = (f"Generated from {count} recording{'' if count == 1 else 's'}, "
            f"last updated {stored['updated_at']}")
    if folder["privileged"]:
        line += " · local model only"
    out += [f"_{line}._", ""]

    if stored["new_since"] or stored["dropped_since"] or stored["stale"]:
        notes = []
        if stored["new_since"]:
            n = stored["new_since"]
            notes.append(f"{n} recording{'' if n == 1 else 's'} added since "
                         "it was written")
        if stored["dropped_since"]:
            n = stored["dropped_since"]
            notes.append(f"{n} recording{'' if n == 1 else 's'} it covers "
                         "left the folder")
        reason = "; ".join(notes) if notes else "the folder changed since " \
                                                "it was written"
        out += [f"> **This digest is out of date** ({reason}). What follows "
                "is the last version generated.", ""]

    d = stored["digest"]
    out += ["## Over time", "", d.get("overview", ""), ""]

    themes = d.get("themes") or []
    if themes:
        out += ["## Recurring themes", ""]
        for t in themes:
            head = f"- **{t.get('theme', '')}**"
            if t.get("first_appeared"):
                head += f" (first appeared {t['first_appeared']})"
            out.append(head)
            then, now = t.get("framing_then"), t.get("framing_now")
            if then or now:
                out.append(f"  - {then or ''} → {now or ''}")
            if t.get("how_it_changed"):
                out.append(f"  - {t['how_it_changed']}")
        out.append("")

    shifts = d.get("shifts") or []
    if shifts:
        out += ["## Notable shifts", ""]
        for s in shifts:
            date = f"**{s['date']}**: " if s.get("date") else ""
            note = f" ({s['note']})" if s.get("note") else ""
            out.append(f"- {date}{s.get('shift', '')}{note}")
        out.append("")

    threads = d.get("open_threads") or []
    if threads:
        out += ["## Raised and never resolved", ""]
        for t in threads:
            meta = []
            if t.get("raised_on"):
                meta.append(f"raised {t['raised_on']}")
            if t.get("last_mentioned"):
                meta.append(f"last mentioned {t['last_mentioned']}")
            tail = f" ({', '.join(meta)})" if meta else ""
            out.append(f"- {t.get('thread', '')}{tail}")
            if t.get("note"):
                out.append(f"  - {t['note']}")
        out.append("")

    commitments = d.get("commitments") or []
    if commitments:
        out += ["## Commitments across this folder", ""]
        for c in commitments:
            made = f" · {c['made_on']}" if c.get("made_on") else ""
            out.append(f"- **{c.get('owner', '')}**: "
                       f"{c.get('commitment', '')} "
                       f"[{c.get('status', 'unclear')}]{made}")
            if c.get("note"):
                out.append(f"  - {c['note']}")
        out.append("")

    quotes = d.get("quotes") or []
    if quotes:
        out += ["## In their own words", ""]
        for q in quotes:
            out.append(f"> {q.get('quote', '')}")
            who = q.get("speaker") or "Unknown"
            stamp = f" · {q['timestamp']}" if q.get("timestamp") else ""
            why = f" ({q['why']})" if q.get("why") else ""
            out += [f">", f"> — {who}{stamp}{why}", ""]

    return "\n".join(out)
