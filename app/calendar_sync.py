"""Calendar matching: pair recordings with the events they overlap.

Two providers behind one interface. The fake provider serves
deterministic events (from data/fake_calendar.json or injected in
tests) so all matching logic works without a Google account. The
Google provider is code-complete and activates the first time
google_credentials.json appears in the project root.

A match auto-titles the recording from the event and surfaces attendee
names as speaker-label suggestions. Wrong matches can be unlinked;
unlinked recordings are not re-matched.

PLAN.md asks the matcher to code defensively for moved or duplicate
events and to fall back to asking when ambiguous, which it never did.
It does now: when two events fit about equally well the candidates are
stored and the recording asks which meeting it was, and a match whose
event has since moved is re-evaluated instead of being trusted.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from app import config

FAKE_EVENTS_PATH = config.DATA_DIR / "fake_calendar.json"
GOOGLE_CREDENTIALS = config.PROJECT_ROOT / "google_credentials.json"
GOOGLE_TOKEN = config.DATA_DIR / "google_token.json"

# Voice Memos names files like "20250817 210558-0B89FB69.m4a": the
# recording's local start time, then a random suffix.
FILENAME_TS = re.compile(r"^(\d{8}) (\d{6})")


class FakeCalendarProvider:
    """Deterministic events for tests and for building without Google."""

    def __init__(self, events: list[dict] | None = None):
        if events is None:
            events = []
            if FAKE_EVENTS_PATH.exists():
                events = json.loads(FAKE_EVENTS_PATH.read_text())
        self.events = events

    def list_events(self, start: datetime, end: datetime) -> list[dict]:
        out = []
        for e in self.events:
            e_start = datetime.fromisoformat(e["start"])
            e_end = datetime.fromisoformat(e["end"])
            # Compare on wall-clock time; events may be timezone-aware.
            if (e_start.replace(tzinfo=None) < end.replace(tzinfo=None)
                    and e_end.replace(tzinfo=None) > start.replace(tzinfo=None)):
                out.append({
                    "id": e["id"],
                    "title": e["title"],
                    "start": e_start,
                    "end": e_end,
                    "attendees": e.get("attendees", []),
                })
        return out


class GoogleCalendarProvider:
    """Real Google Calendar, read-only. Activates once the user drops
    google_credentials.json into the project root and connects."""

    SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

    def __init__(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None
        if GOOGLE_TOKEN.exists():
            creds = Credentials.from_authorized_user_file(
                str(GOOGLE_TOKEN), self.SCOPES
            )
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(GOOGLE_CREDENTIALS), self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            GOOGLE_TOKEN.write_text(creds.to_json())
        self.service = build("calendar", "v3", credentials=creds)

    def list_events(self, start: datetime, end: datetime) -> list[dict]:
        result = self.service.events().list(
            calendarId="primary",
            timeMin=start.astimezone().isoformat(),
            timeMax=end.astimezone().isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        out = []
        for e in result.get("items", []):
            # All-day events have "date" instead of "dateTime"; they are
            # not meetings, skip them.
            if "dateTime" not in e.get("start", {}):
                continue
            out.append({
                "id": e["id"],
                "title": e.get("summary", "Untitled event"),
                "start": datetime.fromisoformat(e["start"]["dateTime"]),
                "end": datetime.fromisoformat(e["end"]["dateTime"]),
                "attendees": [
                    a.get("displayName") or a.get("email", "")
                    for a in e.get("attendees", [])
                    if not a.get("self")
                ],
            })
        return out


def google_connected() -> bool:
    """Authorized with Google (a usable token from a completed OAuth
    flow), not merely configured. The credentials file on its own only
    means the Connect button can start the flow; until the flow has
    run, the app must keep using the fake provider and keep showing
    the Connect button."""
    if not GOOGLE_TOKEN.exists():
        return False
    try:
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_file(
            str(GOOGLE_TOKEN), GoogleCalendarProvider.SCOPES
        )
    except Exception:
        return False
    return bool(creds and (creds.valid or creds.refresh_token))


def get_provider():
    name = config.CALENDAR_PROVIDER
    if name == "google" or (name == "auto" and google_connected()):
        return GoogleCalendarProvider()
    return FakeCalendarProvider()


def recording_window(file_path: str, created_at: str,
                     duration_seconds: float | None):
    """When the recording happened: from the Voice Memos filename when
    possible, else the processing timestamp (UTC, naive)."""
    m = FILENAME_TS.match(Path(file_path).name)
    if m:
        start = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    else:
        start = datetime.fromisoformat(created_at)
    return start, start + timedelta(seconds=duration_seconds or 0)


# Ambiguity. A runner-up whose overlap is within this much of the
# winner's is not a runner-up, it is a coin toss, and the app should ask
# rather than pick. Identical titles are treated the same way: two
# instances of one recurring meeting cannot be told apart by overlap.
TIE_RATIO = 0.8
# For "is a meeting starting now", closeness is measured in seconds, so
# the same 20% margin needs a floor: two events 10 and 20 seconds from
# now are both simply "now".
TIE_FLOOR_SECONDS = 60.0


def _titles_match(a: str, b: str) -> bool:
    return (a or "").strip().casefold() == (b or "").strip().casefold()


def _overlap(a_start, a_end, b_start, b_end) -> float:
    # Compare on naive local wall-clock time: filename timestamps are
    # local time, and Google events arrive timezone-aware.
    def naive(dt):
        return dt.replace(tzinfo=None)
    lo = max(naive(a_start), naive(b_start))
    hi = min(naive(a_end), naive(b_end))
    return max(0.0, (hi - lo).total_seconds())


def _event_row(e: dict) -> dict:
    return {
        "id": e["id"],
        "title": e["title"],
        "start": e["start"].isoformat() if hasattr(e["start"], "isoformat")
        else str(e["start"]),
        "end": e["end"].isoformat() if hasattr(e["end"], "isoformat")
        else str(e["end"]),
        "attendees": list(e.get("attendees") or []),
    }


def _store_candidates(conn, recording_id: int, events: list[dict]) -> None:
    """Park the tied events on the recording, replacing any match.

    A stored match and stored candidates are mutually exclusive: leaving
    an old match in place while asking which meeting it was would keep
    showing an answer the app has just said it does not have.
    """
    conn.execute("DELETE FROM calendar_matches WHERE recording_id = ? "
                 "AND status = 'matched'", (recording_id,))
    conn.execute(
        "INSERT INTO calendar_candidates (recording_id, candidates_json) "
        "VALUES (?, ?) ON CONFLICT (recording_id) DO UPDATE SET "
        "candidates_json = excluded.candidates_json, "
        "created_at = datetime('now')",
        (recording_id, json.dumps([_event_row(e) for e in events])),
    )
    conn.commit()


def _clear_candidates(conn, recording_id: int) -> None:
    conn.execute("DELETE FROM calendar_candidates WHERE recording_id = ?",
                 (recording_id,))


def get_candidates(conn, recording_id: int) -> list[dict]:
    """The tied events this recording is waiting to be told apart, or []."""
    row = conn.execute(
        "SELECT candidates_json FROM calendar_candidates "
        "WHERE recording_id = ?", (recording_id,),
    ).fetchone()
    return json.loads(row[0]) if row else []


def choose_event(conn, recording_id: int, event_id: str | None) -> dict | None:
    """Answer "Which meeting was this?".

    An event id picks that candidate and stores it as a normal match;
    None means "none of these", which dismisses the recording the same
    way unlinking a wrong match does, so it is not asked again.
    """
    candidates = get_candidates(conn, recording_id)
    if not candidates:
        raise LookupError("This recording is not waiting on a choice")
    if event_id is None:
        _clear_candidates(conn, recording_id)
        conn.execute(
            "INSERT INTO calendar_matches (recording_id, event_id, "
            "event_title, event_start, event_end, attendees_json, status) "
            "VALUES (?, '', '', '', '', '[]', 'dismissed') "
            "ON CONFLICT (recording_id) DO UPDATE SET status = 'dismissed'",
            (recording_id,),
        )
        conn.commit()
        return None
    chosen = next((c for c in candidates if c["id"] == event_id), None)
    if chosen is None:
        raise LookupError("That event is not one of the candidates")
    _clear_candidates(conn, recording_id)
    _write_match(conn, recording_id, chosen["id"], chosen["title"],
                 chosen["start"], chosen["end"], chosen["attendees"])
    return get_match(conn, recording_id)


def _write_match(conn, recording_id: int, event_id: str, title: str,
                 start: str, end: str, attendees: list) -> None:
    conn.execute(
        """
        INSERT INTO calendar_matches
            (recording_id, event_id, event_title, event_start, event_end,
             attendees_json, status)
        VALUES (?, ?, ?, ?, ?, ?, 'matched')
        ON CONFLICT (recording_id) DO UPDATE SET
            event_id = excluded.event_id,
            event_title = excluded.event_title,
            event_start = excluded.event_start,
            event_end = excluded.event_end,
            attendees_json = excluded.attendees_json,
            status = 'matched'
        """,
        (recording_id, event_id, title, start, end, json.dumps(attendees)),
    )
    conn.commit()


def _event_moved(provider, start, end, event_id: str,
                 stored_start: str, stored_end: str) -> bool:
    """Whether the calendar no longer agrees with a stored match.

    True when the event has different times than the ones recorded, or
    has left the window around the recording entirely (moved far, or
    deleted). False when the calendar cannot be read at all, because a
    calendar hiccup must never silently unpick a good match.
    """
    try:
        events = provider.list_events(start - timedelta(minutes=15),
                                      end + timedelta(minutes=15))
    except Exception:
        return False
    for e in events:
        if e["id"] != event_id:
            continue
        return not (str(e["start"]) == stored_start
                    or e["start"].isoformat() == stored_start) \
            or not (str(e["end"]) == stored_end
                    or e["end"].isoformat() == stored_end)
    return True


def match_recording(conn, recording_id: int, provider=None) -> dict | None:
    """Match one recording to its best-overlapping event and store it.
    Returns the stored match, or None. Dismissed recordings stay
    unmatched until re-linked on purpose."""
    row = conn.execute(
        "SELECT file_path, created_at, duration_seconds FROM recordings "
        "WHERE id = ?",
        (recording_id,),
    ).fetchone()
    if row is None:
        return None
    existing = conn.execute(
        "SELECT status, event_id, event_start, event_end "
        "FROM calendar_matches WHERE recording_id = ?",
        (recording_id,),
    ).fetchone()
    if existing is not None and existing[0] == "dismissed":
        return None

    provider = provider or get_provider()
    start, end = recording_window(*row)
    if end <= start:
        return None

    if existing is not None:
        # A stored match is trusted only while the calendar still agrees
        # with it. An event that was moved (or deleted) after the match
        # was written is re-evaluated from scratch, which is the whole
        # point of storing its start and end rather than just its id.
        if not _event_moved(provider, start, end, existing[1], existing[2],
                            existing[3]):
            return get_match(conn, recording_id)
    events = provider.list_events(start - timedelta(minutes=15),
                                  end + timedelta(minutes=15))
    scored = sorted(
        ((_overlap(start, end, e["start"], e["end"]), e) for e in events),
        key=lambda pair: pair[0], reverse=True,
    )
    scored = [(ov, e) for ov, e in scored if ov > 0]
    if not scored:
        return None

    best_overlap, best = scored[0]
    tied = [e for ov, e in scored[1:]
            if ov >= TIE_RATIO * best_overlap
            or _titles_match(e["title"], best["title"])]
    if tied:
        # Do not guess. Store the candidates and let the recording ask.
        _store_candidates(conn, recording_id, [best] + tied)
        return None
    _clear_candidates(conn, recording_id)

    _write_match(conn, recording_id, best["id"], best["title"],
                 best["start"].isoformat(), best["end"].isoformat(),
                 best["attendees"])
    return get_match(conn, recording_id)


def match_all(conn, provider=None) -> int:
    """Match every done recording that has no match record yet, and
    re-check the ones that do in case their event moved. Dismissed
    recordings are left alone: that was a deliberate answer."""
    provider = provider or get_provider()
    rec_ids = [r[0] for r in conn.execute(
        "SELECT id FROM recordings WHERE status = 'done' AND id NOT IN "
        "(SELECT recording_id FROM calendar_matches "
        " WHERE status = 'dismissed')"
    ).fetchall()]
    changed = 0
    for rid in rec_ids:
        before = conn.execute(
            "SELECT event_id FROM calendar_matches WHERE recording_id = ? "
            "AND status = 'matched'", (rid,),
        ).fetchone()
        match = match_recording(conn, rid, provider=provider)
        after = match["event_id"] if match else None
        # Only a new or genuinely different match counts. Re-checking an
        # unchanged one is work the caller did not ask to hear about.
        if after != (before[0] if before else None):
            changed += 1
    return changed


STARTING_BEFORE = timedelta(minutes=5)
STARTING_AFTER = timedelta(minutes=5)


def starting_event(provider=None, now=None,
                   before=STARTING_BEFORE, after=STARTING_AFTER) -> dict | None:
    """The calendar event that is starting right about now, if any.

    An event qualifies when it has attendees, has not ended, and its
    start is within [now - before, now + after]. Used to offer a
    "record this meeting?" prompt. Returns the nearest such event or
    None. Never triggers any recording itself.

    The same tie rule as matching applies here: when a second event is
    starting about as close to now as the first, or carries the same
    title, the prompt says so and names both rather than picking one.
    The extra events come back under "others"; the shape of the answer
    is otherwise unchanged.
    """
    provider = provider or get_provider()
    now = now or datetime.now()
    events = provider.list_events(now - before, now + after)
    candidates = []
    for e in events:
        start = e["start"].replace(tzinfo=None) if hasattr(
            e["start"], "replace") else datetime.fromisoformat(e["start"])
        end = e["end"].replace(tzinfo=None) if hasattr(
            e["end"], "replace") else datetime.fromisoformat(e["end"])
        if not e.get("attendees") or end <= now:
            continue
        if now - before <= start <= now + after:
            candidates.append((abs((start - now).total_seconds()), e, start))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    distance, event, start = candidates[0]

    def shape(e, s):
        return {"id": e["id"], "title": e["title"], "start": s.isoformat(),
                "attendees": e["attendees"]}

    margin = max(TIE_FLOOR_SECONDS, (1 - TIE_RATIO) * distance)
    others = [shape(e, s) for d, e, s in candidates[1:]
              if d - distance <= margin or _titles_match(e["title"],
                                                         event["title"])]
    out = shape(event, start)
    if others:
        out["ambiguous"] = True
        out["others"] = others
    return out


def get_match(conn, recording_id: int) -> dict | None:
    row = conn.execute(
        "SELECT event_id, event_title, event_start, event_end, "
        "attendees_json, status FROM calendar_matches WHERE recording_id = ?",
        (recording_id,),
    ).fetchone()
    if row is None or row[5] != "matched":
        return None
    return {
        "event_id": row[0],
        "title": row[1],
        "start": row[2],
        "end": row[3],
        "attendees": json.loads(row[4]),
    }


def unlink(conn, recording_id: int) -> bool:
    """Dismiss a wrong match; the recording will not be re-matched."""
    cur = conn.execute(
        "UPDATE calendar_matches SET status = 'dismissed' "
        "WHERE recording_id = ?",
        (recording_id,),
    )
    conn.commit()
    return cur.rowcount == 1
