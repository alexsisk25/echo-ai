"""FastAPI worker: JSON API for recordings, search, and summaries, plus
the static web UI. Run with: uvicorn app.main:app --reload"""

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal

from pydantic import BaseModel
from starlette.background import BackgroundTask

from app import (calendar_sync, commitments, config, db, digest, llm,
                 manage, merged, notes, organize, people, provenance,
                 recorder, reprocess, semantic, summarize,
                 sync as sync_memos, translate, voices)

log = logging.getLogger("otter.main")


def warm_embedding_model() -> None:
    """Load the embedding model in the background at startup.

    Ask imports sentence-transformers the first time it runs, which
    pulls in torch and transformers: thousands of small file reads. On a
    machine where each of those reads is intercepted by a security
    scanner this takes many minutes, and whoever asks the first question
    is the one who waits, with the UI showing nothing but a spinner.
    Starting it here means the wait happens while the app comes up
    instead. Failure is not fatal: the request path still loads the
    model itself if this never finishes.
    """
    if os.environ.get("OTTER_NO_WARMUP"):
        return

    def warm():
        try:
            semantic._get_model()
            log.info("Embedding model ready")
        except Exception:
            log.exception("Embedding model warm-up failed; the first "
                          "search or question will load it instead")

    threading.Thread(target=warm, name="warm-embeddings",
                     daemon=True).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warm_embedding_model()
    yield


app = FastAPI(title="Echo AI", lifespan=lifespan)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/recordings")
def list_recordings():
    conn = db.connect()
    try:
        speaker_rows = conn.execute(
            """
            SELECT t.recording_id, s.speaker FROM segments s
            JOIN transcripts t ON t.id = s.transcript_id
            WHERE s.speaker IS NOT NULL
            GROUP BY t.recording_id, s.speaker
            """
        ).fetchall()
        speakers_by_rec = {}
        for rec_id, speaker in speaker_rows:
            speakers_by_rec.setdefault(rec_id, []).append(speaker)

        event_titles = dict(conn.execute(
            "SELECT recording_id, event_title FROM calendar_matches "
            "WHERE status = 'matched'"
        ).fetchall())

        rows = conn.execute(
            """
            SELECT r.id, r.file_path, r.duration_seconds, r.created_at,
                   r.status, t.id, s.summary_json, r.recorded_at,
                   r.title, r.folder_id, f.name, r.suggested_folder
            FROM recordings r
            LEFT JOIN transcripts t ON t.recording_id = r.id
            LEFT JOIN summaries s ON s.transcript_id = t.id
            LEFT JOIN folders f ON f.id = r.folder_id
            ORDER BY r.recorded_at DESC
            """
        ).fetchall()
        result = []
        for r in rows:
            summary = json.loads(r[6]) if r[6] else None
            result.append({
                "id": r[0],
                "file_name": Path(r[1]).name,
                "duration_seconds": r[2],
                "created_at": r[3],
                "status": r[4],
                "transcript_id": r[5],
                # Manual titles are pinned: they win over the calendar
                # event, which wins over the AI title.
                "title": r[8] or event_titles.get(r[0])
                         or (summary["title"] if summary else None),
                "title_is_manual": r[8] is not None,
                "date": (summary or {}).get("date"),
                "topics": (summary or {}).get("topics", []),
                "speakers": sorted(speakers_by_rec.get(r[0], [])),
                "recorded_at": r[7] or r[3],
                "folder_id": r[9],
                "folder": r[10],
                "suggested_folder": r[11],
            })
        return result
    finally:
        conn.close()


@app.get("/api/recordings/{recording_id}")
def get_recording(recording_id: int):
    conn = db.connect()
    try:
        row = conn.execute(
            """
            SELECT r.id, r.file_path, r.duration_seconds, r.created_at,
                   r.status, t.id, t.full_text, t.language, t.model,
                   s.summary_json, r.recorded_at
            FROM recordings r
            LEFT JOIN transcripts t ON t.recording_id = r.id
            LEFT JOIN summaries s ON s.transcript_id = t.id
            WHERE r.id = ?
            """,
            (recording_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Recording not found")
        segments = []
        if row[5] is not None:
            segments = [
                {"id": s[0], "start": s[1], "end": s[2], "text": s[3],
                 "speaker": s[4], "auto": s[5] is not None,
                 "auto_original": s[5], "pinned": bool(s[6])}
                for s in conn.execute(
                    "SELECT id, start_seconds, end_seconds, text, speaker, "
                    "auto_original, pinned "
                    "FROM segments WHERE transcript_id = ? ORDER BY start_seconds",
                    (row[5],),
                ).fetchall()
            ]
        translations = []
        if row[5] is not None:
            translations = [r[0] for r in conn.execute(
                "SELECT lang FROM translations WHERE transcript_id = ?",
                (row[5],),
            ).fetchall()]
        return {
            "id": row[0],
            "file_name": Path(row[1]).name,
            "duration_seconds": row[2],
            "created_at": row[3],
            "recorded_at": row[10] or row[3],
            "status": row[4],
            "transcript_id": row[5],
            "full_text": row[6],
            "language": row[7],
            "model": row[8],
            "summary": json.loads(row[9]) if row[9] else None,
            "segments": segments,
            "privileged": db.is_privileged(conn, row[0]),
            "translations": translations,
            "calendar_event": calendar_sync.get_match(conn, row[0]),
            # Two events fit this recording about equally well, so the
            # app refused to guess and asks instead.
            "calendar_candidates": calendar_sync.get_candidates(
                conn, row[0]),
            "notes": notes.get(conn, row[0]),
            "reset_status": RESET_STATUS.get(row[0]),
            # A long recording that came out as one speaker: the UI
            # offers a reprocess with a speaker-count hint.
            "diarization_suspect": bool(conn.execute(
                "SELECT diarization_suspect FROM recordings WHERE id = ?",
                (row[0],),
            ).fetchone()[0]),
            # Two clusters under one name, still worth mentioning.
            "merged_name_suspect": conn.execute(
                "SELECT CASE WHEN merged_name_dismissed = 1 THEN NULL "
                "ELSE merged_name_suspect END FROM recordings WHERE id = ?",
                (row[0],),
            ).fetchone()[0],
            **_organization(conn, row[0]),
        }
    finally:
        conn.close()


@app.delete("/api/recordings/{recording_id}")
def delete_recording(recording_id: int):
    conn = db.connect()
    try:
        result = manage.delete_recording(conn, recording_id)
    finally:
        conn.close()
    if result is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    return result


class BulkIds(BaseModel):
    ids: list[int]


@app.post("/api/recordings/bulk-delete")
def bulk_delete(body: BulkIds):
    conn = db.connect()
    try:
        deleted = sum(
            1 for rid in body.ids
            if manage.delete_recording(conn, rid, how="bulk") is not None
        )
    finally:
        conn.close()
    return {"deleted": deleted}


@app.get("/api/export")
def export_recordings(ids: str):
    try:
        recording_ids = [int(part) for part in ids.split(",") if part.strip()]
    except ValueError:
        raise HTTPException(status_code=422, detail="ids must be integers")
    if not recording_ids:
        raise HTTPException(status_code=422, detail="No ids given")
    conn = db.connect()
    try:
        result = manage.export_zip(conn, recording_ids)
    finally:
        conn.close()
    if result is None:
        raise HTTPException(status_code=404, detail="No such recordings")
    zip_path, name = result
    return FileResponse(
        zip_path, media_type="application/zip", filename=name,
        background=BackgroundTask(os.unlink, zip_path),
    )


def _organization(conn, recording_id: int) -> dict:
    row = conn.execute(
        """
        SELECT r.title, r.folder_id, f.name, r.suggested_folder
        FROM recordings r LEFT JOIN folders f ON f.id = r.folder_id
        WHERE r.id = ?
        """,
        (recording_id,),
    ).fetchone()
    return {
        "manual_title": row[0],
        "folder_id": row[1],
        "folder": row[2],
        "suggested_folder": row[3],
    }


@app.get("/api/search")
def search(q: str, folder_id: int | None = None):
    """Keyword search across transcripts, the user's own notes, and
    titles. folder_id narrows every one of them to a single folder, so
    searching while the list is filtered searches what is on screen."""
    conn = db.connect()
    try:
        if folder_id is not None:
            exists = conn.execute(
                "SELECT 1 FROM folders WHERE id = ?", (folder_id,)
            ).fetchone()
            if exists is None:
                raise HTTPException(status_code=404,
                                    detail="No such folder")
        return (db.search_transcripts(conn, q, folder_id=folder_id)
                + db.search_notes(conn, q, folder_id=folder_id)
                + organize.search_titles(conn, q, folder_id=folder_id))
    finally:
        conn.close()


class TitleBody(BaseModel):
    title: str


@app.put("/api/recordings/{recording_id}/title")
def set_title(recording_id: int, body: TitleBody):
    """Set (or clear, with an empty string) the manual title. Manual
    titles are pinned: nothing automatic ever overwrites them."""
    conn = db.connect()
    try:
        cur = conn.execute(
            "UPDATE recordings SET title = ? WHERE id = ?",
            (body.title.strip() or None, recording_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Recording not found")
        return {"id": recording_id, "title": body.title.strip() or None}
    finally:
        conn.close()


class FolderBody(BaseModel):
    name: str


@app.get("/api/folders")
def get_folders():
    conn = db.connect()
    try:
        return organize.list_folders(conn)
    finally:
        conn.close()


@app.post("/api/folders")
def post_folder(body: FolderBody):
    conn = db.connect()
    try:
        try:
            return organize.create_folder(conn, body.name)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err))
    finally:
        conn.close()


@app.put("/api/folders/{folder_id}")
def put_folder(folder_id: int, body: FolderBody):
    conn = db.connect()
    try:
        try:
            ok = organize.rename_folder(conn, folder_id, body.name)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err))
        except FileExistsError as err:
            raise HTTPException(status_code=409, detail=str(err))
        if not ok:
            raise HTTPException(status_code=404, detail="Folder not found")
        return {"id": folder_id, "name": body.name.strip()}
    finally:
        conn.close()


@app.delete("/api/folders/{folder_id}")
def remove_folder(folder_id: int):
    conn = db.connect()
    try:
        unfiled = organize.delete_folder(conn, folder_id)
        if unfiled is None:
            raise HTTPException(status_code=404, detail="Folder not found")
        return {"deleted": folder_id, "unfiled": unfiled}
    finally:
        conn.close()


class AssignBody(BaseModel):
    folder_id: int | None = None


@app.put("/api/recordings/{recording_id}/folder")
def put_recording_folder(recording_id: int, body: AssignBody):
    conn = db.connect()
    try:
        try:
            changed = organize.assign_folder(conn, [recording_id],
                                             body.folder_id)
        except LookupError as err:
            raise HTTPException(status_code=404, detail=str(err))
        if changed == 0:
            raise HTTPException(status_code=404, detail="Recording not found")
        return {"id": recording_id, "folder_id": body.folder_id}
    finally:
        conn.close()


class BulkAssignBody(BaseModel):
    ids: list[int]
    folder_id: int | None = None


@app.post("/api/recordings/bulk-folder")
def bulk_folder(body: BulkAssignBody):
    conn = db.connect()
    try:
        try:
            changed = organize.assign_folder(conn, body.ids, body.folder_id)
        except LookupError as err:
            raise HTTPException(status_code=404, detail=str(err))
        return {"filed": changed}
    finally:
        conn.close()


class SuggestBody(BaseModel):
    ids: list[int] | None = None


@app.post("/api/folders/suggest")
def suggest_folders(body: SuggestBody):
    conn = db.connect()
    try:
        try:
            suggestions = organize.suggest_for_unfiled(conn, body.ids)
        except llm.LocalModelUnavailable as err:
            raise HTTPException(status_code=503, detail=str(err))
        return {"suggestions": suggestions}
    finally:
        conn.close()


@app.post("/api/recordings/{recording_id}/suggestion/accept")
def accept_folder_suggestion(recording_id: int):
    conn = db.connect()
    try:
        result = organize.accept_suggestion(conn, recording_id)
        if result is None:
            raise HTTPException(status_code=404,
                                detail="No pending suggestion")
        return result
    finally:
        conn.close()


@app.post("/api/recordings/{recording_id}/suggestion/dismiss")
def dismiss_folder_suggestion(recording_id: int):
    conn = db.connect()
    try:
        if not organize.dismiss_suggestion(conn, recording_id):
            raise HTTPException(status_code=404, detail="Recording not found")
        return {"id": recording_id, "dismissed": True}
    finally:
        conn.close()


@app.post("/api/folders/accept-all")
def accept_all_folder_suggestions():
    conn = db.connect()
    try:
        return {"filed": organize.accept_all_suggestions(conn)}
    finally:
        conn.close()


class DigestBody(BaseModel):
    # Rebuild from every recording instead of folding in what is new.
    force: bool = False


def _digest_state(conn, folder_id: int) -> dict:
    """What the digest view needs: the folder, the stored digest (or
    None), and how much of the folder it currently covers."""
    folder = digest.folder_info(conn, folder_id)
    stored = digest.get(conn, folder_id)
    recordings = digest.folder_recordings(conn, folder_id)
    return {
        "folder": folder,
        "privileged": digest.folder_is_privileged(conn, folder_id),
        "recordings_in_folder": len(recordings),
        "digest": stored,
    }


@app.get("/api/folders/{folder_id}/digest")
def get_folder_digest(folder_id: int):
    conn = db.connect()
    try:
        return _digest_state(conn, folder_id)
    except LookupError as err:
        raise HTTPException(status_code=404, detail=str(err))
    finally:
        conn.close()


@app.post("/api/folders/{folder_id}/digest")
def build_folder_digest(folder_id: int, body: DigestBody | None = None):
    """Generate the digest, or fold in the recordings added since the
    last one. Runs inline like the other LLM endpoints."""
    conn = db.connect()
    try:
        result = digest.generate(conn, folder_id,
                                 force=bool(body and body.force))
        return {**_digest_state(conn, folder_id),
                "cached": result["cached"],
                "incremental": result["incremental"]}
    except LookupError as err:
        raise HTTPException(status_code=404, detail=str(err))
    except ValueError as err:
        raise HTTPException(status_code=409, detail=str(err))
    except llm.LocalModelUnavailable as err:
        raise HTTPException(status_code=503, detail=str(err))
    finally:
        conn.close()


class FolderPrivilegedBody(BaseModel):
    privileged: bool


@app.put("/api/folders/{folder_id}/privileged")
def set_folder_privileged(folder_id: int, body: FolderPrivilegedBody):
    conn = db.connect()
    try:
        return organize.set_folder_privileged(conn, folder_id,
                                              body.privileged)
    except LookupError as err:
        raise HTTPException(status_code=404, detail=str(err))
    finally:
        conn.close()


class NotesBody(BaseModel):
    text: str


@app.put("/api/recordings/{recording_id}/notes")
def save_notes(recording_id: int, body: NotesBody):
    conn = db.connect()
    try:
        exists = conn.execute(
            "SELECT 1 FROM recordings WHERE id = ?", (recording_id,)
        ).fetchone()
        if exists is None:
            raise HTTPException(status_code=404, detail="Recording not found")
        notes.save(conn, recording_id, body.text)
        return {"id": recording_id, "saved": True}
    finally:
        conn.close()


@app.post("/api/recordings/{recording_id}/notes/enhance")
def enhance_notes(recording_id: int):
    conn = db.connect()
    try:
        try:
            return notes.enhance(conn, recording_id)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err))
        except LookupError as err:
            raise HTTPException(status_code=404, detail=str(err))
        except llm.LocalModelUnavailable as err:
            raise HTTPException(status_code=503, detail=str(err))
    finally:
        conn.close()


@app.post("/api/recordings/{recording_id}/summarize")
def create_summary(recording_id: int):
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT id, full_text FROM transcripts WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="No transcript found")
        transcript_id, full_text = row
        privileged = db.is_privileged(conn, recording_id)
        try:
            summary = summarize.summarize(full_text, privileged=privileged)
        except llm.LocalModelUnavailable as err:
            raise HTTPException(status_code=503, detail=str(err))
        db.upsert_summary(
            conn, transcript_id, summary.model_dump_json(),
            model="local" if privileged else config.LLM_MODEL,
        )
        commitments.extract(conn, transcript_id)
        return summary.model_dump()
    finally:
        conn.close()


@app.get("/api/commitments")
def get_commitments(status: str | None = None, owner: str | None = None,
                    due: str | None = None, mine: bool | None = None):
    """Commitments, most urgent first. due is "overdue" or "due-soon";
    mine narrows to (or away from) the user's own, by OTTER_USER_NAME."""
    if due is not None and due not in ("overdue", "due-soon"):
        raise HTTPException(
            status_code=422,
            detail="due must be \"overdue\" or \"due-soon\"")
    conn = db.connect()
    try:
        return commitments.list_commitments(conn, status=status, owner=owner,
                                            due=due, mine=mine)
    finally:
        conn.close()


class CommitmentStatus(BaseModel):
    status: str


@app.post("/api/commitments/{commitment_id}/status")
def set_commitment_status(commitment_id: int, body: CommitmentStatus):
    if body.status not in ("open", "done"):
        raise HTTPException(status_code=422, detail="status must be open or done")
    conn = db.connect()
    try:
        if not commitments.set_status(conn, commitment_id, body.status):
            raise HTTPException(status_code=404, detail="Commitment not found")
        return {"id": commitment_id, "status": body.status}
    finally:
        conn.close()


@app.get("/api/people")
def list_people():
    conn = db.connect()
    try:
        return people.list_people(conn)
    finally:
        conn.close()


@app.get("/api/people/colors")
def people_colors():
    conn = db.connect()
    try:
        out = {}
        for name, color in conn.execute(
                "SELECT name, color FROM person_colors").fetchall():
            try:
                out[name] = int(color)
            except (TypeError, ValueError):
                # An unmigratable value behaves as Auto rather than
                # breaking every chip in the app.
                continue
        return out
    finally:
        conn.close()


class PersonColorBody(BaseModel):
    # A hue number (0-360, OKLCH hue angle) or None/'auto'. Raw color
    # values never cross the API; lightness and saturation live in
    # theme tokens on the client.
    color: float | Literal["auto"] | None = None


@app.put("/api/people/{name}/color")
def set_person_color(name: str, body: PersonColorBody):
    """Pick a chip hue for a person, or None/'auto' to go back to the
    automatic per-name hue."""
    conn = db.connect()
    try:
        if body.color is None or body.color == "auto":
            conn.execute("DELETE FROM person_colors WHERE name = ?", (name,))
            conn.commit()
            return {"name": name, "color": None}
        if not (0 <= body.color < 360):
            raise HTTPException(
                status_code=422,
                detail="Hue must be a number from 0 to 359")
        hue = int(round(body.color)) % 360
        conn.execute(
            "INSERT INTO person_colors (name, color) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET color = excluded.color",
            (name, hue))
        conn.commit()
        return {"name": name, "color": hue}
    finally:
        conn.close()


@app.get("/api/people/{name}")
def get_person(name: str):
    conn = db.connect()
    try:
        data = people.person_data(conn, name)
        if data is None:
            raise HTTPException(status_code=404, detail="Person not found")
        data["dossier"] = people.get_dossier(conn, name)
        return data
    finally:
        conn.close()


@app.post("/api/people/{name}/dossier")
def build_dossier(name: str):
    conn = db.connect()
    try:
        try:
            return people.build_dossier(conn, name).model_dump()
        except ValueError:
            raise HTTPException(status_code=404, detail="Person not found")
        except llm.LocalModelUnavailable as err:
            raise HTTPException(status_code=503, detail=str(err))
    finally:
        conn.close()


@app.get("/api/ask")
def ask(q: str, folder_id: int | None = None):
    conn = db.connect()
    try:
        return semantic.ask(conn, q, folder_id=folder_id)
    except llm.LocalModelUnavailable as err:
        raise HTTPException(status_code=503, detail=str(err))
    finally:
        conn.close()


@app.get("/api/recordings/{recording_id}/audio")
def get_audio(recording_id: int):
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT file_path FROM recordings WHERE id = ?", (recording_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None or not Path(row[0]).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(row[0], media_type="audio/mp4")


@app.get("/api/calendar/status")
def calendar_status():
    return {
        "provider": ("google" if config.CALENDAR_PROVIDER == "google"
                     or (config.CALENDAR_PROVIDER == "auto"
                         and calendar_sync.google_connected())
                     else "fake"),
        "google_connected": calendar_sync.google_connected(),
    }


@app.post("/api/calendar/connect")
def calendar_connect():
    """One-time Google connect: runs the OAuth flow in a local browser.
    Until google_credentials.json exists this explains what to do."""
    if not calendar_sync.GOOGLE_CREDENTIALS.exists():
        raise HTTPException(
            status_code=503,
            detail="Put google_credentials.json (an OAuth desktop client "
                   "from Google Cloud Console) in the project root, then "
                   "click Connect again.",
        )
    try:
        calendar_sync.GoogleCalendarProvider()
    except Exception as err:
        raise HTTPException(status_code=502,
                            detail=f"Google connection failed: {err}")
    return {"connected": True}


@app.get("/api/calendar/starting")
def calendar_starting():
    """The meeting starting right now (with attendees), for the record
    prompt. Never records; just reports. Returns {event: {...}|null}."""
    try:
        return {"event": calendar_sync.starting_event()}
    except Exception:
        # A calendar hiccup must never break the poll; just no prompt.
        return {"event": None}


@app.post("/api/calendar/match")
def calendar_match_all():
    conn = db.connect()
    try:
        matched = calendar_sync.match_all(conn)
        return {"matched": matched}
    finally:
        conn.close()


@app.post("/api/recordings/{recording_id}/calendar/unlink")
def calendar_unlink(recording_id: int):
    conn = db.connect()
    try:
        if not calendar_sync.unlink(conn, recording_id):
            raise HTTPException(status_code=404, detail="No match to unlink")
        return {"recording_id": recording_id, "calendar_event": None}
    finally:
        conn.close()


class CalendarChoice(BaseModel):
    # None means "none of these": the recording is dismissed rather than
    # matched, and is not asked about again.
    event_id: str | None = None


@app.post("/api/recordings/{recording_id}/calendar/choose")
def calendar_choose(recording_id: int, body: CalendarChoice):
    """Answer "Which meeting was this?" for an ambiguous recording."""
    conn = db.connect()
    try:
        try:
            match = calendar_sync.choose_event(conn, recording_id,
                                               body.event_id)
        except LookupError as err:
            raise HTTPException(status_code=404, detail=str(err))
        return {"recording_id": recording_id, "calendar_event": match,
                "calendar_candidates": []}
    finally:
        conn.close()


class TranslateRequest(BaseModel):
    target: str


@app.post("/api/recordings/{recording_id}/translate")
def translate_recording(recording_id: int, body: TranslateRequest):
    if body.target not in translate.LANG_NAMES:
        raise HTTPException(status_code=422, detail="target must be en or es")
    conn = db.connect()
    try:
        privileged = db.is_privileged(conn, recording_id)
        try:
            return translate.translate_recording(
                conn, recording_id, body.target, privileged=privileged
            )
        except LookupError:
            raise HTTPException(status_code=404, detail="No transcript found")
        except llm.LocalModelUnavailable as err:
            raise HTTPException(status_code=503, detail=str(err))
    finally:
        conn.close()


class PrivilegedFlag(BaseModel):
    privileged: bool


@app.post("/api/recordings/{recording_id}/privileged")
def set_privileged(recording_id: int, body: PrivilegedFlag):
    conn = db.connect()
    try:
        cur = conn.execute(
            "UPDATE recordings SET privileged = ? WHERE id = ?",
            (1 if body.privileged else 0, recording_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Recording not found")
        return {"id": recording_id, "privileged": body.privileged}
    finally:
        conn.close()


class RecordStartBody(BaseModel):
    mode: Literal["online", "in_person"] = "online"
    # How many people are in the room, for an in-person capture.
    num_speakers: int | None = None


@app.post("/api/record/start")
def record_start(body: RecordStartBody | None = None):
    before = recorder.status()
    if before["state"] == "error":
        recorder.reset_error()
    if body and body.num_speakers is not None \
            and not (1 <= body.num_speakers <= 20):
        raise HTTPException(
            status_code=422,
            detail="Speaker count must be between 1 and 20")
    result = recorder.start(
        mode=body.mode if body else "online",
        num_speakers=body.num_speakers if body else None)
    if result["state"] == "error":
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@app.post("/api/record/stop")
def record_stop():
    try:
        result = recorder.stop()
    except RuntimeError as err:
        raise HTTPException(status_code=409, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    name = result.get("file")
    if name:
        conn = db.connect()
        try:
            provenance.log_captured(conn, Path(name).stem, name)
        finally:
            conn.close()
    return result


@app.get("/api/record/status")
def record_status():
    return recorder.status()


@app.get("/api/record/levels")
def record_levels():
    return recorder.levels()


@app.post("/api/sync")
def sync_from_iphone():
    log = logging.getLogger("otter.api")
    conn = db.connect()
    try:
        copied = sync_memos.sync_voice_memos(conn=conn)
    except PermissionError as err:
        # Never reported as "nothing new": an unreadable folder is a
        # setup problem, and looks identical to an empty one otherwise.
        log.warning("Sync could not read the Voice Memos folder: %s", err)
        raise HTTPException(
            status_code=503,
            detail="Cannot read the Voice Memos folder. Grant Full Disk "
                   "Access to the app running the server, then retry.",
        )
    finally:
        conn.close()
    log.info("Sync read the Voice Memos folder: %d new recording(s)", copied)
    return {"copied": copied}


@app.get("/api/activity")
def activity(type: str | None = None):
    conn = db.connect()
    try:
        provenance.seed(conn)  # idempotent; populates the past once
        return provenance.list_events(conn, event_type=type)
    finally:
        conn.close()


class SpeakerRename(BaseModel):
    old_label: str
    new_label: str
    force: bool = False
    # Pinned lines are human decisions, so a group rename leaves them
    # alone unless the user explicitly asks to include them.
    include_pinned: bool = False


@app.post("/api/transcripts/{transcript_id}/speakers")
def rename_speaker(transcript_id: int, body: SpeakerRename):
    new_label = body.new_label.strip()
    if not new_label:
        raise HTTPException(status_code=422, detail="New label is empty")
    conn = db.connect()
    try:
        row = conn.execute(
            """
            SELECT r.file_path FROM transcripts t
            JOIN recordings r ON r.id = t.recording_id WHERE t.id = ?
            """,
            (transcript_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        # Merge guard: renaming onto a name that already exists in this
        # recording folds two speakers into one and cannot be unpicked
        # line by line afterwards. Require explicit confirmation.
        if (new_label != body.old_label
                and db.speaker_exists(conn, transcript_id, new_label)
                and not body.force):
            raise HTTPException(
                status_code=409,
                detail=f'"{new_label}" already exists in this recording. '
                       f'Renaming "{body.old_label}" merges every one of '
                       f'its lines into "{new_label}", which cannot be '
                       'un-merged. If you meant to fix one misattributed '
                       "line, use that line's reassign button instead.",
            )

        enrolled = 0
        # Renaming a diarized label to a real name is the enrollment
        # moment: fingerprint the voice before the label changes.
        # Best-effort; the rename itself must never fail on this.
        if body.old_label.startswith("SPEAKER_") and not \
                new_label.startswith("SPEAKER_"):
            try:
                enrolled = voices.enroll_from_recording(
                    conn, transcript_id, row[0], body.old_label, new_label
                )
            except Exception:
                logging.getLogger("otter.api").exception(
                    "Voice enrollment failed for transcript %d",
                    transcript_id,
                )

        # Counted before the update, so a no-op rename can explain
        # itself: every remaining line of this speaker is pinned.
        pinned_skipped = 0 if body.include_pinned else \
            db.count_pinned_for_speaker(conn, transcript_id, body.old_label)
        changed = db.rename_speaker(
            conn, transcript_id, body.old_label, new_label,
            include_pinned=body.include_pinned,
        )
        # A fresh enrollment sweeps the whole archive in the background
        # so this voice gets named everywhere it already appears.
        if enrolled:
            start_retag_sweep(new_label)
        return {"changed": changed, "enrolled_samples": enrolled,
                "pinned_skipped": pinned_skipped}
    finally:
        conn.close()


def start_retag_sweep(name: str) -> None:
    """Retag the archive on a background thread. db_path and key are
    resolved now because the thread outlives the request."""
    threading.Thread(
        target=voices.retag_archive,
        args=(name, config.DB_PATH, db.get_or_create_key()),
        daemon=True,
    ).start()


class SegmentSpeaker(BaseModel):
    name: str


@app.post("/api/transcripts/{transcript_id}/segments/{segment_id}/speaker")
def reassign_segment_speaker(transcript_id: int, segment_id: int,
                             body: SegmentSpeaker):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Name is empty")
    conn = db.connect()
    try:
        if not db.reassign_segment(conn, transcript_id, segment_id, name):
            raise HTTPException(status_code=404, detail="Segment not found")
        return {"segment_id": segment_id, "speaker": name, "pinned": True}
    finally:
        conn.close()


class SegmentsAssign(BaseModel):
    ids: list[int]
    name: str


@app.post("/api/transcripts/{transcript_id}/segments/assign")
def assign_segments_speaker(transcript_id: int, body: SegmentsAssign):
    """Assign several lines to one speaker: the manual cluster-split."""
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Name is empty")
    conn = db.connect()
    try:
        changed = db.assign_segments(conn, transcript_id, body.ids, name)
        return {"assigned": changed, "speaker": name}
    finally:
        conn.close()


@app.post("/api/transcripts/{transcript_id}/segments/{segment_id}/confirm")
def confirm_segment_line(transcript_id: int, segment_id: int):
    conn = db.connect()
    try:
        if not db.confirm_segment(conn, transcript_id, segment_id):
            raise HTTPException(status_code=404,
                                detail="No auto-tag on this line")
        return {"segment_id": segment_id, "confirmed": True}
    finally:
        conn.close()


@app.post("/api/transcripts/{transcript_id}/segments/{segment_id}/revert")
def revert_segment_line(transcript_id: int, segment_id: int):
    conn = db.connect()
    try:
        if not db.revert_segment(conn, transcript_id, segment_id):
            raise HTTPException(status_code=404,
                                detail="No auto-tag on this line")
        return {"segment_id": segment_id, "reverted": True}
    finally:
        conn.close()


class SegmentSnapshot(BaseModel):
    id: int
    speaker: str | None = None
    auto_original: str | None = None
    pinned: bool = False


class RestoreBody(BaseModel):
    segments: list[SegmentSnapshot]


@app.get("/api/transcripts/{transcript_id}/segments/states")
def get_segment_states(transcript_id: int, ids: str):
    try:
        seg_ids = [int(p) for p in ids.split(",") if p.strip()]
    except ValueError:
        raise HTTPException(status_code=422, detail="ids must be integers")
    conn = db.connect()
    try:
        return db.segment_states(conn, transcript_id, seg_ids)
    finally:
        conn.close()


@app.post("/api/transcripts/{transcript_id}/segments/restore")
def restore_segment_states(transcript_id: int, body: RestoreBody):
    """Undo primitive: put lines back to an exact prior snapshot."""
    conn = db.connect()
    try:
        changed = db.restore_segments(
            conn, transcript_id, [s.model_dump() for s in body.segments])
        return {"restored": changed}
    finally:
        conn.close()


# Reset runs diarization again, which takes minutes; it happens on a
# background thread and the UI polls the recording until done.
RESET_STATUS: dict[int, str] = {}


def start_speaker_reset(recording_id: int) -> None:
    db_path, key = config.DB_PATH, db.get_or_create_key()

    def work():
        conn = db.connect(db_path=db_path, key=key)
        try:
            result = voices.reset_speakers(conn, recording_id)
            RESET_STATUS[recording_id] = (
                "done" if result is not None else "failed"
            )
        except Exception:
            logging.getLogger("otter.api").exception(
                "Speaker reset failed for recording %d", recording_id
            )
            RESET_STATUS[recording_id] = "failed"
        finally:
            conn.close()

    RESET_STATUS[recording_id] = "running"
    threading.Thread(target=work, daemon=True).start()


@app.post("/api/recordings/{recording_id}/speakers/reset")
def reset_recording_speakers(recording_id: int):
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT r.file_path FROM recordings r "
            "JOIN transcripts t ON t.recording_id = r.id WHERE r.id = ?",
            (recording_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404,
                            detail="Recording has no transcript")
    if not Path(row[0]).exists():
        raise HTTPException(status_code=404, detail="Audio file is missing")
    if RESET_STATUS.get(recording_id) != "running":
        start_speaker_reset(recording_id)
    return {"status": "running"}


# Recovery actions: retry a failed recording, re-sync a Voice Memos
# original over a corrupt local copy, or reprocess a done recording.
# All run in the background; the UI polls the job endpoint.

def _recording_row(recording_id: int) -> tuple[Path, str]:
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT file_path, status FROM recordings WHERE id = ?",
            (recording_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="No such recording")
    return Path(row[0]), row[1]


@app.post("/api/recordings/{recording_id}/retry")
def retry_recording(recording_id: int):
    path, status = _recording_row(recording_id)
    # "transcribing" with nothing to show for it means the worker died
    # mid-run (a restart, a crash). The watcher never picks such a row
    # up again, because the file's hash is already in the database, so
    # Retry is the only way back and it has to accept that state.
    if status not in ("failed", "transcribing"):
        raise HTTPException(
            status_code=409,
            detail="Retry is for recordings that failed or got stuck "
                   f"mid-processing; this one is {status}. Use Reprocess "
                   "instead.")
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="The audio file is gone from the inbox. If it came "
                   "from Voice Memos, use \"Re-sync from source\".")
    ok, reason = reprocess.check_readable(path)
    if not ok:
        conn = db.connect()
        try:
            provenance.log_event(conn, provenance.RETRIED, path.stem,
                                 recording_id=recording_id,
                                 outcome="blocked-unreadable")
        finally:
            conn.close()
        raise HTTPException(status_code=422,
                            detail=reprocess.unreadable_message(reason))
    if not reprocess.start_retry(recording_id, path):
        raise HTTPException(status_code=409,
                            detail="A job is already running for this "
                                   "recording")
    return {"state": "running"}


@app.post("/api/recordings/{recording_id}/resync")
def resync_recording(recording_id: int):
    path, _status = _recording_row(recording_id)
    source = reprocess.find_memo_source(path.name)
    if source is None:
        raise HTTPException(
            status_code=404,
            detail=f"\"{path.stem}\" was not found in the Voice Memos "
                   "folder on this Mac, so there is no source to re-sync "
                   "from. (Deleted from Voice Memos, or not a Voice "
                   "Memos recording.)")
    if not reprocess.start_resync(recording_id, path, source):
        raise HTTPException(status_code=409,
                            detail="A job is already running for this "
                                   "recording")
    return {"state": "running"}


class ReprocessBody(BaseModel):
    scope: Literal["everything", "speakers", "notes"]
    # How many people are in the room. Pins pyannote's speaker count,
    # which otherwise collapses similar voices in noisy audio.
    num_speakers: int | None = None
    # Run speech enhancement before diarizing. Off by default: it can
    # hurt as easily as help.
    denoise: bool = False


@app.post("/api/recordings/{recording_id}/reprocess")
def reprocess_recording(recording_id: int, body: ReprocessBody):
    path, status = _recording_row(recording_id)
    if status != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Reprocess is for completed recordings; this one is "
                   f"{status}." + (" Use Retry." if status == "failed"
                                   else ""))
    conn = db.connect()
    try:
        has_transcript = conn.execute(
            "SELECT 1 FROM transcripts WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()
    finally:
        conn.close()
    if not has_transcript:
        raise HTTPException(status_code=404,
                            detail="Recording has no transcript")
    if body.scope in ("everything", "speakers"):
        if not path.exists():
            raise HTTPException(status_code=404,
                                detail="Audio file is missing")
        ok, reason = reprocess.check_readable(path)
        if not ok:
            raise HTTPException(status_code=422,
                                detail=reprocess.unreadable_message(reason))
    if body.num_speakers is not None and not (1 <= body.num_speakers <= 20):
        raise HTTPException(
            status_code=422,
            detail="Speaker count must be between 1 and 20")
    if not reprocess.start_reprocess(
            recording_id, path, body.scope,
            num_speakers=body.num_speakers, denoise=body.denoise):
        raise HTTPException(status_code=409,
                            detail="A job is already running for this "
                                   "recording")
    return {"state": "running"}


@app.post("/api/recordings/{recording_id}/diarization-suspect/dismiss")
def dismiss_diarization_suspect(recording_id: int):
    """Stop offering the reprocess suggestion for this recording."""
    conn = db.connect()
    try:
        cur = conn.execute(
            "UPDATE recordings SET diarization_suspect = 0 WHERE id = ?",
            (recording_id,),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="No such recording")
        return {"id": recording_id, "diarization_suspect": False}
    finally:
        conn.close()


@app.post("/api/recordings/{recording_id}/merged-name/dismiss")
def dismiss_merged_name(recording_id: int):
    """Stop mentioning the merged-name suggestion for this recording."""
    conn = db.connect()
    try:
        cur = conn.execute(
            "UPDATE recordings SET merged_name_dismissed = 1 WHERE id = ?",
            (recording_id,),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="No such recording")
        return {"id": recording_id, "merged_name_suspect": None}
    finally:
        conn.close()


@app.get("/api/recordings/{recording_id}/job")
def recording_job(recording_id: int):
    _path, status = _recording_row(recording_id)
    job = reprocess.job_status(recording_id)
    return {"recording_status": status, "job": job}


@app.get("/api/voices")
def list_voices():
    conn = db.connect()
    try:
        return voices.list_voices(conn)
    finally:
        conn.close()


class VoiceRename(BaseModel):
    new_name: str


@app.post("/api/voices/{name}/rename")
def rename_voice(name: str, body: VoiceRename):
    new_name = body.new_name.strip()
    if not new_name or new_name.startswith("SPEAKER_"):
        raise HTTPException(status_code=422, detail="Invalid name")
    conn = db.connect()
    try:
        try:
            changed = voices.rename_voice(conn, name, new_name)
        except ValueError as err:
            raise HTTPException(status_code=409, detail=str(err))
        if changed is None:
            raise HTTPException(status_code=404, detail="Voice not found")
        return {"name": new_name, "segments_changed": changed}
    finally:
        conn.close()


@app.delete("/api/voices/{name}")
def delete_voice(name: str):
    conn = db.connect()
    try:
        reverted = voices.delete_voice(conn, name)
        if reverted is None:
            raise HTTPException(status_code=404, detail="Voice not found")
        return {"deleted": name, "segments_reverted": reverted}
    finally:
        conn.close()


class TagAction(BaseModel):
    label: str


@app.post("/api/transcripts/{transcript_id}/speakers/confirm")
def confirm_auto_tag(transcript_id: int, body: TagAction):
    conn = db.connect()
    try:
        changed = voices.confirm_tag(conn, transcript_id, body.label)
        if changed == 0:
            raise HTTPException(status_code=404,
                                detail="No auto-tag to confirm")
        return {"confirmed": body.label, "segments": changed}
    finally:
        conn.close()


@app.post("/api/transcripts/{transcript_id}/speakers/reject")
def reject_auto_tag(transcript_id: int, body: TagAction):
    conn = db.connect()
    try:
        changed = voices.reject_tag(conn, transcript_id, body.label)
        if changed == 0:
            raise HTTPException(status_code=404,
                                detail="No auto-tag to reject")
        return {"rejected": body.label, "segments": changed}
    finally:
        conn.close()


if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"),
              name="assets")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/logo.svg")
    def logo():
        return FileResponse(STATIC_DIR / "logo.svg", media_type="image/svg+xml")
