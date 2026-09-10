"""Recovery actions for existing recordings: retry, re-sync, reprocess.

Three ways to run the pipeline again without losing human work:

- retry: a failed recording gets the full pipeline again from its
  existing audio file. An unreadable file (no moov atom) is refused up
  front with guidance toward re-sync.
- resync: a Voice Memos recording whose local copy went bad is deleted
  locally, re-copied (re-converted if .qta) from the Voice Memos
  folder, and reprocessed under the same recording id.
- reprocess: a done recording re-runs everything, speakers only, or
  notes only.

The invariant for every path: human work survives. Manual titles,
pinned segments, human-set speaker names, tag rejections, My notes and
enhanced notes, folder assignment, commitment checkboxes, and the
privileged flag are never cleared. Re-transcription replaces segment
rows, so pinned and human-named lines are re-attached to the new
segments by time overlap. Re-diarization only replaces machine labels
(generic SPEAKER_XX, auto-tags, or unlabeled) on unpinned lines.
Calendar matching runs only when the recording has no calendar match
row at all, so existing matches and dismissals stand.

Jobs run on background threads; the UI polls GET /api/recordings/{id}/job.
Every action writes a provenance event when it finishes (and a blocked
retry writes one too), so the paper trail stays complete.
"""

import logging
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from app import (commitments, config, db, merged, semantic, summarize,
                 sync, transcribe, voices)

log = logging.getLogger("otter.reprocess")

# recording_id -> {"action": str, "state": running|done|failed,
#                  "error": str | None}. One job per recording at a time.
JOBS: dict[int, dict] = {}
_jobs_lock = threading.Lock()

SCOPES = ("everything", "speakers", "notes")

# Matches segments whose speaker is machine output: unlabeled, a generic
# diarization label, or a voice auto-tag (auto_original set). Everything
# else is a name a human chose.
_MACHINE_SEGMENT = (
    "pinned = 0 AND (speaker IS NULL OR "
    "speaker LIKE 'SPEAKER^_%' ESCAPE '^' OR auto_original IS NOT NULL)"
)


def check_readable(path: Path) -> tuple[bool, str]:
    """Can ffmpeg decode this file? Returns (ok, reason-if-not)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    if probe.returncode == 0 and probe.stdout.strip():
        return True, ""
    return False, (probe.stderr or "").strip()


def unreadable_message(reason: str) -> str:
    detail = "the file has no moov atom, so no audio can be read" \
        if "moov" in reason.lower() else "ffmpeg cannot read any audio from it"
    return (f"The audio file itself is broken ({detail}). Retrying the "
            "pipeline cannot fix a bad file. If this recording came from "
            "Voice Memos, use \"Re-sync from source\" to fetch a fresh "
            "copy from the Voice Memos folder.")


def job_status(recording_id: int) -> dict | None:
    with _jobs_lock:
        job = JOBS.get(recording_id)
        return dict(job) if job else None


def _start_job(recording_id: int, action: str, work) -> bool:
    """Register and launch a background job. False if one is running."""
    with _jobs_lock:
        if JOBS.get(recording_id, {}).get("state") == "running":
            return False
        JOBS[recording_id] = {"action": action, "state": "running",
                              "error": None}

    def run():
        conn = db.connect()
        try:
            work(conn)
            with _jobs_lock:
                JOBS[recording_id] = {"action": action, "state": "done",
                                      "error": None}
        except Exception as err:
            log.exception("%s failed for recording %d", action, recording_id)
            with _jobs_lock:
                JOBS[recording_id] = {"action": action, "state": "failed",
                                      "error": str(err)}
        finally:
            conn.close()

    threading.Thread(target=run, daemon=True).start()
    return True


def _log_event(conn, event_type: str, recording_id: int, **context) -> None:
    """Best-effort provenance write: never break the action itself."""
    from app import provenance
    try:
        row = conn.execute(
            "SELECT file_path FROM recordings WHERE id = ?",
            (recording_id,),
        ).fetchone()
        stem = Path(row[0]).stem if row else str(recording_id)
        provenance.log_event(conn, event_type, stem,
                             recording_id=recording_id, **context)
    except Exception:
        log.exception("Provenance write failed (%s)", event_type)


def _snapshot_human_segments(conn, transcript_id: int) -> list[dict]:
    """The segments a human touched: pinned lines and human-set names."""
    rows = conn.execute(
        "SELECT start_seconds, end_seconds, speaker, pinned FROM segments "
        "WHERE transcript_id = ? AND (pinned = 1 OR (auto_original IS NULL "
        "AND speaker IS NOT NULL "
        "AND speaker NOT LIKE 'SPEAKER^_%' ESCAPE '^'))",
        (transcript_id,),
    ).fetchall()
    return [{"start": r[0], "end": r[1], "speaker": r[2], "pinned": r[3]}
            for r in rows]


def _reapply_human_segments(conn, transcript_id: int,
                            saved: list[dict]) -> int:
    """Re-attach saved human labels to the new segments by time overlap.

    Each saved line claims the new segment it overlaps most (and at
    least half of the shorter of the two), the same alignment idea the
    diarizer uses. Returns how many labels were re-applied.
    """
    if not saved:
        return 0
    new_rows = conn.execute(
        "SELECT id, start_seconds, end_seconds FROM segments "
        "WHERE transcript_id = ?", (transcript_id,),
    ).fetchall()
    applied = 0
    for s in saved:
        best, best_overlap, best_dur = None, 0.0, 0.0
        for seg_id, start, end in new_rows:
            overlap = min(s["end"], end) - max(s["start"], start)
            if overlap > best_overlap:
                best, best_overlap, best_dur = seg_id, overlap, end - start
        if best is None:
            continue
        shorter = min(s["end"] - s["start"], best_dur)
        if shorter > 0 and best_overlap >= shorter / 2:
            conn.execute(
                "UPDATE segments SET speaker = ?, pinned = ?, "
                "auto_original = NULL WHERE id = ?",
                (s["speaker"], s["pinned"], best),
            )
            applied += 1
    conn.commit()
    return applied


def _diarize_fresh(path: Path, segments: list[dict],
                   num_speakers: int | None = None,
                   denoise: bool = False) -> None:
    """Fresh speaker labels for the given segment dicts, channel-aware
    for captured meetings, plain diarization otherwise.

    num_speakers pins pyannote's speaker count when the user knows it.
    denoise runs a speech-enhancement pass first, for noisy rooms; it
    only affects what the diarizer hears, never the stored audio.
    """
    from app import denoise as denoise_mod
    from app import diarize

    if Path(path).name.endswith("-meeting.m4a"):
        if diarize.channel_aware_assign(str(path), segments,
                                        config.USER_NAME):
            return

    heard = Path(path)
    cleaned = denoise_mod.enhance(path) if denoise else None
    if cleaned is not None:
        heard = cleaned
    try:
        turns = diarize.diarize(heard, num_speakers=num_speakers)
    finally:
        if cleaned is not None:
            cleaned.unlink(missing_ok=True)
    diarize.assign_speakers(segments, turns)


def _full_process(conn, recording_id: int, path: Path,
                  num_speakers: int | None = None,
                  denoise: bool = False) -> None:
    """Transcribe and enrich an existing recording row, preserving human
    work. Used by retry, resync, and reprocess-everything."""
    existing = conn.execute(
        "SELECT id FROM transcripts WHERE recording_id = ?", (recording_id,),
    ).fetchone()
    saved = _snapshot_human_segments(conn, existing[0]) if existing else []

    db.set_recording_status(conn, recording_id, "transcribing")
    try:
        result = transcribe.transcribe(path)
        try:
            _diarize_fresh(path, result.segments,
                           num_speakers=num_speakers, denoise=denoise)
        except Exception:
            log.exception("Diarization failed for %s", path.name)

        if existing:
            transcript_id = existing[0]
            # Same transcript row, new content: rejections, commitments,
            # and translations keep their foreign key. The FTS update
            # trigger reindexes the text.
            conn.execute(
                "UPDATE transcripts SET full_text = ?, language = ?, "
                "model = ? WHERE id = ?",
                (result.text, result.language, result.model, transcript_id),
            )
            conn.execute("DELETE FROM segments WHERE transcript_id = ?",
                         (transcript_id,))
            conn.executemany(
                "INSERT INTO segments (transcript_id, start_seconds, "
                "end_seconds, text, speaker, pinned) VALUES (?, ?, ?, ?, ?, ?)",
                [(transcript_id, s["start"], s["end"], s["text"],
                  s.get("speaker"), 1 if s.get("pinned") else 0)
                 for s in result.segments],
            )
            # New text means cached translations describe the old text.
            conn.execute("DELETE FROM translations WHERE transcript_id = ?",
                         (transcript_id,))
            conn.commit()
            _reapply_human_segments(conn, transcript_id, saved)
        else:
            transcript_id = db.insert_transcript(
                conn, recording_id, result.text, result.language,
                result.model, result.segments,
            )

        if result.duration_seconds is not None:
            conn.execute(
                "UPDATE recordings SET duration_seconds = ? WHERE id = ?",
                (result.duration_seconds, recording_id),
            )
            conn.commit()
        db.set_recording_status(conn, recording_id, "done")
    except Exception:
        db.set_recording_status(conn, recording_id, "failed")
        raise

    # Enrichment, best-effort like the watcher. Calendar matching only
    # when the recording has never been matched or dismissed, so human
    # calendar decisions stand.
    try:
        from app import calendar_sync
        has_match = conn.execute(
            "SELECT 1 FROM calendar_matches WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()
        if not has_match:
            calendar_sync.match_recording(conn, recording_id)
    except Exception:
        log.exception("Calendar match failed for %s", path.name)
    split = {}
    try:
        split = voices.guided_split(conn, transcript_id, str(path))
    except Exception:
        log.exception("Guided split failed for %s", path.name)
    try:
        voices.auto_tag(conn, transcript_id, str(path),
                        exclude=voices.split_pairs(split))
    except Exception:
        log.exception("Voice auto-tag failed for %s", path.name)
    db.refresh_diarization_suspect(conn, recording_id)
    try:
        merged.refresh(conn, recording_id)
    except Exception:
        log.exception("Merged-name check failed for %s", path.name)
    try:
        semantic.index_transcript(conn, transcript_id, result.text)
    except Exception:
        log.exception("Embedding index failed for %s", path.name)
    _summarize(conn, recording_id, transcript_id, result.text)


def _summarize(conn, recording_id: int, transcript_id: int,
               text: str) -> None:
    """Fresh AI summary plus commitment extraction. The manual title
    lives in its own column and commitments.extract keeps existing rows
    (and their open/done state), only adding new ones."""
    privileged = db.is_privileged(conn, recording_id)
    summary = summarize.summarize(text, privileged=privileged)
    db.upsert_summary(
        conn, transcript_id, summary.model_dump_json(),
        model="local" if privileged else config.LLM_MODEL,
    )
    commitments.extract(conn, transcript_id)


def _respeaker(conn, recording_id: int, path: Path,
               num_speakers: int | None = None,
               denoise: bool = False) -> int:
    """Fresh machine labels for machine-labeled, unpinned lines only.
    Pinned lines, human names, and tag rejections are untouched."""
    transcript_id = conn.execute(
        "SELECT id FROM transcripts WHERE recording_id = ?", (recording_id,),
    ).fetchone()[0]
    rows = conn.execute(
        f"SELECT id, start_seconds, end_seconds FROM segments "
        f"WHERE transcript_id = ? AND {_MACHINE_SEGMENT}",
        (transcript_id,),
    ).fetchall()
    seg_dicts = [{"id": r[0], "start": r[1], "end": r[2]} for r in rows]
    if seg_dicts:
        _diarize_fresh(path, seg_dicts, num_speakers=num_speakers,
                       denoise=denoise)
        for s in seg_dicts:
            conn.execute(
                "UPDATE segments SET speaker = ?, auto_original = NULL, "
                "pinned = ? WHERE id = ?",
                (s.get("speaker"), 1 if s.get("pinned") else 0, s["id"]),
            )
        conn.commit()
    # Look inside each cluster before naming it, so a cluster holding
    # two voices is separated rather than named after one of them.
    split = {}
    try:
        split = voices.guided_split(conn, transcript_id, str(path))
    except Exception:
        log.exception("Guided split failed for %s", path.name)
    try:
        voices.auto_tag(conn, transcript_id, str(path),
                        exclude=voices.split_pairs(split))
    except Exception:
        log.exception("Voice auto-tag failed for %s", path.name)
    db.refresh_diarization_suspect(conn, recording_id)
    try:
        merged.refresh(conn, recording_id)
    except Exception:
        log.exception("Merged-name check failed for %s", path.name)
    return len(seg_dicts)


def start_retry(recording_id: int, path: Path) -> bool:
    def work(conn):
        try:
            _full_process(conn, recording_id, path)
        except Exception:
            _log_event(conn, "retried", recording_id, outcome="failed")
            raise
        _log_event(conn, "retried", recording_id, outcome="done")

    return _start_job(recording_id, "retry", work)


def find_memo_source(file_name: str) -> Path | None:
    """The Voice Memos file this recording came from, if it exists."""
    stem = Path(file_name).stem
    for suffix in (".m4a", ".qta"):
        candidate = sync.VOICE_MEMOS_DIR / (stem + suffix)
        if candidate.exists():
            return candidate
    return None


def start_resync(recording_id: int, local_path: Path,
                 source: Path) -> bool:
    def work(conn):
        try:
            _resync(conn, recording_id, local_path, source)
        except Exception:
            _log_event(conn, "resynced", recording_id, outcome="failed",
                       source_format=source.suffix.lower())
            raise
        _log_event(conn, "resynced", recording_id, outcome="done",
                   source_format=source.suffix.lower())

    return _start_job(recording_id, "resync", work)


def _resync(conn, recording_id: int, local_path: Path,
            source: Path) -> None:
    """Replace the local audio with a fresh copy from Voice Memos, then
    rerun the pipeline under the same recording id."""
    workdir = Path(tempfile.mkdtemp(prefix="otter-resync-"))
    try:
        fresh = workdir / local_path.name
        if source.suffix.lower() == ".qta":
            sync._convert_qta(source, fresh)
        else:
            shutil.copy2(source, fresh)
        ok, reason = check_readable(fresh)
        if not ok:
            raise RuntimeError(
                "The copy in the Voice Memos folder is also unreadable "
                f"({reason or 'ffmpeg cannot decode it'}). The local file "
                "was left untouched.")

        # The fresh bytes belong to this recording id: point the row's
        # hash at them (so the watcher's duplicate check skips the file
        # event the move fires) before the file lands in the inbox.
        from app import watcher
        digest = watcher.file_hash(fresh)
        clash = conn.execute(
            "SELECT id FROM recordings WHERE file_hash = ? AND id != ?",
            (digest, recording_id),
        ).fetchone()
        if clash:
            raise RuntimeError(
                f"The fresh copy is byte-identical to recording "
                f"{clash[0]}; nothing was replaced.")
        conn.execute("UPDATE recordings SET file_hash = ? WHERE id = ?",
                     (digest, recording_id))
        conn.commit()

        # Ledger out, fresh copy in, ledger back: the stem leaves the
        # ledger only for the swap, so a concurrent sync cannot skip it
        # as already-synced with bad bytes, and it never re-imports as
        # a second recording afterwards.
        sync.remove_from_ledger(local_path.name)
        local_path.unlink(missing_ok=True)
        shutil.move(str(fresh), local_path)
        sync.add_to_ledger(local_path.name)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    _full_process(conn, recording_id, local_path)


def start_reprocess(recording_id: int, path: Path, scope: str,
                    num_speakers: int | None = None,
                    denoise: bool = False) -> bool:
    def work(conn):
        try:
            if scope == "everything":
                _full_process(conn, recording_id, path,
                              num_speakers=num_speakers, denoise=denoise)
            elif scope == "speakers":
                _respeaker(conn, recording_id, path,
                           num_speakers=num_speakers, denoise=denoise)
            else:
                transcript_id, text = conn.execute(
                    "SELECT id, full_text FROM transcripts "
                    "WHERE recording_id = ?", (recording_id,),
                ).fetchone()
                _summarize(conn, recording_id, transcript_id, text)
        except Exception:
            _log_event(conn, "reprocessed", recording_id, scope=scope,
                       outcome="failed", num_speakers=num_speakers,
                       denoise=denoise)
            raise
        _log_event(conn, "reprocessed", recording_id, scope=scope,
                   outcome="done", num_speakers=num_speakers,
                   denoise=denoise)

    return _start_job(recording_id, f"reprocess-{scope}", work)
