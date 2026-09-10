"""Watch the inbox folder and run the pipeline on each new audio file.

Run with: .venv/bin/python -m app.watcher
"""

import hashlib
import logging
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app import (calendar_sync, commitments, config, db, diarize, hints,
                 merged, semantic, summarize, transcribe, voices)

log = logging.getLogger("otter.watcher")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wait_until_stable(path: Path, interval: float = 1.0,
                      timeout: float = 300.0) -> bool:
    """Wait until the file stops growing. iCloud and AirDrop write in chunks,
    so reacting to the first event would read a half-written file."""
    deadline = time.time() + timeout
    last_size = -1
    while time.time() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last_size and size > 0:
            return True
        last_size = size
        time.sleep(interval)
    return False


def process_file(path: Path, conn) -> None:
    """Hash, dedupe, transcribe, and store one audio file."""
    if path.suffix.lower() not in config.AUDIO_EXTENSIONS:
        return
    if not wait_until_stable(path):
        log.warning("File never stabilized, skipping: %s", path)
        return

    digest = file_hash(path)
    if db.recording_exists(conn, digest):
        log.info("Duplicate, skipping: %s", path.name)
        return

    # A deleted recording's file reappearing in the folder must not
    # resurrect it (same contract as the sync ledger, which the folder
    # scan used to bypass). The block is visible in the Activity log.
    from app import provenance
    if provenance.blocked_as_deleted(conn, path.stem):
        provenance.log_sync_skipped(conn, path.stem)
        log.info("Blocked resurrection of deleted recording: %s", path.name)
        return

    recording_id = db.insert_recording(conn, str(path), digest)
    db.set_recording_status(conn, recording_id, "transcribing")
    log.info("Transcribing %s", path.name)
    try:
        result = transcribe.transcribe(path)
        # Speaker labels are best-effort: diarization failure must not
        # lose the transcript.
        try:
            # A captured meeting is stereo with known channels (mic =
            # the user on the left, system audio = everyone else on the
            # right). Label by channel so the user is never merged with
            # a remote voice; fall back to normal diarization otherwise.
            channel_aware = False
            if path.name.endswith("-meeting.m4a"):
                channel_aware = diarize.channel_aware_assign(
                    str(path), result.segments, config.USER_NAME)
            if channel_aware:
                log.info("Channel-aware labels for %s (mic=%s, system "
                         "diarized)", path.name, config.USER_NAME)
            else:
                # An in-person capture can carry the speaker count the
                # user entered when starting the recording.
                hint = hints.take(path.name)
                turns = diarize.diarize(path, num_speakers=hint)
                diarize.assign_speakers(result.segments, turns)
                log.info("Diarized %s: %d speaker turns%s", path.name,
                         len(turns),
                         f" (hint: {hint} speakers)" if hint else "")
        except Exception:
            log.exception("Diarization failed for %s", path.name)
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
        log.info("Done: %s", path.name)
        # Everything past this point is best-effort enrichment: a failure
        # must not lose the transcript.
        try:
            match = calendar_sync.match_recording(conn, recording_id)
            if match:
                log.info("Calendar match for %s: %s", path.name,
                         match["title"])
        except Exception:
            log.exception("Calendar match failed for %s", path.name)
        split = {}
        try:
            # Look inside each cluster before naming it, so a cluster
            # holding two voices is separated rather than named after
            # whichever one it matched.
            split = voices.guided_split(conn, transcript_id, str(path))
        except Exception:
            log.exception("Guided split failed for %s", path.name)
        try:
            tagged = voices.auto_tag(conn, transcript_id, str(path),
                                     exclude=voices.split_pairs(split))
            if tagged:
                log.info("Auto-tagged speakers in %s: %s", path.name, tagged)
        except Exception:
            log.exception("Voice auto-tag failed for %s", path.name)
        try:
            if db.refresh_diarization_suspect(conn, recording_id):
                log.warning(
                    "%s is %.0f min but diarized to one speaker; the UI "
                    "will suggest a reprocess with a speaker count",
                    path.name, (result.duration_seconds or 0) / 60)
        except Exception:
            log.exception("Suspect check failed for %s", path.name)
        try:
            name = merged.refresh(conn, recording_id)
            if name:
                log.warning("Two distinct voices appear under \"%s\" in "
                            "%s; the UI will suggest a fix", name, path.name)
        except Exception:
            log.exception("Merged-name check failed for %s", path.name)
        try:
            n_chunks = semantic.index_transcript(
                conn, transcript_id, result.text
            )
            log.info("Indexed %s: %d chunks", path.name, n_chunks)
        except Exception:
            log.exception("Embedding index failed for %s", path.name)
        try:
            privileged = db.is_privileged(conn, recording_id)
            summary = summarize.summarize(result.text, privileged=privileged)
            db.upsert_summary(
                conn, transcript_id, summary.model_dump_json(),
                model="local" if privileged else config.LLM_MODEL,
            )
            log.info("Summarized: %s", path.name)
            n_new = commitments.extract(conn, transcript_id)
            if n_new:
                log.info("Extracted %d commitment(s) from %s", n_new,
                         path.name)
        except Exception:
            log.exception("Summary failed for %s", path.name)
    except Exception:
        db.set_recording_status(conn, recording_id, "failed")
        log.exception("Transcription failed for %s", path.name)


class InboxHandler(FileSystemEventHandler):
    """Handles events on watchdog's observer thread. SQLite connections
    cannot cross threads, so the handler opens its own on first use."""

    def __init__(self):
        self.conn = None

    def _get_conn(self):
        if self.conn is None:
            self.conn = db.connect()
        return self.conn

    def on_created(self, event):
        if not event.is_directory:
            process_file(Path(event.src_path), self._get_conn())

    def on_moved(self, event):
        # Finder and iCloud often write to a temp name then rename.
        if not event.is_directory:
            process_file(Path(event.dest_path), self._get_conn())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config.INBOX_DIR.mkdir(parents=True, exist_ok=True)

    # Catch up on files that arrived while the watcher was not running.
    conn = db.connect()
    for path in sorted(config.INBOX_DIR.iterdir()):
        if path.is_file():
            process_file(path, conn)
    conn.close()

    observer = Observer()
    observer.schedule(InboxHandler(), str(config.INBOX_DIR))
    observer.start()
    log.info("Watching %s", config.INBOX_DIR)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
