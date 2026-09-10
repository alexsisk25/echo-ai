"""One-shot batch processor: run every unprocessed audio file through the
full pipeline (transcribe, diarize, summarize).

Processes what is already in inbox/ first, then moves backlog/ files into
inbox/ in small batches. A file that fails is logged and skipped, never
fatal. Progress lands in data/process_all.log so a rerun is easy to audit;
already-processed files are skipped by hash, so rerunning is safe.

Run with: .venv/bin/python -m scripts.process_all
"""

import logging
import shutil
import time

from app import config, db, semantic, summarize, watcher

BATCH_SIZE = 5
LOG_PATH = config.DATA_DIR / "process_all.log"


def audio_files(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in config.AUDIO_EXTENSIONS
    )


def process(path, conn, log):
    started = time.time()
    try:
        watcher.process_file(path, conn)
        log.info("ok %s (%.0fs)", path.name, time.time() - started)
        return True
    except Exception:
        log.exception("FAILED %s", path.name)
        return False


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("otter.process_all")
    conn = db.connect()
    failures = []

    # A recording stuck mid-flight from an interrupted run has a status
    # row but no transcript. Drop the stub so the file gets reprocessed
    # (its hash would otherwise make the dedupe check skip it forever).
    stale = conn.execute(
        "DELETE FROM recordings WHERE status != 'done' AND id NOT IN "
        "(SELECT recording_id FROM transcripts)"
    ).rowcount
    conn.commit()
    if stale:
        log.info("cleared %d interrupted recording stub(s)", stale)

    inbox = audio_files(config.INBOX_DIR)
    log.info("inbox pass: %d files", len(inbox))
    for path in inbox:
        if not process(path, conn, log):
            failures.append(path.name)

    backlog = audio_files(config.BACKLOG_DIR)
    log.info("backlog pass: %d files in batches of %d", len(backlog),
             BATCH_SIZE)
    for i in range(0, len(backlog), BATCH_SIZE):
        batch = backlog[i:i + BATCH_SIZE]
        log.info("batch %d/%d", i // BATCH_SIZE + 1,
                 (len(backlog) + BATCH_SIZE - 1) // BATCH_SIZE)
        for src in batch:
            dest = config.INBOX_DIR / src.name
            shutil.move(str(src), str(dest))
            if not process(dest, conn, log):
                failures.append(dest.name)

    # Repair pass: transcripts whose summary failed earlier (for example
    # a network blip during the LLM call) get one more attempt.
    missing = conn.execute(
        "SELECT t.id, t.full_text FROM transcripts t "
        "LEFT JOIN summaries s ON s.transcript_id = t.id WHERE s.id IS NULL"
    ).fetchall()
    if missing:
        log.info("summary repair pass: %d transcripts", len(missing))
        for tid, text in missing:
            try:
                privileged = db.transcript_privileged(conn, tid)
                result = summarize.summarize(text, privileged=privileged)
                db.upsert_summary(conn, tid, result.model_dump_json(),
                                  "local" if privileged else config.LLM_MODEL)
                log.info("summary ok for transcript %d", tid)
            except Exception:
                log.exception("summary FAILED for transcript %d", tid)
                failures.append(f"summary:{tid}")

    # Repair pass: transcripts not yet in the semantic index.
    unindexed = conn.execute(
        "SELECT t.id, t.full_text FROM transcripts t "
        "WHERE t.id NOT IN (SELECT DISTINCT transcript_id FROM chunks)"
    ).fetchall()
    if unindexed:
        log.info("embedding repair pass: %d transcripts", len(unindexed))
        for tid, text in unindexed:
            try:
                n = semantic.index_transcript(conn, tid, text)
                log.info("indexed transcript %d (%d chunks)", tid, n)
            except Exception:
                log.exception("indexing FAILED for transcript %d", tid)
                failures.append(f"index:{tid}")

    conn.close()
    if failures:
        log.warning("done with %d failures: %s", len(failures),
                    ", ".join(failures))
    else:
        log.info("done, no failures")


if __name__ == "__main__":
    main()
