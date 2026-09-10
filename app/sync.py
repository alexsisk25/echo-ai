"""Copy new Voice Memos recordings into the watched inbox folder.

Voice Memos on iPhone syncs to this Mac via iCloud, landing in a protected
folder. Sync is deliberate and manual: the UI button (POST /api/sync) or
scripts/sync_voice_memos.py calls this. A ledger in data/synced_memos.txt
remembers what was already copied so nothing syncs twice.

Newer macOS Voice Memos stores recordings as .qta (a QuickTime container)
instead of .m4a. Both are handled: .m4a files are copied as-is, and .qta
files have their audio extracted into an .m4a of the same base name so
the rest of the pipeline (which expects .m4a) is unchanged.

The ledger dedupes by base name (stem) WITHOUT the extension. Apple's
.qta migration renamed existing memos from "X.m4a" to "X.qta"; a ledger
that keyed off the full filename treated "X.qta" as new and re-synced
(and re-imported) memos that had already been synced or deleted. Keying
off the stem makes a memo the same memo regardless of container. The
ledger file is migrated to stems on first load.

Whatever process runs this needs Full Disk Access to read the Voice Memos
folder.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

from app import config

log = logging.getLogger("otter.sync")

VOICE_MEMOS_DIR = (
    Path.home()
    / "Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings"
)
LEDGER = config.DATA_DIR / "synced_memos.txt"

# The containers Voice Memos writes; anything else in the folder is
# metadata we do not touch.
MEMO_SUFFIXES = (".m4a", ".qta")


class VoiceMemosUnreadable(PermissionError):
    """The Voice Memos folder exists but this process cannot read it.

    A PermissionError subclass, so every caller that already handles
    that keeps working, while logs can name the real cause.
    """


def list_memo_files(source_dir: Path) -> list[Path]:
    """Every memo in the folder, or an error if it cannot be read.

    Path.glob swallows PermissionError and yields nothing, so a process
    without Full Disk Access sees an empty folder and reports "nothing
    new" while the folder is in fact full of recordings. That happened
    live: a terminal with access saw 28 .qta files while the server's
    Sync button said there was nothing to do. Listing the directory
    explicitly turns that silence back into the error it always was.
    """
    try:
        names = os.listdir(source_dir)
    except FileNotFoundError:
        # Genuinely absent (Voice Memos never used, or a wrong path).
        # Not a permission problem, but never report it as "nothing new"
        # without saying so.
        log.warning("Voice Memos folder does not exist: %s", source_dir)
        return []
    except PermissionError as err:
        raise VoiceMemosUnreadable(
            f"Cannot read {source_dir}: Full Disk Access is missing"
        ) from err
    except OSError as err:
        raise VoiceMemosUnreadable(
            f"Cannot read {source_dir}: {err}") from err
    return sorted(
        source_dir / name for name in names
        if not name.startswith(".")
        and Path(name).suffix.lower() in MEMO_SUFFIXES
    )


def _load_ledger() -> set[str]:
    """Return the set of already-synced memo stems, migrating any legacy
    entries that still carry an extension to their stem."""
    if not LEDGER.exists():
        return set()
    return {
        Path(line.strip()).stem
        for line in LEDGER.read_text().splitlines()
        if line.strip()
    }


def _write_ledger(stems: set[str]) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text("\n".join(sorted(stems)) + "\n" if stems else "")


def _convert_qta(source: Path, dest_m4a: Path) -> None:
    """Extract a .qta's audio into an .m4a. Stream-copy the first audio
    stream (a .qta also carries a spatial-audio stream and data streams
    we do not want); re-encode to AAC only if the copy fails."""
    copy_cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
                "-map", "0:a:0", "-c:a", "copy", str(dest_m4a)]
    result = subprocess.run(copy_cmd, capture_output=True, text=True)
    if result.returncode == 0 and dest_m4a.exists() \
            and dest_m4a.stat().st_size > 0:
        return
    dest_m4a.unlink(missing_ok=True)
    encode_cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
                  "-map", "0:a:0", "-c:a", "aac", str(dest_m4a)]
    subprocess.run(encode_cmd, check=True, capture_output=True, text=True)


def sync_voice_memos(source_dir: Path | None = None, conn=None) -> int:
    """Copy new memos into inbox/, return how many were new.

    .m4a memos are copied as-is; .qta memos are converted to .m4a of the
    same base name. When conn is given, writes provenance events:
    imported for each new memo, and sync-skipped-as-deleted for a memo
    the ledger blocks that had previously been deleted (a blocked
    resurrection). Raises VoiceMemosUnreadable (a PermissionError) when
    the folder cannot be read, so "0 copied" always means the folder was
    read and held nothing new, never that the read silently failed.
    """
    from app import provenance

    source_dir = source_dir or VOICE_MEMOS_DIR
    # Raises VoiceMemosUnreadable rather than quietly reporting an empty
    # folder when the process lacks Full Disk Access.
    files = list_memo_files(source_dir)

    config.INBOX_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    synced = _load_ledger()

    copied = 0
    for f in sorted(files):
        # Dedupe by stem, so the same memo counts as synced whether the
        # folder now holds it as .m4a or .qta.
        if f.name.startswith("."):
            continue
        if f.stem in synced:
            # A blocked resurrection is worth recording: the ledger is
            # keeping a previously-deleted memo out.
            if conn is not None and provenance.was_deleted(conn, f.stem):
                provenance.log_sync_skipped(conn, f.stem)
            continue
        if f.suffix.lower() == ".qta":
            dest = config.INBOX_DIR / (f.stem + ".m4a")
            try:
                _convert_qta(f, dest)
            except subprocess.CalledProcessError:
                # A bad file must not stop the rest of the sync, and it
                # must not be marked synced (so a fixed copy retries).
                continue
        else:
            shutil.copy2(f, config.INBOX_DIR / f.name)
        synced.add(f.stem)
        copied += 1
        if conn is not None:
            provenance.log_imported(conn, f.stem, f.name, f.suffix.lower())

    # Persist as stems, migrating the file if it still held extensions.
    _write_ledger(synced)
    # Say which of the two "0 copied" cases this was: the folder was
    # read successfully and held nothing new, as opposed to unreadable,
    # which raised above and never gets here.
    log.info("Voice Memos folder read: %d memo(s) present, %d new",
             len(files), copied)
    return copied


def add_to_ledger(file_name: str) -> None:
    """Remember a memo (by stem) so sync never copies it in again.

    Used when a recording is deleted: without this, the next sync would
    bring the memo right back from the iPhone. Storing the stem means a
    memo deleted as X.m4a stays deleted even after Apple renames it to
    X.qta in the Voice Memos folder.
    """
    synced = _load_ledger()
    synced.add(Path(file_name).stem)
    _write_ledger(synced)


def remove_from_ledger(file_name: str) -> None:
    """Forget a memo (by stem) so it may be copied from source again.

    Used by re-sync from source: the stem leaves the ledger while a
    fresh copy replaces a corrupt local file, then goes right back in.
    """
    synced = _load_ledger()
    synced.discard(Path(file_name).stem)
    _write_ledger(synced)
