"""Speaker-count hints handed from recording to processing.

When a recording starts in person, the user can say how many people are
in the room. That number is worth a lot to diarization, but the audio
file lands in inbox/ minutes later and the recording row does not exist
until the watcher creates it, so the hint needs somewhere to wait.

It waits here: a small JSON map of file stem to speaker count in data/.
The watcher takes the hint when it processes the file, which also
removes it, so a hint is used once and never leaks onto a later
recording that happens to share a name.
"""

import json
import logging
import threading
from pathlib import Path

from app import config

log = logging.getLogger("otter.hints")

_lock = threading.Lock()


def _path() -> Path:
    return config.DATA_DIR / "speaker_hints.json"


def _read() -> dict:
    try:
        return json.loads(_path().read_text())
    except (OSError, ValueError):
        return {}


def _write(data: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps(data, indent=1, sort_keys=True))


def put(file_name: str, num_speakers: int | None) -> None:
    """Remember how many speakers a recording is expected to have."""
    if not num_speakers:
        return
    with _lock:
        data = _read()
        data[Path(file_name).stem] = int(num_speakers)
        _write(data)
    log.info("Speaker hint for %s: %d", file_name, num_speakers)


def take(file_name: str) -> int | None:
    """Read and consume the hint for this file, if there is one."""
    stem = Path(file_name).stem
    with _lock:
        data = _read()
        value = data.pop(stem, None)
        if value is not None:
            _write(data)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def peek(file_name: str) -> int | None:
    """The hint for this file without consuming it (for tests and UI)."""
    value = _read().get(Path(file_name).stem)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
