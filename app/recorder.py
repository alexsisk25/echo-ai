"""Bot-free meeting capture: record system audio plus the mic.

A small Swift helper (scripts/otter_record.swift, compiled on demand
with the system toolchain) captures system audio via ScreenCaptureKit
and the microphone via AVFoundation into two wav files. On stop, the
two tracks are mixed with ffmpeg into one m4a named like a Voice Memo
(so recorded_at parses the start time) and dropped into inbox/, where
the watcher runs the normal pipeline. Nothing leaves the machine.

The first ever start triggers macOS permission prompts (microphone and
system-audio recording); until granted, start reports an error state
with the helper's guidance.
"""

import logging
import shutil
import signal
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from app import config

log = logging.getLogger("otter.recorder")

HELPER_SRC = config.PROJECT_ROOT / "scripts" / "otter_record.swift"
HELPER_BIN = config.DATA_DIR / "otter-record"

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_workdir: Path | None = None
_started_at: datetime | None = None
_state = "idle"          # idle | starting | recording | error
_error: str | None = None
_last_file: str | None = None

# Capture modes. "online" records mic plus system audio into the
# two-channel file the channel-aware pipeline expects (mic = the user,
# pinned). "in_person" records the mic only: everyone is in the room on
# one channel, so the file goes down the standard mono pipeline with
# full diarization and no channel assumptions.
MODES = ("online", "in_person")
_mode = "online"
# How many people are in the room, for an in-person capture. Handed to
# diarization through app.hints when the file lands in inbox/.
_num_speakers: int | None = None

# Live input levels (0..1 RMS) and when each source was last heard, so
# the UI can show meters and warn about a source that has gone silent.
_mic_level = 0.0
_system_level = 0.0
_mic_last_sound: datetime | None = None
_system_last_sound: datetime | None = None

# Above this RMS a source counts as "hearing something".
SOUND_FLOOR = 0.005
# Warn after this many seconds of silence from a source while recording.
SILENCE_WARN_SECONDS = 10.0


def ensure_helper() -> Path:
    """Compile the capture helper if missing or older than its source."""
    if HELPER_BIN.exists() and (
            HELPER_BIN.stat().st_mtime >= HELPER_SRC.stat().st_mtime):
        return HELPER_BIN
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["swiftc", "-O", "-parse-as-library", str(HELPER_SRC),
         "-o", str(HELPER_BIN)],
        check=True, capture_output=True, text=True,
    )
    return HELPER_BIN


def _parse_level(line: str) -> None:
    """Update levels from a 'LEVEL mic=.. [system=..]' helper line.

    In mic-only capture the helper emits no system field; the system
    meter and its last-heard clock are then left untouched.
    """
    global _mic_level, _system_level, _mic_last_sound, _system_last_sound
    mic = sys = None
    for part in line.split()[1:]:
        key, _, val = part.partition("=")
        try:
            if key == "mic":
                mic = float(val)
            elif key == "system":
                sys = float(val)
        except ValueError:
            return
    if mic is None:
        return
    now = datetime.now()
    with _lock:
        _mic_level = mic
        if mic >= SOUND_FLOOR:
            _mic_last_sound = now
        if sys is not None:
            _system_level = sys
            if sys >= SOUND_FLOOR:
                _system_last_sound = now


def _watch_output(proc: subprocess.Popen) -> None:
    """Track the helper's lifecycle from its stdout on a thread."""
    global _state, _error
    for line in proc.stdout:
        line = line.strip()
        if line == "STARTED":
            with _lock:
                if _state == "starting":
                    _state = "recording"
        elif line.startswith("LEVEL "):
            _parse_level(line)
        elif line.startswith("ERROR"):
            with _lock:
                _state = "error"
                _error = line[len("ERROR"):].strip()
    # stdout closed: the helper exited. An exit without STOPPED while
    # we thought we were recording is a failure.
    proc.wait()
    with _lock:
        if _state in ("starting", "recording") and proc.returncode != 0:
            _state = "error"
            _error = _error or (
                f"Recorder exited unexpectedly (code {proc.returncode})")


def start(mode: str = "online",
          num_speakers: int | None = None) -> dict:
    """Launch the capture helper in the given mode. Returns the status."""
    global _proc, _workdir, _started_at, _state, _error, _mode, _num_speakers
    global _mic_level, _system_level, _mic_last_sound, _system_last_sound
    if mode not in MODES:
        raise ValueError(f"Unknown capture mode: {mode}")
    with _lock:
        if _state in ("starting", "recording"):
            return {"state": _state, "error": None}
        _state = "starting"
        _error = None
        _mode = mode
        # Only meaningful in person: an online capture separates by
        # channel, which already knows there are two sides.
        _num_speakers = num_speakers if mode == "in_person" else None
        # A source is given a grace period from the start before it can
        # be called silent, so "last heard" begins at launch time. In
        # person there is no system source: its clock stays unset so the
        # meeting-audio silence warning can never fire.
        now = datetime.now()
        _mic_level = _system_level = 0.0
        _mic_last_sound = now
        _system_last_sound = now if mode == "online" else None
    try:
        helper = ensure_helper()
        workdir = Path(tempfile.mkdtemp(prefix="otter-rec-"))
        args = [str(helper), str(workdir)]
        if mode == "in_person":
            args.append("mic-only")
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except Exception as err:
        with _lock:
            _state = "error"
            _error = f"Could not start the recorder: {err}"
        return status()
    with _lock:
        _proc, _workdir, _started_at = proc, workdir, datetime.now()
    threading.Thread(target=_watch_output, args=(proc,), daemon=True).start()
    return status()


def stop() -> dict:
    """Stop capture, mix the track(s), drop the m4a into inbox/."""
    global _proc, _workdir, _started_at, _state, _error, _last_file
    with _lock:
        proc, workdir, started_at = _proc, _workdir, _started_at
        current, mode, num_speakers = _state, _mode, _num_speakers
    if proc is None or current not in ("starting", "recording"):
        raise RuntimeError("Not recording")

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    system_wav = workdir / "system.wav"
    mic_wav = workdir / "mic.wav"
    sources = (mic_wav,) if mode == "in_person" else (system_wav, mic_wav)
    tracks = [p for p in sources if p.exists() and p.stat().st_size > 44]
    try:
        if not tracks:
            raise RuntimeError(
                "The recorder produced no audio. If this was the first "
                "run, grant the permissions it asked for and try again.")
        # "-meeting" marks a channel-aware capture for the pipeline
        # (mic = the user, pinned). An in-person capture must not carry
        # it: everyone is on the one mic channel, so it gets the plain
        # mono pipeline. Both prefixes parse as recorded_at.
        marker = "-inperson" if mode == "in_person" else "-meeting"
        name = f"{started_at:%Y%m%d %H%M%S}{marker}.m4a"
        out = workdir / name
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        if system_wav in tracks and mic_wav in tracks:
            # Channel-aware capture: keep each source's identity by
            # placing the mic (the user) on the left channel and the
            # system audio (everyone else) on the right. The pipeline
            # uses this to label the user's lines and diarize only the
            # remote channel, so the user is never merged with a remote
            # voice. Both resampled to 16 kHz mono before the join.
            cmd += ["-i", str(mic_wav), "-i", str(system_wav),
                    "-filter_complex",
                    "[0:a]aresample=16000,pan=mono|c0=c0[m];"
                    "[1:a]aresample=16000,pan=mono|c0=c0[s];"
                    "[m][s]join=inputs=2:channel_layout=stereo[out]",
                    "-map", "[out]"]
        else:
            # Single track: a normal one-channel file for the standard
            # mono pipeline (in-person capture, or a source that never
            # produced audio).
            for t in tracks:
                cmd += ["-i", str(t)]
            cmd += ["-ac", "1"]
        cmd += ["-c:a", "aac", str(out)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        config.INBOX_DIR.mkdir(parents=True, exist_ok=True)
        # Park the speaker count BEFORE the file lands, so the watcher
        # can never see the audio without its hint.
        if num_speakers:
            from app import hints
            hints.put(name, num_speakers)
        shutil.move(str(out), config.INBOX_DIR / name)
    except Exception as err:
        with _lock:
            _proc, _workdir, _started_at = None, None, None
            _state = "error"
            _error = str(err)
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    shutil.rmtree(workdir, ignore_errors=True)
    with _lock:
        _proc, _workdir, _started_at = None, None, None
        _state = "idle"
        _error = None
        _last_file = name
    log.info("Meeting recording saved to inbox: %s", name)
    return {"state": "idle", "file": name}


def status() -> dict:
    with _lock:
        seconds = (
            int((datetime.now() - _started_at).total_seconds())
            if _started_at and _state in ("starting", "recording") else 0
        )
        return {
            "state": _state,
            "seconds": seconds,
            "mode": _mode,
            "num_speakers": _num_speakers,
            "error": _error,
            "last_file": _last_file,
        }


def levels() -> dict:
    """Live input levels plus a warning when a source has gone silent.

    Returns mic/system levels (0..1), and warnings naming any source
    that has produced no sound for SILENCE_WARN_SECONDS while recording.
    """
    with _lock:
        recording = _state == "recording"
        mic, sysl = _mic_level, _system_level
        mic_last, sys_last = _mic_last_sound, _system_last_sound
    warnings = []
    if recording:
        now = datetime.now()
        if mic_last and (now - mic_last).total_seconds() >= \
                SILENCE_WARN_SECONDS:
            warnings.append({
                "source": "mic",
                "message": "Not hearing your microphone. Check it is not "
                           "muted and is the right input device."})
        if sys_last and (now - sys_last).total_seconds() >= \
                SILENCE_WARN_SECONDS:
            warnings.append({
                "source": "system",
                "message": "Not hearing any meeting audio. Make sure the "
                           "call is playing out loud on this Mac (not on "
                           "headphones) and is not muted."})
    return {
        "recording": recording,
        "mic": round(mic, 4),
        "system": round(sysl, 4),
        "warnings": warnings,
    }


def reset_error() -> None:
    """Back to idle after a failed start (e.g. permission just granted)."""
    global _state, _error, _proc, _workdir, _started_at
    with _lock:
        if _state == "error":
            _state = "idle"
            _error = None
            _proc, _workdir, _started_at = None, None, None
