"""Speaker diarization: audio file in, labeled speaker turns out, plus
the alignment that assigns a speaker to each transcript segment.

Runs fully local (pyannote on this machine). The model download needs a
HuggingFace token (HF_TOKEN in .env) the first time only.
"""

import os
import subprocess
import tempfile
from pathlib import Path

from app import config

# Formats libsndfile can read directly; anything else goes through ffmpeg.
DIRECT_FORMATS = {".wav", ".flac", ".ogg"}

_pipeline = None


def _get_pipeline():
    """Load the pyannote pipeline once per process (model load is slow)."""
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline

        # pyannote 3.x calls the token argument use_auth_token; 4.x
        # renamed it to token.
        try:
            _pipeline = Pipeline.from_pretrained(
                config.DIARIZATION_MODEL,
                use_auth_token=os.environ.get("HF_TOKEN"),
            )
        except TypeError:
            _pipeline = Pipeline.from_pretrained(
                config.DIARIZATION_MODEL, token=os.environ.get("HF_TOKEN")
            )

        # The Apple GPU runs this pipeline about 10x faster than CPU
        # (measured on real audio, identical output).
        import torch

        if torch.backends.mps.is_available():
            _pipeline.to(torch.device("mps"))
    return _pipeline


def diarize(audio_path: str | Path, num_speakers: int | None = None,
            min_speakers: int | None = None,
            max_speakers: int | None = None) -> list[dict]:
    """Return speaker turns as [{"start", "end", "speaker"}, ...].

    num_speakers pins the count when the user knows it (pyannote's own
    clustering decides otherwise, and it collapses two similar voices in
    noisy rooms). min/max bound the search without fixing it.
    """
    audio_path = Path(audio_path)
    tmp = None
    try:
        if audio_path.suffix.lower() not in DIRECT_FORMATS:
            # pyannote reads via libsndfile, which cannot decode m4a/mp3.
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
                 "-ar", "16000", "-ac", "1", tmp.name],
                check=True,
            )
            audio_path = Path(tmp.name)

        hints = {}
        if num_speakers is not None:
            hints["num_speakers"] = int(num_speakers)
        if min_speakers is not None:
            hints["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            hints["max_speakers"] = int(max_speakers)
        result = _get_pipeline()(str(audio_path), **hints)
        # pyannote 4.x wraps the annotation; older versions return it
        # directly.
        annotation = getattr(result, "speaker_diarization", result)
        return [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
    finally:
        if tmp is not None:
            os.unlink(tmp.name)


def channel_count(audio_path: str | Path) -> int:
    """Number of audio channels, via ffprobe. 0 if it cannot be read."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels", "-of",
             "csv=p=0", str(audio_path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return int(out) if out else 0
    except Exception:
        return 0


def _decode_channel(audio_path: str | Path, channel: int,
                    sample_rate: int = 16000) -> "np.ndarray":
    """Decode one channel (0=left/mic, 1=right/system) to mono samples."""
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
             "-af", f"pan=mono|c0=c{channel}", "-ar", str(sample_rate),
             tmp_name],
            check=True, capture_output=True, text=True,
        )
        import soundfile as sf

        data, _ = sf.read(tmp_name, dtype="float32")
        return data if data.ndim == 1 else data[:, 0]
    finally:
        os.unlink(tmp_name)


def channel_aware_assign(audio_path: str | Path, segments: list[dict],
                         user_name: str, sample_rate: int = 16000) -> bool:
    """Label segments from a two-channel captured meeting by which source
    was louder: the left (mic) channel is the user, the right (system)
    channel is everyone else. The user's lines are labeled user_name and
    marked pinned (ground truth: the mic is known to be the user, so no
    sweep or auto-tag may move them). Remote lines are diarized on the
    system channel only, so the user can never be merged with a remote
    voice. Mutates segments in place; returns True when it ran.

    Falls back (returns False) when the file is not two-channel.
    """
    import numpy as np

    if channel_count(audio_path) != 2:
        return False

    mic = _decode_channel(audio_path, 0, sample_rate)
    system = _decode_channel(audio_path, 1, sample_rate)

    def rms(sig, start, end):
        lo = max(0, int(start * sample_rate))
        hi = min(len(sig), int(end * sample_rate))
        if hi <= lo:
            return 0.0
        clip = sig[lo:hi]
        return float(np.sqrt(np.mean(clip * clip))) if len(clip) else 0.0

    # Diarize the system channel alone so remote voices separate among
    # themselves without ever seeing the user's voice.
    system_turns = []
    try:
        import numpy as _np
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sys_wav = tmp.name
        sf.write(sys_wav, system.astype(_np.float32), sample_rate)
        try:
            system_turns = diarize(sys_wav)
        finally:
            os.unlink(sys_wav)
    except Exception:
        system_turns = []

    for seg in segments:
        mic_e = rms(mic, seg["start"], seg["end"])
        sys_e = rms(system, seg["start"], seg["end"])
        if mic_e == 0.0 and sys_e == 0.0:
            seg["speaker"] = None
            continue
        if mic_e >= sys_e:
            seg["speaker"] = user_name
            seg["pinned"] = True
        else:
            # A remote line: use the system-channel diarization turn it
            # overlaps most, else a single generic remote label.
            best, overlap = None, 0.0
            for turn in system_turns:
                ov = min(seg["end"], turn["end"]) - max(
                    seg["start"], turn["start"])
                if ov > overlap:
                    overlap, best = ov, turn["speaker"]
            seg["speaker"] = best or "SPEAKER_00"
    return True


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Give each transcript segment the speaker it overlaps most with.

    Segments with no overlapping turn keep speaker None. Returns the same
    segment dicts with a "speaker" key filled in.
    """
    for seg in segments:
        best_speaker = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(seg["end"], turn["end"]) - max(seg["start"], turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        seg["speaker"] = best_speaker
    return segments
