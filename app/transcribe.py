"""MLX Whisper wrapper: audio file in, transcript text and segments out."""

from dataclasses import dataclass, field
from pathlib import Path

from app import config


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    model: str
    segments: list[dict] = field(default_factory=list)
    duration_seconds: float | None = None


def transcribe(audio_path: str | Path,
               model: str | None = None) -> TranscriptionResult:
    """Transcribe an audio file with MLX Whisper.

    Imported lazily so tests that fake this function never load the model.
    """
    import mlx_whisper

    model = model or config.WHISPER_MODEL
    result = mlx_whisper.transcribe(
        str(audio_path), path_or_hf_repo=model
    )

    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]
    duration = segments[-1]["end"] if segments else None

    return TranscriptionResult(
        text=result["text"].strip(),
        language=result.get("language"),
        model=model,
        segments=segments,
        duration_seconds=duration,
    )
