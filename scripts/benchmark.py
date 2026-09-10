"""Time Whisper model sizes on a real audio file to pick one for good.

Usage: .venv/bin/python scripts/benchmark.py path/to/recording.m4a
Each model downloads from Hugging Face on first use, so the first run
of each size is slower than the timing shown (download is excluded).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import transcribe  # noqa: E402

MODELS = [
    "mlx-community/whisper-tiny",
    "mlx-community/whisper-base-mlx",
    "mlx-community/whisper-small-mlx",
    "mlx-community/whisper-medium-mlx",
    "mlx-community/whisper-large-v3-turbo",
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    audio = Path(sys.argv[1])
    if not audio.exists():
        print(f"No such file: {audio}")
        return 1

    print(f"Benchmarking on {audio.name}\n")
    for model in MODELS:
        # Warm the model cache so download time does not pollute the timing.
        transcribe.transcribe(audio, model=model)
        start = time.perf_counter()
        result = transcribe.transcribe(audio, model=model)
        elapsed = time.perf_counter() - start
        duration = result.duration_seconds or 0
        speed = duration / elapsed if elapsed else 0
        print(f"{model}")
        print(f"  {elapsed:.1f}s for {duration:.0f}s of audio "
              f"({speed:.1f}x realtime)")
        print(f"  first words: {result.text[:80]}...\n")
    print("Pick the largest model that feels comfortable and set "
          "OTTER_WHISPER_MODEL or edit app/config.py, then record the "
          "choice in GOAL.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
