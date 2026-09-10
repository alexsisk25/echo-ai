"""Central configuration: paths, model choice, and where the DB key lives."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The folder we watch for new audio files.
INBOX_DIR = Path(os.environ.get("OTTER_INBOX", PROJECT_ROOT / "inbox"))

# Where the encrypted database lives.
DATA_DIR = Path(os.environ.get("OTTER_DATA", PROJECT_ROOT / "data"))
DB_PATH = DATA_DIR / "otter.db"

# Parked Voice Memos history waiting to be processed.
BACKLOG_DIR = Path(os.environ.get("OTTER_BACKLOG", PROJECT_ROOT / "backlog"))

# Keychain identity for the database encryption key.
KEYCHAIN_SERVICE = "local-otter"
KEYCHAIN_ACCOUNT = "db-key"

# Whisper model, chosen via scripts/benchmark.py on real audio (2026-07-07).
# large-v3-turbo beat medium on both speed (15.5x vs 13.7x realtime) and
# accuracy on this machine. Decision recorded in GOAL.md.
WHISPER_MODEL = os.environ.get(
    "OTTER_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"
)

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".mp4"}

# Local speaker diarization model (runs on this machine, gated download).
# 3.1 because that is the gate accepted on this HF token; PLAN.md allows
# community-1 or 3.1.
DIARIZATION_MODEL = os.environ.get(
    "OTTER_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
)

# Local embedding model for semantic search. Committed choice: changing
# it means re-embedding every transcript.
EMBEDDING_MODEL = os.environ.get(
    "OTTER_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Calendar provider: "auto" uses Google once google_credentials.json
# exists in the project root, else the fake provider. Force with
# "fake" or "google".
CALENDAR_PROVIDER = os.environ.get("OTTER_CALENDAR_PROVIDER", "auto")

# Local speaker-recognition model for voice fingerprinting.
VOICE_MODEL = os.environ.get(
    "OTTER_VOICE_MODEL", "speechbrain/spkrec-ecapa-voxceleb"
)

# Local LLM for privileged recordings (privacy vault). Runs via mlx-lm,
# entirely on this machine.
LOCAL_LLM_MODEL = os.environ.get(
    "OTTER_LOCAL_LLM_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit"
)

# Cloud LLM for summaries and action items, via LiteLLM. Only text is sent.
# Haiku is the cheap default per GOAL.md; override with OTTER_LLM_MODEL.
LLM_MODEL = os.environ.get("OTTER_LLM_MODEL", "anthropic/claude-haiku-4-5")

# What to label the user's own voice in a captured meeting. The mic
# track is known to be the user, so those segments are labeled this
# and pinned; remote participants (system track) are diarized.
USER_NAME = os.environ.get("OTTER_USER_NAME", "Me")


def _load_dotenv(path=PROJECT_ROOT / ".env"):
    """Load KEY=VALUE lines from .env so LiteLLM finds the API key.
    Values already set in the environment win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()
