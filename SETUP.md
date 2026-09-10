# Echo AI Setup

Echo AI is a local-first meeting intelligence app for Apple Silicon
Macs. Your audio and transcripts never leave your machine: transcription,
speaker labeling, voice recognition, and search embeddings all run
locally. Only plain meeting text is sent to a cloud LLM for summaries,
and recordings you mark privileged skip the cloud entirely and use a
local model.

This guide takes you from `git clone` to a working app. Expect 20 to 30
minutes, most of it model downloads.

## Requirements

- An Apple Silicon Mac (M1 or newer). The ML stack uses the Apple GPU;
  Intel Macs are not supported.
- macOS 14 or newer, with ffmpeg installed (`brew install ffmpeg`).
- Node.js 20 or newer (`brew install node`) for building the web UI.
- [uv](https://docs.astral.sh/uv/) for Python (`brew install uv`).
- An Anthropic API key (for cloud summaries).
- A free HuggingFace account (for the speaker-diarization models).

## 1. Clone and create the Python environment

```sh
git clone <your-fork-or-this-repo> local-otter
cd local-otter
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Python 3.12 specifically: the pinned ML stack (see requirements.txt for
why several pins exist) is tested against it.

## 2. Keys and model gates

```sh
cp .env.example .env
```

Then edit `.env`:

1. **ANTHROPIC_API_KEY**: create one at https://console.anthropic.com.
2. **HF_TOKEN**: create a read token at
   https://huggingface.co/settings/tokens, and accept the terms on both
   gated model pages (a click each, instant approval):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

The database encryption key is not something you configure: it is
generated on first run and stored in your macOS Keychain. If you ever
migrate Macs, that Keychain item must come along or the database is
unreadable.

## 3. Build the web UI

```sh
cd frontend
npm install
npm run build
cd ..
```

## 4. First run

```sh
./run.sh
```

This starts the web server and the folder watcher together; Ctrl-C stops
both. Open http://localhost:8000. Drop any audio file (m4a, mp3, wav)
into the `inbox/` folder and watch it appear with a transcript, speaker
labels, and AI notes.

The first recording is slow: it downloads the Whisper model (~1.5GB),
the diarization models (~500MB), the embedding model (~90MB), and on
first privileged use a local LLM (~1.8GB). After that everything is
warm.

## 5. iPhone recordings (optional)

Record with the built-in Voice Memos app and let iCloud sync bring
recordings to your Mac. Then the "Sync from iPhone" button in the app
header copies new memos into the inbox.

This requires Full Disk Access for the process running the app (Voice
Memos lives in a protected folder): System Settings > Privacy &
Security > Full Disk Access, add your terminal app, restart it.

Newer macOS Voice Memos stores recordings as `.qta` (a QuickTime
container) instead of `.m4a`. Sync handles both: `.qta` memos have
their audio extracted into an `.m4a` of the same name via ffmpeg on the
way into the inbox, so make sure `ffmpeg` is installed (`brew install
ffmpeg`; it is already needed for transcription). Nothing else changes.

Sync is manual by design; you decide what enters the archive. If you
want automatic background sync every 5 minutes, see the instructions
inside `scripts/com.local-otter.voicememo-sync.plist`.

## 6. Google Calendar (optional)

Auto-titles recordings from the calendar event they overlap and suggests
attendee names when labeling speakers. Until connected, a sample
calendar provider is used so the feature is testable.

1. In Google Cloud Console, create an OAuth client of type "Desktop
   app" with the Calendar API enabled, and download the JSON.
2. Save it as `google_credentials.json` in the project root (gitignored).
3. Click "Connect Google Calendar" in the app header. A browser window
   opens for the one-time Google login; the token is stored locally in
   `data/` (also gitignored).

## 7. Verify

```sh
.venv/bin/python -m pytest -m "not slow"   # Python suite
cd frontend && npm test                     # UI suite
```

Both should pass with zero failures.

## Privacy model, in one paragraph

Audio, transcripts, embeddings, and voiceprints live in an encrypted
SQLite database on your Mac (key in your Keychain). The only network
calls are: model downloads (one-time), and plain-text meeting content
sent to the Anthropic API for summaries, dossiers, translation, and
cross-meeting answers. Mark a recording "privileged" and even that text
stays local, analyzed by a small on-device model; the code fails closed
if the local model is missing rather than falling back to the cloud.

## Troubleshooting

- **"Cannot read the Voice Memos folder"**: grant Full Disk Access
  (step 5) to the app running the server, then restart it.
- **Diarization fails with a gated-repo error**: accept both model
  gates in step 2 with the same account that made your HF_TOKEN.
- **Port 8000 in use**: `OTTER_PORT=8010 ./run.sh`.
- **Dependency versions**: the diarization stack is pinned on purpose
  (pyannote.audio<4, torch 2.5.1, huggingface_hub<1.0, transformers<5).
  Upgrading any of them breaks a link in the chain; see the comments in
  requirements.txt.
