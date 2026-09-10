# Echo AI

A local-first meeting intelligence app for Apple Silicon Macs. Drop in audio (or sync recordings from your iPhone's Voice Memos) and get a searchable, private archive: transcripts with speaker labels, AI summaries and action items, semantic search across every meeting, and voices that get recognized automatically once you name them.

The privacy model is the point: audio never leaves your machine. Transcription, speaker diarization, voice fingerprinting, and search embeddings all run locally on the Apple GPU. Only meeting text goes to a cloud LLM for summaries and cross-meeting questions, and recordings you mark as privileged skip the cloud entirely, analyzed by a local model instead, with a gateway that fails closed rather than ever falling back to the cloud.

## What it does

Transcription runs on MLX Whisper (large-v3-turbo, roughly 15x realtime on an M-series Mac). pyannote separates speakers, and renaming a speaker once enrolls their voice so future recordings auto-tag them. Every meeting gets a validated summary with decisions, action items, and owners. Search works by keyword (FTS5) and by meaning (local embeddings in sqlite-vec), and an Ask feature answers questions across your whole archive with cited sources. A People tab builds a dossier per named speaker; a Commitments tab tracks every promise made in any meeting, linked to the moment it was said. Recordings can be matched to Google Calendar events for auto-titling. Transcripts and summaries translate between English and Spanish. Everything lives in one SQLCipher-encrypted SQLite database, with the key in your macOS Keychain.

## Getting started

See [SETUP.md](SETUP.md) for the full walkthrough from `git clone` to a working app, including the API keys you'll need (Anthropic, and a free HuggingFace token for the diarization models) and the optional Google Calendar connection.

## Stack

Python FastAPI worker, React (Vite) UI served locally, MLX Whisper, pyannote, SpeechBrain ECAPA-TDNN, SQLCipher + FTS5 + sqlite-vec, LiteLLM as the model gateway (Claude by default, swappable), mlx-lm for the local privileged-mode model.

## License

MIT. See [LICENSE](LICENSE).
