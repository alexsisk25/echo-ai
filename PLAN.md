# Local Otter: Build Spec

This is the technical spec for the app. GOAL.md tracks progress against it. Build in phases. Each phase lists what to build, how to verify it, and a suggested effort level. Do not jump ahead. Verify a phase before starting the next.

## Architecture in one paragraph
A Python FastAPI worker runs the heavy pipeline (transcription, diarization, embeddings) and serves a local web UI at localhost. All audio and transcripts are stored on this Mac in an encrypted SQLite database. A LiteLLM gateway sends only text to a cloud LLM for summaries, action items, and cross-meeting questions, and can be pointed at a local model for privileged content. n8n watches a folder and orchestrates the steps. Nothing about the audio leaves the machine.

## Why local-first
Privacy is the point. Some recordings may be sensitive or legally privileged. Local storage means no per-minute cloud fees, full data ownership, offline access, and no vendor lock-in. The only recurring cost is text-only LLM API calls, realistically five to thirty dollars a month for one heavy user.

## Core stack (do not swap without asking)
- Transcription: MLX Whisper (fastest on Apple Silicon). Fallback: whisper.cpp. Pick medium or large-v3-turbo once benchmarked on real audio.
- Diarization: pyannote (community-1 or 3.1). WhisperX is an option but falls back to CPU on Apple Silicon, so it is slower.
- Voice fingerprinting: SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb), 192-dim embeddings, cosine similarity against enrolled voiceprints.
- Storage: SQLite encrypted with SQLCipher (AES-256). Keyword search via FTS5. Semantic search via sqlite-vec in the same database.
- Embeddings: a local model (BGE-M3 or nomic-embed via Ollama) for privileged content. A cloud embedding model is fine for non-sensitive content, but pick one and commit, because changing it later means re-embedding everything.
- Backend: Python FastAPI. Frontend: React or Next served locally.
- LLM gateway: LiteLLM. Pin to version 1.82.6 or earlier (two later releases shipped malware). Put the API key in an environment variable or the macOS Keychain, never in code. The key lives in .env (gitignored) as ANTHROPIC_API_KEY.
- Model routing default: route summaries, action items, and other routine per-meeting calls to a cheap model (Claude Haiku, or Sonnet at most). Reserve expensive models for cross-meeting Q&A, and only if quality demands it. Model names are config values, not hard-coded strings, so routing can change without code edits.
- Orchestration: n8n, self-hosted and run natively (its Local File Trigger does not fire reliably inside Docker on macOS, and is disabled by default from v2.0 so it must be enabled).

## Reference projects worth reading first (low effort)
Study Meetily (whisper.cpp plus SQLite plus vector DB plus bring-your-own Claude/Ollama), Speakr (self-hosted, browser upload, diarization), and aTrain (local Whisper, ships large-v3-turbo). Meetily is closest to this design and may be forkable. Read these before scaffolding so you copy patterns rather than inventing them.

## Phase 1: Capture and transcribe (MVP). Effort: high
Build:
- SQLite schema: recordings, transcripts, segments. Encrypt with SQLCipher, key from Keychain or env var.
- A folder-watch listener that fires when a new audio file lands.
- An MLX Whisper transcribe function that writes the transcript to the database.
- iPhone path: Voice Memos with iCloud sync, then a small script or Folder Action that exports new recordings into the watched folder.
Verify:
- Automated tests for the transcribe function and the database write.
- Manually drop a real audio file and confirm a transcript row appears.
Benchmark: time tiny through medium Whisper models on your own audio, keep the largest that runs comfortably, record the choice in GOAL.md decisions.

## Phase 2: Searchable archive and summaries. Effort: high for the LLM layer, medium for UI
Build:
- FTS5 keyword search over transcripts.
- A simple web UI: a list of recordings, and a transcript view.
- LiteLLM summaries and action items. Prompt at temperature 0 to 0.1 for a strict JSON schema: title, date, attendees, summary, decisions, action_items (owner, task, due_date, priority), risks, topics. Validate with Pydantic. Add a cleanup pass if the JSON is malformed.
Verify:
- Search returns the right transcripts.
- Every transcript gets a valid, schema-checked summary and action-item list.

## Phase 3: Diarization and calendar. Effort: high
Build:
- pyannote diarization producing speaker labels per segment.
- A one-click relabel control in the UI (labels are assistive, not perfect, so make correcting them fast).
- Google Calendar match: find the event whose time window overlaps the recording, auto-title the meeting, pull attendee names as context and as candidate speaker labels. Code defensively for moved or duplicate events, and fall back to asking me when ambiguous.
Verify:
- Speaker labels appear and can be corrected.
- A recording lands with the right calendar title and attendees.
Expectation to bake into the UI: two clean speakers label well (around ninety percent), three or more with cross-talk drops to seventy-five to eighty-five percent. Do not promise perfection.

## Phase 4: Semantic search and voice fingerprinting. Effort: xhigh
Build:
- Embed transcripts and store vectors in sqlite-vec. Add semantic search plus an LLM synthesis pass so I can ask "what did we decide about X" across the whole archive.
- Speaker enrollment: store an averaged ECAPA-TDNN embedding per known person from clean samples. For each new segment, cosine-match against enrolled voiceprints, auto-tag above a threshold, else label unknown. Start the threshold conservative (around 0.5) and require two or more matching segments before auto-tagging.
Verify:
- Cross-meeting question returns a correct, sourced answer.
- A known voice in a new recording gets the right name, and an unknown voice is not falsely tagged.
Honest caveat: voice matching that looks near-perfect on clean benchmarks degrades badly on noisy meeting audio. Treat it as assistive with easy manual correction.

## Phase 5: Envelope features. Effort: high, medium for glue
Build any of, in this order of value:
- Cross-meeting intelligence (already seeded in Phase 4): the killer feature.
- Person dossiers: an LLM-maintained profile per recognized voice (what they care about, commitments made, follow-ups owed).
- Commitment tracker: extract every promise tagged to speaker and due date, surfaced as a running list with overdue alerts.
- Privacy vault: a separate encrypted store whose analysis is pinned to a local model (Ollama), no cloud calls ever. "Cloud-allowed" is an explicit per-recording flag defaulting to off for anything privileged.
- Multilingual: Whisper auto-detects language and can translate Spanish to English in one pass. Build bilingual transcripts.
- n8n action pipelines: auto-draft follow-up emails, create Notion pages, post Slack summaries. These send things outward, so they must ask before sending.
Verify: each feature demoable end to end.

## Security rules (apply from Phase 1)
- Keep FileVault on (full-disk encryption).
- Encrypt the database with SQLCipher on top. Key in the macOS Keychain, not in code.
- Owner-only file permissions (chmod 600) on the database and its journal files.
- Privileged recordings analyze with a local model only. Never send privileged text to a cloud API.
- Respect recording-consent laws. Get consent as required.

## Where a beginner will struggle (so plan extra time)
- Hardest: installing and version-matching the Python ML stack (pyannote, torch, faster-whisper) in a fresh virtual environment, and tuning the voice-fingerprint threshold.
- Medium: aligning transcript words to speaker segments, SQLCipher key management, Google Calendar OAuth, n8n folder-watch reliability.
- Easiest: the LiteLLM layer, prompt and schema design, the web UI, the SQLite schema and FTS5, and n8n workflows.
