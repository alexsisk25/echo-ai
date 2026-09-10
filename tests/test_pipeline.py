"""Pipeline tests. The fast ones fake the transcriber; the slow one runs
the real tiny model on a clip generated with the macOS say command."""

import shutil
import subprocess

import pytest

from app import db, summarize, transcribe, watcher


@pytest.fixture
def conn(tmp_path):
    c = db.connect(db_path=tmp_path / "test.db", key="ab" * 32)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def fake_summarize(monkeypatch):
    """No test in this module may hit the real LLM."""
    def fake(text, privileged=False):
        return summarize.MeetingSummary(
            title="Fake Meeting", summary="A fake summary.",
        )
    monkeypatch.setattr(watcher.summarize, "summarize", fake)


@pytest.fixture(autouse=True)
def fake_auto_tag(monkeypatch):
    """No test in this module may load the real voice model."""
    monkeypatch.setattr(watcher.voices, "auto_tag",
                        lambda conn, tid, path: {})


@pytest.fixture(autouse=True)
def fake_index(monkeypatch):
    """No test in this module may load the real embedding model."""
    monkeypatch.setattr(watcher.semantic, "index_transcript",
                        lambda conn, tid, text: 1)


@pytest.fixture(autouse=True)
def fake_diarize(monkeypatch):
    """No test in this module may load the real diarization model."""
    def fake(audio_path, num_speakers=None, **kwargs):
        return [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    monkeypatch.setattr(watcher.diarize, "diarize", fake)


@pytest.fixture
def fake_transcribe(monkeypatch):
    def fake(audio_path, model=None):
        return transcribe.TranscriptionResult(
            text="fake transcript",
            language="en",
            model="fake-model",
            segments=[{"start": 0.0, "end": 2.0, "text": "fake transcript"}],
            duration_seconds=2.0,
        )
    monkeypatch.setattr(watcher.transcribe, "transcribe", fake)


def make_audio(path):
    """Write a tiny real audio file so hashing and stability checks run."""
    path.write_bytes(b"RIFF" + b"\x00" * 128)


def test_process_file_stores_transcript(tmp_path, conn, fake_transcribe):
    audio = tmp_path / "meeting.m4a"
    make_audio(audio)
    watcher.process_file(audio, conn)

    row = conn.execute(
        "SELECT r.status, t.full_text FROM recordings r "
        "JOIN transcripts t ON t.recording_id = r.id"
    ).fetchone()
    assert row == ("done", "fake transcript")


def test_process_file_stores_speaker_labels(tmp_path, conn, fake_transcribe):
    audio = tmp_path / "meeting.m4a"
    make_audio(audio)
    watcher.process_file(audio, conn)

    speaker = conn.execute("SELECT speaker FROM segments").fetchone()[0]
    assert speaker == "SPEAKER_00"


def test_diarization_failure_keeps_transcript(tmp_path, conn,
                                              fake_transcribe, monkeypatch):
    def boom(audio_path):
        raise RuntimeError("no model")
    monkeypatch.setattr(watcher.diarize, "diarize", boom)

    audio = tmp_path / "meeting.m4a"
    make_audio(audio)
    watcher.process_file(audio, conn)

    row = conn.execute(
        "SELECT r.status, t.full_text FROM recordings r "
        "JOIN transcripts t ON t.recording_id = r.id"
    ).fetchone()
    assert row == ("done", "fake transcript")
    speaker = conn.execute("SELECT speaker FROM segments").fetchone()[0]
    assert speaker is None


def test_process_file_stores_summary(tmp_path, conn, fake_transcribe):
    audio = tmp_path / "meeting.m4a"
    make_audio(audio)
    watcher.process_file(audio, conn)

    stored = conn.execute(
        "SELECT s.summary_json FROM summaries s "
        "JOIN transcripts t ON t.id = s.transcript_id"
    ).fetchone()
    assert stored is not None
    assert "Fake Meeting" in stored[0]


def test_summary_failure_does_not_fail_recording(tmp_path, conn,
                                                 fake_transcribe, monkeypatch):
    def boom(text, privileged=False):
        raise RuntimeError("cloud is down")
    monkeypatch.setattr(watcher.summarize, "summarize", boom)

    audio = tmp_path / "meeting.m4a"
    make_audio(audio)
    watcher.process_file(audio, conn)

    status = conn.execute("SELECT status FROM recordings").fetchone()[0]
    assert status == "done"
    count = conn.execute("SELECT count(*) FROM summaries").fetchone()[0]
    assert count == 0


def test_process_file_skips_duplicates(tmp_path, conn, fake_transcribe):
    audio = tmp_path / "meeting.m4a"
    make_audio(audio)
    watcher.process_file(audio, conn)

    copy = tmp_path / "meeting-copy.m4a"
    shutil.copy(audio, copy)
    watcher.process_file(copy, conn)

    count = conn.execute("SELECT count(*) FROM recordings").fetchone()[0]
    assert count == 1


def test_process_file_ignores_non_audio(tmp_path, conn, fake_transcribe):
    note = tmp_path / "notes.txt"
    note.write_text("not audio")
    watcher.process_file(note, conn)
    count = conn.execute("SELECT count(*) FROM recordings").fetchone()[0]
    assert count == 0


def test_failure_marks_recording_failed(tmp_path, conn, monkeypatch):
    def boom(audio_path, model=None):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(watcher.transcribe, "transcribe", boom)

    audio = tmp_path / "meeting.m4a"
    make_audio(audio)
    watcher.process_file(audio, conn)

    status = conn.execute("SELECT status FROM recordings").fetchone()[0]
    assert status == "failed"


@pytest.mark.slow
def test_real_transcription_tiny_model(tmp_path, conn):
    """End to end with the real tiny model on generated speech."""
    aiff = tmp_path / "speech.aiff"
    wav = tmp_path / "speech.wav"
    subprocess.run(
        ["say", "-o", str(aiff), "the quick brown fox jumps over the lazy dog"],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
         "-ar", "16000", str(wav)],
        check=True,
    )

    watcher.process_file(wav, conn)

    row = conn.execute(
        "SELECT r.status, t.full_text FROM recordings r "
        "JOIN transcripts t ON t.recording_id = r.id"
    ).fetchone()
    assert row is not None
    status, text = row
    assert status == "done"
    assert "fox" in text.lower()


def test_deleted_recordings_file_reappearing_does_not_reimport(
        tmp_path, conn, fake_transcribe, monkeypatch):
    # The resurrection incident: a deleted recording's audio file shows
    # up in the folder again (leftover or restored copy). The folder
    # scan must block it like the sync ledger would, and say so in the
    # provenance log.
    from app import config, manage, sync
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(sync, "LEDGER", tmp_path / "data" / "ledger.txt")

    audio = tmp_path / "memo.m4a"
    make_audio(audio)
    watcher.process_file(audio, conn)
    rec_id = conn.execute("SELECT id FROM recordings").fetchone()[0]
    manage.delete_recording(conn, rec_id, how="single")
    assert not audio.exists()

    # The file comes back; processing it again must import nothing.
    make_audio(audio)
    watcher.process_file(audio, conn)
    assert conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0] == 0
    blocked = conn.execute(
        "SELECT stem FROM provenance WHERE event_type = "
        "'sync-skipped-as-deleted'").fetchall()
    assert blocked == [("memo",)]

    # A deliberate re-import via sync still works: sync writes a fresh
    # imported event, which unblocks the folder scan.
    from app import provenance
    provenance.log_imported(conn, "memo", "memo.m4a", ".m4a")
    watcher.process_file(audio, conn)
    assert conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0] == 1
