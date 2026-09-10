"""Channel-aware diarization: mic = user (pinned), system = diarized.

A captured meeting is a two-channel file with the user's mic on the
left and everyone else on the right. These tests build a synthetic
stereo file (left tone in the first half, right tone in the second)
and confirm the user is labeled and pinned while the remote channel is
diarized, so the two can never merge.
"""

import time

import numpy as np
import pytest
import soundfile as sf

from app import config, db, diarize


@pytest.fixture
def stereo_file(tmp_path):
    sr = 16000
    dur = 4
    t = np.linspace(0, dur, sr * dur, endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    left = tone.copy()
    left[sr * 2:] = 0.0          # user (mic) speaks in the first 2s
    right = tone.copy()
    right[:sr * 2] = 0.0         # remote (system) speaks in the last 2s
    path = tmp_path / "20260725 120000-meeting.wav"
    sf.write(path, np.stack([left, right], axis=1), sr)
    return path


def test_channel_count(stereo_file, tmp_path):
    assert diarize.channel_count(stereo_file) == 2
    mono = tmp_path / "mono.wav"
    sf.write(mono, np.zeros(16000, dtype=np.float32), 16000)
    assert diarize.channel_count(mono) == 1


def test_mic_segments_are_user_and_pinned(stereo_file, monkeypatch):
    # The system channel diarizes to one remote speaker in the 2-4s span.
    monkeypatch.setattr(diarize, "diarize",
                        lambda p: [{"start": 2.0, "end": 4.0,
                                    "speaker": "SPEAKER_00"}])
    segments = [
        {"start": 0.0, "end": 2.0, "text": "hi from me"},
        {"start": 2.0, "end": 4.0, "text": "hi from them"},
    ]
    ran = diarize.channel_aware_assign(stereo_file, segments, "Me")
    assert ran is True
    # First segment: mic-dominant -> the user, pinned.
    assert segments[0]["speaker"] == "Me"
    assert segments[0]["pinned"] is True
    # Second segment: system-dominant -> the diarized remote label.
    assert segments[1]["speaker"] == "SPEAKER_00"
    assert segments[1].get("pinned") is not True


def test_two_sided_yields_two_distinct_speakers(stereo_file, monkeypatch):
    monkeypatch.setattr(diarize, "diarize",
                        lambda p: [{"start": 2.0, "end": 4.0,
                                    "speaker": "SPEAKER_00"}])
    segments = [
        {"start": 0.0, "end": 1.0, "text": "a"},
        {"start": 1.0, "end": 2.0, "text": "b"},
        {"start": 2.0, "end": 3.0, "text": "c"},
        {"start": 3.0, "end": 4.0, "text": "d"},
    ]
    diarize.channel_aware_assign(stereo_file, segments, "Me")
    speakers = {s["speaker"] for s in segments}
    # Exactly two speakers: the user and the one remote. The user is
    # never blended into the remote cluster.
    assert speakers == {"Me", "SPEAKER_00"}
    assert "Me" not in [s["speaker"] for s in segments if s["start"] >= 2.0]


def test_falls_back_on_mono(tmp_path):
    mono = tmp_path / "voice-memo.wav"
    sf.write(mono, np.zeros(16000, dtype=np.float32), 16000)
    segments = [{"start": 0.0, "end": 1.0, "text": "x"}]
    assert diarize.channel_aware_assign(mono, segments, "Me") is False


def test_non_diarize_overhead_is_small(stereo_file, monkeypatch):
    # With diarization mocked out, the channel split + energy pass on a
    # 4-second clip must be quick (the added work vs the existing single
    # diarization pass is well under the 2x abandon threshold).
    monkeypatch.setattr(diarize, "diarize", lambda p: [])
    segments = [{"start": i * 0.5, "end": i * 0.5 + 0.5, "text": "x"}
                for i in range(8)]
    t0 = time.time()
    diarize.channel_aware_assign(stereo_file, segments, "Me")
    assert time.time() - t0 < 5.0


def test_insert_transcript_persists_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "ca.db")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)
    conn = db.connect()
    rec = db.insert_recording(conn, str(tmp_path / "m.m4a"), "h1", 4.0)
    tid = db.insert_transcript(
        conn, rec, "full", "en", "tiny",
        [{"start": 0, "end": 2, "text": "me", "speaker": "Me",
          "pinned": True},
         {"start": 2, "end": 4, "text": "them", "speaker": "SPEAKER_00"}])
    rows = conn.execute(
        "SELECT speaker, pinned FROM segments WHERE transcript_id = ? "
        "ORDER BY start_seconds", (tid,)).fetchall()
    conn.close()
    assert rows == [("Me", 1), ("SPEAKER_00", 0)]
