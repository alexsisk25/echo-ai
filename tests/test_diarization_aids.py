"""Diarization aids: speaker-count hints, guided splitting, denoise,
and the suspicious-result flag.

The failure these exist for: a 42-minute two-person restaurant
conversation diarized as one speaker, then auto-tagged wholly as the
user.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, db, diarize, hints, main, reprocess, voices


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "diar.db")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "INBOX_DIR", tmp_path / "inbox")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)
    (tmp_path / "inbox").mkdir()
    conn = db.connect()
    yield TestClient(main.app), conn, tmp_path
    conn.close()


def make_audio(path):
    """A real, tiny, decodable file: the reprocess endpoint refuses to
    start on audio ffprobe cannot read."""
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=1", "-c:a", "aac", "-f", "mov",
         str(path)],
        check=True, capture_output=True, text=True)


def seed(conn, tmp_path, duration=2400.0, segments=None):
    audio = config.INBOX_DIR / "20260721 122422-ROOM.m4a"
    make_audio(audio)
    rec_id = db.insert_recording(conn, str(audio), "hash-room", duration)
    conn.execute("UPDATE recordings SET status='done' WHERE id=?", (rec_id,))
    segs = segments if segments is not None else [
        {"start": float(i * 10), "end": float(i * 10 + 8),
         "text": f"line {i}", "speaker": "SPEAKER_00"}
        for i in range(12)
    ]
    tid = db.insert_transcript(conn, rec_id, "room talk", "en", "m", segs)
    conn.commit()
    return rec_id, tid, audio


# --- 1. Speaker-count hint -------------------------------------------

def test_diarize_passes_num_speakers_to_pyannote(monkeypatch):
    seen = {}

    class FakePipeline:
        def __call__(self, path, **kwargs):
            seen.update(kwargs)
            return _FakeAnnotation()

    class _FakeAnnotation:
        def itertracks(self, yield_label=False):
            return iter([])

    monkeypatch.setattr(diarize, "_get_pipeline", lambda: FakePipeline())
    diarize.diarize("x.wav", num_speakers=2)
    assert seen == {"num_speakers": 2}

    seen.clear()
    diarize.diarize("x.wav", min_speakers=2, max_speakers=5)
    assert seen == {"min_speakers": 2, "max_speakers": 5}

    seen.clear()
    diarize.diarize("x.wav")
    assert seen == {}, "no hint means pyannote decides, as before"


def test_reprocess_speakers_forwards_hint_and_denoise(env, monkeypatch):
    client, conn, tmp_path = env
    rec_id, tid, audio = seed(conn, tmp_path)
    calls = {}

    def fake_diarize_fresh(path, segments, num_speakers=None, denoise=False):
        calls["num_speakers"] = num_speakers
        calls["denoise"] = denoise
        for s in segments:
            s["speaker"] = "SPEAKER_00"
    monkeypatch.setattr(reprocess, "_diarize_fresh", fake_diarize_fresh)
    monkeypatch.setattr(voices, "guided_split", lambda *a, **k: {"split": {}})
    monkeypatch.setattr(voices, "auto_tag", lambda *a, **k: {})

    resp = client.post(f"/api/recordings/{rec_id}/reprocess",
                       json={"scope": "speakers", "num_speakers": 2,
                             "denoise": True})
    assert resp.status_code == 200
    deadline = __import__("time").time() + 10
    while reprocess.job_status(rec_id)["state"] == "running":
        if __import__("time").time() > deadline:
            break
        __import__("time").sleep(0.05)
    assert reprocess.job_status(rec_id)["state"] == "done"
    assert calls == {"num_speakers": 2, "denoise": True}


def test_reprocess_rejects_an_absurd_speaker_count(env):
    client, conn, tmp_path = env
    rec_id, _, _ = seed(conn, tmp_path)
    for bad in (0, -3, 21):
        resp = client.post(f"/api/recordings/{rec_id}/reprocess",
                           json={"scope": "speakers", "num_speakers": bad})
        assert resp.status_code == 422


def test_in_person_hint_survives_until_the_watcher_takes_it(env):
    _client, _conn, tmp_path = env
    hints.put("20260721 122422-inperson.m4a", 3)
    # Peeking leaves it in place; taking consumes it exactly once.
    assert hints.peek("20260721 122422-inperson.m4a") == 3
    assert hints.take("20260721 122422-inperson.m4a") == 3
    assert hints.take("20260721 122422-inperson.m4a") is None
    # A recording with no hint simply has none.
    assert hints.take("something-else.m4a") is None


def test_record_start_stores_the_room_size_for_in_person(env, monkeypatch):
    client, _conn, _tmp = env
    started = {}
    monkeypatch.setattr(
        main.recorder, "start",
        lambda mode="online", num_speakers=None: started.update(
            mode=mode, num_speakers=num_speakers) or {"state": "recording"})
    client.post("/api/record/start",
                json={"mode": "in_person", "num_speakers": 4})
    assert started == {"mode": "in_person", "num_speakers": 4}
    assert client.post("/api/record/start",
                       json={"mode": "in_person",
                             "num_speakers": 99}).status_code == 422


# --- 2. Voiceprint-guided split ---------------------------------------

def _enroll(conn, name, vector):
    v = np.array(vector, dtype=np.float32)
    v = v / np.linalg.norm(v)
    conn.execute(
        "INSERT INTO voiceprints (name, embedding, num_samples) "
        "VALUES (?, ?, ?)", (name, v.tobytes(), 10))
    conn.commit()


def _two_voice_cluster(conn, tmp_path, n_each=6):
    """One SPEAKER_00 cluster whose segments are really two people."""
    segs = [{"start": float(i * 30), "end": float(i * 30 + 25),
             "text": f"line {i}", "speaker": "SPEAKER_00"}
            for i in range(n_each * 2)]
    return seed(conn, tmp_path, segments=segs)


def test_guided_split_separates_an_enrolled_voice_from_the_rest(
        env, monkeypatch):
    client, conn, tmp_path = env
    rec_id, tid, audio = _two_voice_cluster(conn, tmp_path)
    _enroll(conn, "Alex Sisk", [1.0, 0.0, 0.0])

    # Alternating speakers: even segments are Alex, odd are a stranger.
    monkeypatch.setattr(voices, "_load_audio", lambda p: np.zeros(16000 * 400))

    def fake_embed(audio_data, segments):
        out = []
        for s in segments:
            near_alex = int(s["start"] // 30) % 2 == 0
            v = np.array([1.0, 0.02, 0.0] if near_alex else [0.0, 1.0, 0.0],
                         dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.array(out)
    monkeypatch.setattr(voices, "embed_segments", fake_embed)

    result = voices.guided_split(conn, tid, str(audio))
    assert result["split"] == {"SPEAKER_00": {"Alex Sisk": 6}}

    rows = conn.execute(
        "SELECT speaker, COUNT(*), SUM(CASE WHEN auto_original IS NOT NULL "
        "THEN 1 ELSE 0 END) FROM segments WHERE transcript_id=? "
        "GROUP BY speaker ORDER BY speaker", (tid,)).fetchall()
    # Alex's half is named and revertible; the stranger keeps the label.
    assert rows == [("Alex Sisk", 6, 6), ("SPEAKER_00", 6, 0)]


def test_guided_split_leaves_a_single_voice_cluster_alone(env, monkeypatch):
    """Everything matching one enrolled voice is auto_tag's job, not a
    split: there is no second person to separate out."""
    client, conn, tmp_path = env
    rec_id, tid, audio = _two_voice_cluster(conn, tmp_path)
    _enroll(conn, "Alex Sisk", [1.0, 0.0, 0.0])
    monkeypatch.setattr(voices, "_load_audio", lambda p: np.zeros(16000 * 400))
    monkeypatch.setattr(
        voices, "embed_segments",
        lambda a, segs: np.array([[1.0, 0.0, 0.0]] * len(segs),
                                 dtype=np.float32))
    assert voices.guided_split(conn, tid, str(audio))["split"] == {}
    assert conn.execute(
        "SELECT COUNT(*) FROM segments WHERE transcript_id=? AND "
        "speaker='SPEAKER_00'", (tid,)).fetchone()[0] == 12


def test_guided_split_ignores_isolated_evidence(env, monkeypatch):
    """A single matching segment is not sustained evidence."""
    client, conn, tmp_path = env
    rec_id, tid, audio = _two_voice_cluster(conn, tmp_path)
    _enroll(conn, "Alex Sisk", [1.0, 0.0, 0.0])
    monkeypatch.setattr(voices, "_load_audio", lambda p: np.zeros(16000 * 400))

    def one_match(audio_data, segments):
        out = []
        for i, _ in enumerate(segments):
            v = np.array([1.0, 0.0, 0.0] if i == 0 else [0.0, 1.0, 0.0],
                         dtype=np.float32)
            out.append(v)
        return np.array(out)
    monkeypatch.setattr(voices, "embed_segments", one_match)
    assert voices.guided_split(conn, tid, str(audio))["split"] == {}


def test_guided_split_never_touches_pinned_or_human_labels(env, monkeypatch):
    client, conn, tmp_path = env
    rec_id, tid, audio = _two_voice_cluster(conn, tmp_path)
    _enroll(conn, "Alex Sisk", [1.0, 0.0, 0.0])
    # Half the cluster is human work: pinned, and a human-set name.
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM segments WHERE transcript_id=? ORDER BY id",
        (tid,)).fetchall()]
    conn.execute("UPDATE segments SET pinned=1 WHERE id=?", (ids[0],))
    conn.execute("UPDATE segments SET speaker='Felix' WHERE id=?", (ids[1],))
    conn.commit()
    monkeypatch.setattr(voices, "_load_audio", lambda p: np.zeros(16000 * 400))
    monkeypatch.setattr(
        voices, "embed_segments",
        lambda a, segs: np.array([[1.0, 0.0, 0.0]] * len(segs),
                                 dtype=np.float32))
    voices.guided_split(conn, tid, str(audio))
    assert conn.execute(
        "SELECT speaker, pinned FROM segments WHERE id=?",
        (ids[0],)).fetchone() == ("SPEAKER_00", 1)
    assert conn.execute(
        "SELECT speaker FROM segments WHERE id=?",
        (ids[1],)).fetchone()[0] == "Felix"


def test_guided_split_respects_a_rejected_tag(env, monkeypatch):
    client, conn, tmp_path = env
    rec_id, tid, audio = _two_voice_cluster(conn, tmp_path)
    _enroll(conn, "Alex Sisk", [1.0, 0.0, 0.0])
    conn.execute(
        "INSERT INTO rejected_tags (transcript_id, label, name) "
        "VALUES (?, 'SPEAKER_00', 'Alex Sisk')", (tid,))
    conn.commit()
    monkeypatch.setattr(voices, "_load_audio", lambda p: np.zeros(16000 * 400))
    monkeypatch.setattr(
        voices, "embed_segments",
        lambda a, segs: np.array([[1.0, 0.0, 0.0]] * len(segs),
                                 dtype=np.float32))
    assert voices.guided_split(conn, tid, str(audio))["split"] == {}


# --- 3. Denoise --------------------------------------------------------

def test_denoise_failure_falls_back_to_the_original_audio(monkeypatch):
    from app import denoise
    monkeypatch.setattr(
        denoise, "_get_enhancer",
        lambda: (_ for _ in ()).throw(RuntimeError("no model")))
    assert denoise.enhance("whatever.m4a") is None


def test_diarize_fresh_uses_the_cleaned_copy_then_removes_it(
        tmp_path, monkeypatch):
    from app import denoise
    cleaned = tmp_path / "cleaned.wav"
    cleaned.write_bytes(b"clean")
    heard = {}
    monkeypatch.setattr(denoise, "enhance", lambda p: cleaned)
    import app.diarize as diarize_mod
    monkeypatch.setattr(diarize_mod, "diarize",
                        lambda p, num_speakers=None: heard.update(path=str(p))
                        or [])
    monkeypatch.setattr(diarize_mod, "assign_speakers", lambda segs, turns: None)

    reprocess._diarize_fresh(tmp_path / "orig.m4a", [], denoise=True)
    assert heard["path"] == str(cleaned)
    assert not cleaned.exists(), "the temporary cleaned copy is removed"


# --- 4. Suspicious-result detection ------------------------------------

def test_long_single_speaker_recording_is_flagged(env):
    _client, conn, tmp_path = env
    rec_id, tid, _ = seed(conn, tmp_path, duration=2520.0)
    assert db.refresh_diarization_suspect(conn, rec_id) is True
    assert conn.execute(
        "SELECT diarization_suspect FROM recordings WHERE id=?",
        (rec_id,)).fetchone()[0] == 1


def test_two_speakers_or_a_short_recording_are_not_flagged(env):
    _client, conn, tmp_path = env
    # Two speakers over the same length: normal.
    rec_id, tid, _ = seed(conn, tmp_path, duration=2520.0)
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM segments WHERE transcript_id=?", (tid,)).fetchall()]
    conn.execute("UPDATE segments SET speaker='SPEAKER_01' WHERE id=?",
                 (ids[0],))
    conn.commit()
    assert db.refresh_diarization_suspect(conn, rec_id) is False

    # A short solo memo is legitimately one speaker.
    audio2 = config.INBOX_DIR / "short.m4a"
    make_audio(audio2)
    rec2 = db.insert_recording(conn, str(audio2), "hash-short", 120.0)
    db.insert_transcript(conn, rec2, "hi", "en", "m", [
        {"start": 0.0, "end": 5.0, "text": "hi", "speaker": "SPEAKER_00"}])
    assert db.refresh_diarization_suspect(conn, rec2) is False


def test_suspect_flag_is_exposed_and_dismissible(env):
    client, conn, tmp_path = env
    rec_id, _, _ = seed(conn, tmp_path, duration=2520.0)
    db.refresh_diarization_suspect(conn, rec_id)
    assert client.get(f"/api/recordings/{rec_id}").json()[
        "diarization_suspect"] is True
    resp = client.post(
        f"/api/recordings/{rec_id}/diarization-suspect/dismiss")
    assert resp.status_code == 200
    assert client.get(f"/api/recordings/{rec_id}").json()[
        "diarization_suspect"] is False
    assert client.post(
        "/api/recordings/9999/diarization-suspect/dismiss").status_code == 404


def test_flagging_never_changes_labels(env):
    _client, conn, tmp_path = env
    rec_id, tid, _ = seed(conn, tmp_path, duration=2520.0)
    before = conn.execute(
        "SELECT id, speaker FROM segments WHERE transcript_id=? ORDER BY id",
        (tid,)).fetchall()
    db.refresh_diarization_suspect(conn, rec_id)
    after = conn.execute(
        "SELECT id, speaker FROM segments WHERE transcript_id=? ORDER BY id",
        (tid,)).fetchall()
    assert before == after


def test_auto_tag_cannot_undo_a_split_it_just_made(env, monkeypatch):
    """The gap that would recreate the original bug.

    A segment scoring just under the split threshold lands in the
    unmatched remainder, but still clears auto_tag's looser threshold.
    Without excluding the pairing, tagging would rename that remainder
    to the very person who was split out of it, putting two people back
    under one name.
    """
    client, conn, tmp_path = env
    rec_id, tid, audio = _two_voice_cluster(conn, tmp_path)
    _enroll(conn, "Alex Sisk", [1.0, 0.0, 0.0])
    monkeypatch.setattr(voices, "_load_audio", lambda p: np.zeros(16000 * 400))

    def borderline(audio_data, segments):
        # Even segments are clearly Alex; odd ones sit at ~0.52: below
        # the 0.55 split bar, above the 0.5 tagging bar.
        out = []
        for s in segments:
            if int(s["start"] // 30) % 2 == 0:
                v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                v = np.array([0.52, 0.854, 0.0], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.array(out)
    monkeypatch.setattr(voices, "embed_segments", borderline)

    split = voices.guided_split(conn, tid, str(audio))
    assert split["split"] == {"SPEAKER_00": {"Alex Sisk": 6}}
    # Exactly the pairing that must not be re-applied.
    assert voices.split_pairs(split) == {("SPEAKER_00", "Alex Sisk")}

    voices.auto_tag(conn, tid, str(audio),
                    exclude=voices.split_pairs(split))
    rows = dict(conn.execute(
        "SELECT speaker, COUNT(*) FROM segments WHERE transcript_id=? "
        "GROUP BY speaker", (tid,)).fetchall())
    # The remainder is still its own speaker, not folded back into Alex.
    assert rows == {"Alex Sisk": 6, "SPEAKER_00": 6}


def test_split_pairs_reads_an_empty_or_missing_result(env):
    assert voices.split_pairs({}) == set()
    assert voices.split_pairs({"split": {}}) == set()
    assert voices.split_pairs(None) == set()


def test_guided_split_needs_sustained_seconds_not_just_segments(
        env, monkeypatch):
    """Four matching segments are not enough if they are all brief."""
    client, conn, tmp_path = env
    # 12 segments of 2s each: clears the count floor, not the 20s floor.
    segs = [{"start": float(i * 30), "end": float(i * 30 + 2),
             "text": f"line {i}", "speaker": "SPEAKER_00"}
            for i in range(12)]
    rec_id, tid, audio = seed(conn, tmp_path, segments=segs)
    _enroll(conn, "Alex Sisk", [1.0, 0.0, 0.0])
    monkeypatch.setattr(voices, "_load_audio", lambda p: np.zeros(16000 * 400))

    def alternating(audio_data, segments):
        out = []
        for s in segments:
            near = int(s["start"] // 30) % 2 == 0
            v = np.array([1.0, 0.0, 0.0] if near else [0.0, 1.0, 0.0],
                         dtype=np.float32)
            out.append(v)
        return np.array(out)
    monkeypatch.setattr(voices, "embed_segments", alternating)
    assert voices.guided_split(conn, tid, str(audio))["split"] == {}


def test_watcher_takes_the_recorded_hint(env, monkeypatch):
    """The in-person count actually reaches diarization."""
    from app import watcher
    _client, conn, tmp_path = env
    audio = config.INBOX_DIR / "20260728 101010-inperson.m4a"
    make_audio(audio)
    hints.put(audio.name, 3)

    seen = {}
    monkeypatch.setattr(
        watcher.transcribe, "transcribe",
        lambda p, model=None: __import__("app.transcribe", fromlist=["x"])
        .TranscriptionResult(text="t", language="en", model="m",
                             segments=[{"start": 0.0, "end": 1.0,
                                        "text": "t"}],
                             duration_seconds=1.0))
    monkeypatch.setattr(
        watcher.diarize, "diarize",
        lambda p, num_speakers=None, **kw: seen.update(
            num_speakers=num_speakers) or [])
    monkeypatch.setattr(watcher.diarize, "assign_speakers",
                        lambda segs, turns: None)
    monkeypatch.setattr(watcher.voices, "guided_split",
                        lambda *a, **k: {"split": {}})
    monkeypatch.setattr(watcher.voices, "auto_tag", lambda *a, **k: {})
    monkeypatch.setattr(watcher.semantic, "index_transcript",
                        lambda *a, **k: 0)
    monkeypatch.setattr(watcher.calendar_sync, "match_recording",
                        lambda *a, **k: None)
    monkeypatch.setattr(watcher.summarize, "summarize",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("no llm in tests")))

    watcher.process_file(audio, conn)
    assert seen == {"num_speakers": 3}
    # Used once: a later file with the same stem gets nothing.
    assert hints.peek(audio.name) is None
