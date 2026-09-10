import numpy as np
import pytest

from app import db, voices


def unit(v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def direction(i, dim=192):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


@pytest.fixture
def conn(tmp_path):
    c = db.connect(db_path=tmp_path / "test.db", key="ab" * 32)
    yield c
    c.close()


def test_match_requires_two_matching_segments():
    prints = {"Sam": direction(0)}
    one_segment = np.array([direction(0)])
    two_segments = np.array([direction(0), direction(0)])
    assert voices.match_speaker(one_segment, prints) is None
    assert voices.match_speaker(two_segments, prints) == "Sam"


def test_match_below_threshold_is_none():
    prints = {"Sam": direction(0)}
    # ~0.32 similarity to the print, well under 0.5
    weak = unit([1.0, 3.0] + [0.0] * 190)
    segments = np.array([weak, weak, weak])
    assert voices.match_speaker(segments, prints) is None


def test_match_picks_best_of_multiple_names():
    prints = {"Sam": direction(0), "Ana": direction(1)}
    near_ana = unit([0.2, 1.0] + [0.0] * 190)
    segments = np.array([near_ana, near_ana])
    assert voices.match_speaker(segments, prints) == "Ana"


def test_match_with_no_voiceprints():
    assert voices.match_speaker(np.array([direction(0)]), {}) is None


def test_enroll_creates_and_averages(conn):
    voices.enroll(conn, "Sam", np.array([direction(0)]))
    first = voices.load_voiceprints(conn)["Sam"]
    assert np.allclose(first, direction(0))

    # A second enrollment from a different direction shifts the average.
    voices.enroll(conn, "Sam", np.array([direction(1)]))
    merged = voices.load_voiceprints(conn)["Sam"]
    assert 0.6 < float(merged @ direction(0)) < 0.8
    assert 0.6 < float(merged @ direction(1)) < 0.8
    n = conn.execute(
        "SELECT num_samples FROM voiceprints WHERE name = 'Sam'"
    ).fetchone()[0]
    assert n == 2


def test_pick_segments_prefers_long_ones():
    segments = [
        {"start": 0.0, "end": 0.5},    # too short
        {"start": 1.0, "end": 4.0},
        {"start": 5.0, "end": 6.5},
    ]
    picked = voices._pick_segments(segments)
    assert picked[0] == {"start": 1.0, "end": 4.0}
    assert len(picked) == 2


def test_auto_tag_renames_matching_speaker(conn, monkeypatch):
    rec_id = db.insert_recording(conn, "/tmp/a.m4a", "h1")
    tid = db.insert_transcript(
        conn, rec_id, "text", "en", "tiny",
        [{"start": 0.0, "end": 3.0, "text": "a", "speaker": "SPEAKER_00"},
         {"start": 3.0, "end": 6.0, "text": "b", "speaker": "SPEAKER_00"},
         {"start": 6.0, "end": 9.0, "text": "c", "speaker": "SPEAKER_01"}],
    )
    voices.enroll(conn, "Sam", np.array([direction(0)]))

    monkeypatch.setattr(voices, "_load_audio",
                        lambda path: np.zeros(16000 * 10, dtype=np.float32))

    def fake_embed(audio, segments):
        # SPEAKER_00's segments sound like Sam; SPEAKER_01's do not.
        if segments[0]["start"] == 0.0:
            return np.array([direction(0)] * len(segments))
        return np.array([direction(5)] * len(segments))
    monkeypatch.setattr(voices, "embed_segments", fake_embed)

    tagged = voices.auto_tag(conn, tid, "/tmp/a.m4a")
    assert tagged == {"SPEAKER_00": "Sam"}
    labels = [r[0] for r in conn.execute(
        "SELECT DISTINCT speaker FROM segments WHERE transcript_id = ? "
        "ORDER BY speaker", (tid,),
    ).fetchall()]
    assert labels == ["SPEAKER_01", "Sam"]


def test_auto_tag_skips_named_speakers(conn, monkeypatch):
    rec_id = db.insert_recording(conn, "/tmp/b.m4a", "h2")
    tid = db.insert_transcript(
        conn, rec_id, "text", "en", "tiny",
        [{"start": 0.0, "end": 3.0, "text": "a", "speaker": "Ana"},
         {"start": 3.0, "end": 6.0, "text": "b", "speaker": "Ana"}],
    )
    voices.enroll(conn, "Sam", np.array([direction(0)]))
    called = []
    monkeypatch.setattr(voices, "_load_audio",
                        lambda path: called.append(path) or np.zeros(10))

    assert voices.auto_tag(conn, tid, "/tmp/b.m4a") == {}
    assert called == []  # named speakers mean no audio work at all


def test_auto_tag_without_voiceprints_is_noop(conn, monkeypatch):
    rec_id = db.insert_recording(conn, "/tmp/c.m4a", "h3")
    tid = db.insert_transcript(
        conn, rec_id, "text", "en", "tiny",
        [{"start": 0.0, "end": 3.0, "text": "a", "speaker": "SPEAKER_00"}],
    )
    monkeypatch.setattr(voices, "_load_audio",
                        lambda path: (_ for _ in ()).throw(AssertionError))
    assert voices.auto_tag(conn, tid, "/tmp/c.m4a") == {}


def test_auto_tag_records_provenance(conn, monkeypatch):
    rec_id = db.insert_recording(conn, "/tmp/d.m4a", "h4")
    tid = db.insert_transcript(
        conn, rec_id, "text", "en", "tiny",
        [{"start": 0.0, "end": 3.0, "text": "a", "speaker": "SPEAKER_00"},
         {"start": 3.0, "end": 6.0, "text": "b", "speaker": "SPEAKER_00"}],
    )
    voices.enroll(conn, "Sam", np.array([direction(0)]))
    monkeypatch.setattr(voices, "_load_audio",
                        lambda path: np.zeros(16000 * 10, dtype=np.float32))
    monkeypatch.setattr(voices, "embed_segments",
                        lambda audio, segs: np.array([direction(0)] * len(segs)))

    assert voices.auto_tag(conn, tid, "/tmp/d.m4a") == {"SPEAKER_00": "Sam"}
    rows = conn.execute(
        "SELECT speaker, auto_original FROM segments WHERE transcript_id = ?",
        (tid,),
    ).fetchall()
    assert rows == [("Sam", "SPEAKER_00"), ("Sam", "SPEAKER_00")]


def _seed_recording(conn, tmp_path, name, speaker="SPEAKER_00", done=True):
    audio = tmp_path / f"{name}.m4a"
    audio.write_bytes(b"fake")
    rec_id = db.insert_recording(conn, str(audio), f"hash-{name}")
    if done:
        db.set_recording_status(conn, rec_id, "done")
    tid = db.insert_transcript(
        conn, rec_id, "text", "en", "tiny",
        [{"start": 0.0, "end": 3.0, "text": "a", "speaker": speaker},
         {"start": 3.0, "end": 6.0, "text": "b", "speaker": speaker}],
    )
    return rec_id, tid


def test_retag_archive_sweeps_matching_recordings(conn, tmp_path, monkeypatch):
    _, tid1 = _seed_recording(conn, tmp_path, "one")
    _, tid2 = _seed_recording(conn, tmp_path, "two")
    _, tid3 = _seed_recording(conn, tmp_path, "named", speaker="Ana")
    voices.enroll(conn, "Sam", np.array([direction(0)]))
    conn.commit()

    monkeypatch.setattr(voices, "_load_audio",
                        lambda path: np.zeros(16000 * 10, dtype=np.float32))
    monkeypatch.setattr(voices, "embed_segments",
                        lambda audio, segs: np.array([direction(0)] * len(segs)))

    results = voices.retag_archive("Sam", db_path=tmp_path / "test.db",
                                   key="ab" * 32)
    assert results == {tid1: {"SPEAKER_00": "Sam"},
                       tid2: {"SPEAKER_00": "Sam"}}
    for tid in (tid1, tid2):
        speakers = {r[0] for r in conn.execute(
            "SELECT speaker FROM segments WHERE transcript_id = ?", (tid,),
        ).fetchall()}
        assert speakers == {"Sam"}
    # The already named recording was untouched.
    assert conn.execute(
        "SELECT DISTINCT speaker FROM segments WHERE transcript_id = ?",
        (tid3,),
    ).fetchall() == [("Ana",)]


def test_retag_archive_only_matches_the_given_voice(conn, tmp_path,
                                                    monkeypatch):
    _, tid = _seed_recording(conn, tmp_path, "one")
    voices.enroll(conn, "Sam", np.array([direction(0)]))
    voices.enroll(conn, "Ana", np.array([direction(1)]))
    monkeypatch.setattr(voices, "_load_audio",
                        lambda path: np.zeros(16000 * 10, dtype=np.float32))
    # The voice in the recording is Sam's, but the sweep is for Ana.
    monkeypatch.setattr(voices, "embed_segments",
                        lambda audio, segs: np.array([direction(0)] * len(segs)))
    results = voices.retag_archive("Ana", db_path=tmp_path / "test.db",
                                   key="ab" * 32)
    assert results == {}


def test_confirm_tag_promotes_to_human(conn, tmp_path, monkeypatch):
    _, tid = _seed_recording(conn, tmp_path, "one")
    voices.enroll(conn, "Sam", np.array([direction(0)]))
    monkeypatch.setattr(voices, "_load_audio",
                        lambda path: np.zeros(16000 * 10, dtype=np.float32))
    monkeypatch.setattr(voices, "embed_segments",
                        lambda audio, segs: np.array([direction(0)] * len(segs)))
    voices.auto_tag(conn, tid, "x")

    assert voices.confirm_tag(conn, tid, "Sam") == 2
    rows = conn.execute(
        "SELECT speaker, auto_original FROM segments WHERE transcript_id = ?",
        (tid,),
    ).fetchall()
    assert rows == [("Sam", None), ("Sam", None)]
    # Confirmed tags are human-set now: deleting the voice keeps them.
    assert voices.delete_voice(conn, "Sam") == 0
    assert conn.execute(
        "SELECT DISTINCT speaker FROM segments WHERE transcript_id = ?",
        (tid,),
    ).fetchall() == [("Sam",)]


def test_reject_tag_reverts_and_blocks_retagging(conn, tmp_path, monkeypatch):
    _, tid = _seed_recording(conn, tmp_path, "one")
    voices.enroll(conn, "Sam", np.array([direction(0)]))
    monkeypatch.setattr(voices, "_load_audio",
                        lambda path: np.zeros(16000 * 10, dtype=np.float32))
    monkeypatch.setattr(voices, "embed_segments",
                        lambda audio, segs: np.array([direction(0)] * len(segs)))
    voices.auto_tag(conn, tid, "x")

    assert voices.reject_tag(conn, tid, "Sam") == 2
    assert conn.execute(
        "SELECT DISTINCT speaker FROM segments WHERE transcript_id = ?",
        (tid,),
    ).fetchall() == [("SPEAKER_00",)]
    # The rejection is remembered: the same voice never re-tags this speaker.
    assert voices.auto_tag(conn, tid, "x") == {}


def test_delete_voice_reverts_only_machine_tags(conn, tmp_path, monkeypatch):
    _, tid_auto = _seed_recording(conn, tmp_path, "one")
    _, tid_human = _seed_recording(conn, tmp_path, "two")
    voices.enroll(conn, "Sam", np.array([direction(0)]))
    monkeypatch.setattr(voices, "_load_audio",
                        lambda path: np.zeros(16000 * 10, dtype=np.float32))
    monkeypatch.setattr(voices, "embed_segments",
                        lambda audio, segs: np.array([direction(0)] * len(segs)))
    voices.auto_tag(conn, tid_auto, "x")
    # A human renamed the speaker in the second recording by hand.
    db.rename_speaker(conn, tid_human, "SPEAKER_00", "Sam")

    assert voices.delete_voice(conn, "Sam") == 2
    assert voices.load_voiceprints(conn) == {}
    assert conn.execute(
        "SELECT DISTINCT speaker FROM segments WHERE transcript_id = ?",
        (tid_auto,),
    ).fetchall() == [("SPEAKER_00",)]
    assert conn.execute(
        "SELECT DISTINCT speaker FROM segments WHERE transcript_id = ?",
        (tid_human,),
    ).fetchall() == [("Sam",)]


def test_delete_missing_voice_returns_none(conn):
    assert voices.delete_voice(conn, "Nobody") is None


def test_rename_voice_renames_print_and_segments(conn, tmp_path, monkeypatch):
    _, tid = _seed_recording(conn, tmp_path, "one")
    voices.enroll(conn, "Sam", np.array([direction(0)]))
    monkeypatch.setattr(voices, "_load_audio",
                        lambda path: np.zeros(16000 * 10, dtype=np.float32))
    monkeypatch.setattr(voices, "embed_segments",
                        lambda audio, segs: np.array([direction(0)] * len(segs)))
    voices.auto_tag(conn, tid, "x")

    assert voices.rename_voice(conn, "Sam", "Samuel") == 2
    assert set(voices.load_voiceprints(conn)) == {"Samuel"}
    assert conn.execute(
        "SELECT DISTINCT speaker FROM segments WHERE transcript_id = ?",
        (tid,),
    ).fetchall() == [("Samuel",)]
    # Provenance survives the rename: deleting still reverts.
    assert voices.delete_voice(conn, "Samuel") == 2


def test_rename_voice_to_existing_name_raises(conn):
    voices.enroll(conn, "Sam", np.array([direction(0)]))
    voices.enroll(conn, "Ana", np.array([direction(1)]))
    with pytest.raises(ValueError):
        voices.rename_voice(conn, "Sam", "Ana")
    assert voices.rename_voice(conn, "Nobody", "X") is None


def test_list_voices_counts(conn, tmp_path, monkeypatch):
    _, tid = _seed_recording(conn, tmp_path, "one")
    voices.enroll(conn, "Sam", np.array([direction(0), direction(0)]))
    monkeypatch.setattr(voices, "_load_audio",
                        lambda path: np.zeros(16000 * 10, dtype=np.float32))
    monkeypatch.setattr(voices, "embed_segments",
                        lambda audio, segs: np.array([direction(0)] * len(segs)))
    voices.auto_tag(conn, tid, "x")

    listed = voices.list_voices(conn)
    assert len(listed) == 1
    v = listed[0]
    assert v["name"] == "Sam"
    assert v["num_samples"] == 2
    assert v["recordings"] == 1
    assert v["segments"] == 2
    assert v["auto_segments"] == 2
    voices.confirm_tag(conn, tid, "Sam")
    assert voices.list_voices(conn)[0]["auto_segments"] == 0
