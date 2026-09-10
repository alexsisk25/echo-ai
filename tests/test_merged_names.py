"""Merged-name detection: two clusters living under one speaker name.

Modelled on recording 87, where a 61% cluster and a 35% cluster both
ended up as the user after a "Rename everywhere".
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, db, main, merged, voices


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "merged.db")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "INBOX_DIR", tmp_path / "inbox")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)
    (tmp_path / "inbox").mkdir()
    conn = db.connect()
    yield TestClient(main.app), conn, tmp_path
    conn.close()


def make_audio(path, seconds=1):
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         f"sine=frequency=440:duration={seconds}", "-c:a", "aac",
         "-f", "mov", str(path)],
        check=True, capture_output=True, text=True)


def seed(conn, tmp_path, segments, duration=2550.0):
    audio = config.INBOX_DIR / "20260721 122422-MERGE.m4a"
    make_audio(audio)
    rec_id = db.insert_recording(conn, str(audio), "hash-merge", duration)
    conn.execute("UPDATE recordings SET status='done' WHERE id=?", (rec_id,))
    tid = db.insert_transcript(conn, rec_id, "talk", "en", "m", segments)
    conn.commit()
    return rec_id, tid, audio


def recording_87_shape(name="Alex Sisk"):
    """A 61/35 split under one name: the tagged half records its origin,
    the renamed half has none, exactly like the real recording."""
    segs = []
    # Renamed-in half (no recorded origin), ~61% of speech.
    for i in range(20):
        segs.append({"start": float(i * 100), "end": float(i * 100 + 61),
                     "text": f"a{i}", "speaker": name})
    # Auto-tagged half (origin recorded), ~35%.
    for i in range(20):
        segs.append({"start": float(i * 100 + 61),
                     "end": float(i * 100 + 96),
                     "text": f"b{i}", "speaker": name})
    return segs


def set_origin(conn, tid, text_prefix, origin):
    conn.execute(
        "UPDATE segments SET auto_original = ? WHERE transcript_id = ? "
        "AND text LIKE ?", (origin, tid, text_prefix + "%"))
    conn.commit()


def fake_voices(monkeypatch, same_voice: bool):
    """Group 'a' vs group 'b' as either one person or two."""
    monkeypatch.setattr(voices, "_load_audio",
                        lambda p: np.zeros(16000 * 3000))

    def embed(audio, segments):
        out = []
        for s in segments:
            # Group b starts 61s into each 100s block.
            is_b = (s["start"] % 100) >= 61
            if same_voice or not is_b:
                v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            out.append(v)
        return np.array(out)
    monkeypatch.setattr(voices, "embed_segments", embed)


def test_detects_two_clusters_under_one_name(env, monkeypatch):
    _client, conn, tmp_path = env
    rec_id, tid, _ = seed(conn, tmp_path, recording_87_shape())
    set_origin(conn, tid, "b", "SPEAKER_02")
    fake_voices(monkeypatch, same_voice=False)

    finding = merged.detect(conn, rec_id)
    assert finding is not None
    assert finding["name"] == "Alex Sisk"
    assert finding["verified"] is True
    # Both sides are substantial, and the unrecorded half is named as such.
    assert set(finding["shares"]) == {"unrecorded origin", "SPEAKER_02"}
    assert all(v >= merged.MIN_SHARE for v in finding["shares"].values())


def test_a_confirmed_cluster_is_not_a_merge(env, monkeypatch):
    """The false positive the acoustic stage exists for.

    Confirming an auto-tagged line clears auto_original, so a heavily
    confirmed cluster looks structurally identical to a merge. It is
    one voice, so it must not be flagged.
    """
    _client, conn, tmp_path = env
    rec_id, tid, _ = seed(conn, tmp_path, recording_87_shape())
    set_origin(conn, tid, "b", "SPEAKER_02")
    fake_voices(monkeypatch, same_voice=True)

    assert merged.detect(conn, rec_id) is None
    # Structurally it did look like a candidate: without the audio
    # check the same data is flagged.
    unverified = merged.detect(conn, rec_id, verify_audio=False)
    assert unverified is not None and unverified["verified"] is None


def test_a_small_second_cluster_is_ignored(env, monkeypatch):
    _client, conn, tmp_path = env
    segs = []
    for i in range(20):
        segs.append({"start": float(i * 100), "end": float(i * 100 + 95),
                     "text": f"a{i}", "speaker": "Alex Sisk"})
    for i in range(20):
        segs.append({"start": float(i * 100 + 95),
                     "end": float(i * 100 + 99),
                     "text": f"b{i}", "speaker": "Alex Sisk"})
    rec_id, tid, _ = seed(conn, tmp_path, segs)
    set_origin(conn, tid, "b", "SPEAKER_02")
    fake_voices(monkeypatch, same_voice=False)
    # The second group is ~4% of speech: a fragment, not a person.
    assert merged.detect(conn, rec_id) is None


def test_generic_labels_are_not_a_merged_name(env, monkeypatch):
    """Two SPEAKER_XX clusters are normal diarization, not a merge."""
    _client, conn, tmp_path = env
    segs = []
    for i in range(20):
        segs.append({"start": float(i * 100), "end": float(i * 100 + 61),
                     "text": f"a{i}", "speaker": "SPEAKER_01"})
        segs.append({"start": float(i * 100 + 61),
                     "end": float(i * 100 + 96),
                     "text": f"b{i}", "speaker": "SPEAKER_02"})
    rec_id, tid, _ = seed(conn, tmp_path, segs)
    fake_voices(monkeypatch, same_voice=False)
    assert merged.detect(conn, rec_id) is None


def test_two_tagged_clusters_under_one_name_are_caught(env, monkeypatch):
    """Both halves carry a recorded origin: the clearest possible case."""
    _client, conn, tmp_path = env
    rec_id, tid, _ = seed(conn, tmp_path, recording_87_shape())
    set_origin(conn, tid, "a", "SPEAKER_01")
    set_origin(conn, tid, "b", "SPEAKER_02")
    fake_voices(monkeypatch, same_voice=False)
    finding = merged.detect(conn, rec_id)
    assert finding is not None
    assert set(finding["shares"]) == {"SPEAKER_01", "SPEAKER_02"}


def test_refresh_records_and_clears_the_flag(env, monkeypatch):
    _client, conn, tmp_path = env
    rec_id, tid, _ = seed(conn, tmp_path, recording_87_shape())
    set_origin(conn, tid, "b", "SPEAKER_02")
    fake_voices(monkeypatch, same_voice=False)

    assert merged.refresh(conn, rec_id) == "Alex Sisk"
    assert conn.execute(
        "SELECT merged_name_suspect FROM recordings WHERE id=?",
        (rec_id,)).fetchone()[0] == "Alex Sisk"

    # Once the two halves carry different names it is clean again, and
    # a previous dismissal is cleared so a future merge still surfaces.
    conn.execute("UPDATE recordings SET merged_name_dismissed=1 WHERE id=?",
                 (rec_id,))
    conn.execute(
        "UPDATE segments SET speaker='Someone Else' WHERE transcript_id=? "
        "AND text LIKE 'b%'", (tid,))
    conn.commit()
    assert merged.refresh(conn, rec_id) is None
    row = conn.execute(
        "SELECT merged_name_suspect, merged_name_dismissed FROM recordings "
        "WHERE id=?", (rec_id,)).fetchone()
    assert row == (None, 0)


def test_detection_never_changes_a_label(env, monkeypatch):
    _client, conn, tmp_path = env
    rec_id, tid, _ = seed(conn, tmp_path, recording_87_shape())
    set_origin(conn, tid, "b", "SPEAKER_02")
    fake_voices(monkeypatch, same_voice=False)
    before = conn.execute(
        "SELECT id, speaker, auto_original, pinned FROM segments "
        "WHERE transcript_id=? ORDER BY id", (tid,)).fetchall()
    merged.refresh(conn, rec_id)
    after = conn.execute(
        "SELECT id, speaker, auto_original, pinned FROM segments "
        "WHERE transcript_id=? ORDER BY id", (tid,)).fetchall()
    assert before == after


def test_flag_is_exposed_and_dismissible(env, monkeypatch):
    client, conn, tmp_path = env
    rec_id, tid, _ = seed(conn, tmp_path, recording_87_shape())
    set_origin(conn, tid, "b", "SPEAKER_02")
    fake_voices(monkeypatch, same_voice=False)
    merged.refresh(conn, rec_id)

    assert client.get(f"/api/recordings/{rec_id}").json()[
        "merged_name_suspect"] == "Alex Sisk"
    assert client.post(
        f"/api/recordings/{rec_id}/merged-name/dismiss").status_code == 200
    assert client.get(f"/api/recordings/{rec_id}").json()[
        "merged_name_suspect"] is None
    # The stored finding survives the dismissal; only the display stops.
    assert conn.execute(
        "SELECT merged_name_suspect FROM recordings WHERE id=?",
        (rec_id,)).fetchone()[0] == "Alex Sisk"
    assert client.post(
        "/api/recordings/9999/merged-name/dismiss").status_code == 404


def test_scan_archive_dry_run_touches_nothing(env, monkeypatch):
    _client, conn, tmp_path = env
    rec_id, tid, _ = seed(conn, tmp_path, recording_87_shape())
    set_origin(conn, tid, "b", "SPEAKER_02")
    fake_voices(monkeypatch, same_voice=False)

    found = merged.scan_archive(conn, dry_run=True)
    assert [f["recording_id"] for f in found] == [rec_id]
    assert found[0]["name"] == "Alex Sisk"
    # Nothing was written.
    assert conn.execute(
        "SELECT merged_name_suspect FROM recordings WHERE id=?",
        (rec_id,)).fetchone()[0] is None

    # A real run records it.
    merged.scan_archive(conn, dry_run=False)
    assert conn.execute(
        "SELECT merged_name_suspect FROM recordings WHERE id=?",
        (rec_id,)).fetchone()[0] == "Alex Sisk"


def test_a_new_name_clears_a_stale_dismissal(env, monkeypatch):
    """Dismissing one merge must not silence a different one later."""
    _client, conn, tmp_path = env
    rec_id, tid, _ = seed(conn, tmp_path, recording_87_shape())
    set_origin(conn, tid, "b", "SPEAKER_02")
    fake_voices(monkeypatch, same_voice=False)

    assert merged.refresh(conn, rec_id) == "Alex Sisk"
    conn.execute("UPDATE recordings SET merged_name_dismissed=1 WHERE id=?",
                 (rec_id,))
    conn.commit()
    # Re-running with the SAME finding leaves the dismissal alone.
    merged.refresh(conn, rec_id)
    assert conn.execute(
        "SELECT merged_name_dismissed FROM recordings WHERE id=?",
        (rec_id,)).fetchone()[0] == 1

    # A merge under a different name is a new finding, so it shows.
    conn.execute("UPDATE segments SET speaker='Someone Else' "
                 "WHERE transcript_id=?", (tid,))
    conn.commit()
    assert merged.refresh(conn, rec_id) == "Someone Else"
    row = conn.execute(
        "SELECT merged_name_suspect, merged_name_dismissed FROM recordings "
        "WHERE id=?", (rec_id,)).fetchone()
    assert row == ("Someone Else", 0)


def test_the_most_substantial_merge_is_the_one_reported(env, monkeypatch):
    """Two merged names in one recording: the bigger one is reported,
    deterministically rather than by dict order."""
    _client, conn, tmp_path = env
    segs = []
    # Name A: a 30/25 split. Name B: a 24/16 split, both over the floor.
    for i in range(10):
        base = float(i * 100)
        segs.append({"start": base, "end": base + 30, "text": f"a{i}",
                     "speaker": "Big Merge"})
        segs.append({"start": base + 30, "end": base + 55, "text": f"b{i}",
                     "speaker": "Big Merge"})
    for i in range(10):
        base = float(1000 + i * 100)
        segs.append({"start": base, "end": base + 24, "text": f"c{i}",
                     "speaker": "Small Merge"})
        segs.append({"start": base + 24, "end": base + 40, "text": f"d{i}",
                     "speaker": "Small Merge"})
    rec_id, tid, _ = seed(conn, tmp_path, segs)
    set_origin(conn, tid, "b", "SPEAKER_02")
    set_origin(conn, tid, "d", "SPEAKER_04")
    # No audio check here: this is about candidate ordering.
    finding = merged.detect(conn, rec_id, verify_audio=False)
    assert finding["name"] == "Big Merge"


def test_a_real_scan_detects_once_per_recording(env, monkeypatch):
    """scan_archive must not pay for the audio work twice."""
    _client, conn, tmp_path = env
    rec_id, tid, _ = seed(conn, tmp_path, recording_87_shape())
    set_origin(conn, tid, "b", "SPEAKER_02")
    fake_voices(monkeypatch, same_voice=False)

    calls = []
    real_detect = merged.detect
    monkeypatch.setattr(
        merged, "detect",
        lambda conn_, rid, **kw: calls.append(rid) or real_detect(
            conn_, rid, **kw))
    merged.scan_archive(conn, dry_run=False)
    assert calls == [rec_id], "detect ran more than once per recording"
