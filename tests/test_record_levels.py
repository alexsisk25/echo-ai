"""Recording level meters and the dead-source silence warning.

These drive the level-tracking logic directly (no subprocess), so they
stay fast and deterministic.
"""

from datetime import datetime, timedelta

import pytest

from app import recorder


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(recorder, "_state", "recording")
    monkeypatch.setattr(recorder, "_mic_level", 0.0)
    monkeypatch.setattr(recorder, "_system_level", 0.0)
    now = datetime.now()
    monkeypatch.setattr(recorder, "_mic_last_sound", now)
    monkeypatch.setattr(recorder, "_system_last_sound", now)
    yield


def test_level_line_updates_both_meters():
    recorder._parse_level("LEVEL mic=0.1234 system=0.0000")
    lv = recorder.levels()
    assert lv["mic"] == 0.1234
    assert lv["system"] == 0.0
    assert lv["recording"] is True
    # Loud mic, silent system: no warning yet (within grace period).
    assert lv["warnings"] == []


def test_garbage_level_line_is_ignored():
    recorder._parse_level("LEVEL mic=oops")
    assert recorder.levels()["mic"] == 0.0


def test_silent_system_for_10s_warns_about_meeting_audio(monkeypatch):
    # Mic heard just now, system last heard 11s ago.
    now = datetime.now()
    monkeypatch.setattr(recorder, "_mic_last_sound", now)
    monkeypatch.setattr(recorder, "_system_last_sound",
                        now - timedelta(seconds=11))
    warnings = recorder.levels()["warnings"]
    assert len(warnings) == 1
    assert warnings[0]["source"] == "system"
    assert "meeting audio" in warnings[0]["message"].lower()


def test_silent_mic_for_10s_warns_about_microphone(monkeypatch):
    now = datetime.now()
    monkeypatch.setattr(recorder, "_mic_last_sound",
                        now - timedelta(seconds=12))
    monkeypatch.setattr(recorder, "_system_last_sound", now)
    warnings = recorder.levels()["warnings"]
    assert [w["source"] for w in warnings] == ["mic"]
    assert "microphone" in warnings[0]["message"].lower()


def test_no_warnings_when_not_recording(monkeypatch):
    monkeypatch.setattr(recorder, "_state", "idle")
    monkeypatch.setattr(recorder, "_mic_last_sound",
                        datetime.now() - timedelta(seconds=60))
    monkeypatch.setattr(recorder, "_system_last_sound",
                        datetime.now() - timedelta(seconds=60))
    assert recorder.levels()["warnings"] == []
    assert recorder.levels()["recording"] is False


def test_sound_above_floor_resets_the_silence_clock():
    # A loud reading marks the source as heard now, clearing a warning.
    recorder._parse_level("LEVEL mic=0.5 system=0.5")
    lv = recorder.levels()
    assert lv["warnings"] == []
    assert lv["mic"] == 0.5 and lv["system"] == 0.5


def test_mic_only_level_line_leaves_system_untouched(monkeypatch):
    # In-person capture: the helper emits no system field, so the system
    # meter and its last-heard clock must not move.
    monkeypatch.setattr(recorder, "_system_last_sound", None)
    recorder._parse_level("LEVEL mic=0.4")
    lv = recorder.levels()
    assert lv["mic"] == 0.4
    assert lv["system"] == 0.0
    assert recorder._system_last_sound is None


def test_in_person_never_warns_about_meeting_audio(monkeypatch):
    # With no system clock at all (in-person start), even a long silence
    # produces no meeting-audio warning; the mic warning still fires.
    now = datetime.now()
    monkeypatch.setattr(recorder, "_system_last_sound", None)
    monkeypatch.setattr(recorder, "_mic_last_sound", now)
    assert recorder.levels()["warnings"] == []
    monkeypatch.setattr(recorder, "_mic_last_sound",
                        now - timedelta(seconds=12))
    warnings = recorder.levels()["warnings"]
    assert [w["source"] for w in warnings] == ["mic"]
