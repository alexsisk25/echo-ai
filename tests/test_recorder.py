"""Meeting recorder: lifecycle, mixing into inbox, permission errors."""

import os
import stat
import time

import pytest
from fastapi.testclient import TestClient

from app import config, db, main, recorder

GOOD_HELPER = """#!/usr/bin/env python3
import signal, struct, sys, time, wave
outdir = sys.argv[1]
for name in ("system.wav", "mic.wav"):
    w = wave.open(outdir + "/" + name, "w")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(struct.pack("<" + "h" * 16000, *([1000] * 16000)))
    w.close()
print("STARTED", flush=True)
stopped = []
signal.signal(signal.SIGINT, lambda *a: stopped.append(1))
while not stopped:
    time.sleep(0.05)
print("STOPPED", flush=True)
"""

DENIED_HELPER = """#!/usr/bin/env python3
print("ERROR microphone permission denied. Allow it in System Settings.",
      flush=True)
raise SystemExit(1)
"""

# Honors the mic-only argument like the real helper: writes system.wav
# only when capturing system audio, and records its argv for assertions.
MODE_AWARE_HELPER = """#!/usr/bin/env python3
import signal, struct, sys, time, wave
outdir = sys.argv[1]
open(outdir + "/args.txt", "w").write(" ".join(sys.argv[1:]))
names = ["mic.wav"] if "mic-only" in sys.argv else ["mic.wav", "system.wav"]
for name in names:
    w = wave.open(outdir + "/" + name, "w")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(struct.pack("<" + "h" * 16000, *([1000] * 16000)))
    w.close()
print("STARTED", flush=True)
stopped = []
signal.signal(signal.SIGINT, lambda *a: stopped.append(1))
while not stopped:
    time.sleep(0.05)
print("STOPPED", flush=True)
"""


def write_helper(tmp_path, body):
    script = tmp_path / "fake-helper"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "rec.db")
    monkeypatch.setattr(config, "INBOX_DIR", tmp_path / "inbox")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)
    # Fresh module state per test.
    recorder._proc = None
    recorder._workdir = None
    recorder._started_at = None
    recorder._state = "idle"
    recorder._error = None
    recorder._last_file = None
    recorder._mode = "online"
    yield TestClient(main.app), tmp_path
    if recorder._proc is not None and recorder._proc.poll() is None:
        recorder._proc.kill()


def test_record_start_stop_lands_mixed_file_in_inbox(env, monkeypatch):
    client, tmp_path = env
    helper = write_helper(tmp_path, GOOD_HELPER)
    monkeypatch.setattr(recorder, "ensure_helper", lambda: helper)

    resp = client.post("/api/record/start")
    assert resp.status_code == 200
    assert resp.json()["state"] in ("starting", "recording")
    assert wait_for(
        lambda: client.get("/api/record/status").json()["state"]
        == "recording")

    resp = client.post("/api/record/stop")
    assert resp.status_code == 200
    name = resp.json()["file"]
    # Voice-Memos-style name so recorded_at parses the start time.
    assert name.endswith("-meeting.m4a")
    assert db.VOICE_MEMO_TS.match(name)

    out = tmp_path / "inbox" / name
    assert out.exists() and out.stat().st_size > 0
    # It is a real m4a that ffmpeg can read back.
    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True)
    assert float(probe.stdout.strip()) > 0.5

    status = client.get("/api/record/status").json()
    assert status["state"] == "idle"
    assert status["last_file"] == name


def test_in_person_records_mic_only_as_plain_mono_file(env, monkeypatch):
    client, tmp_path = env
    helper = write_helper(tmp_path, MODE_AWARE_HELPER)
    monkeypatch.setattr(recorder, "ensure_helper", lambda: helper)

    resp = client.post("/api/record/start", json={"mode": "in_person"})
    assert resp.status_code == 200
    assert wait_for(
        lambda: client.get("/api/record/status").json()["state"]
        == "recording")
    status = client.get("/api/record/status").json()
    assert status["mode"] == "in_person"
    # The helper was told to leave system audio alone.
    workdir = recorder._workdir
    assert wait_for(lambda: (workdir / "args.txt").exists())
    assert "mic-only" in (workdir / "args.txt").read_text()
    assert not (workdir / "system.wav").exists()

    resp = client.post("/api/record/stop")
    assert resp.status_code == 200
    name = resp.json()["file"]
    # No "-meeting" channel-aware marker, but recorded_at still parses.
    assert name.endswith("-inperson.m4a")
    assert "-meeting" not in name
    assert db.VOICE_MEMO_TS.match(name)

    # A normal single-channel file for the standard mono pipeline.
    import subprocess
    out = tmp_path / "inbox" / name
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True)
    assert probe.stdout.strip() == "1"


def test_in_person_ignores_a_stray_system_track(env, monkeypatch):
    # Even if a system.wav somehow exists, in-person output is mic only.
    client, tmp_path = env
    helper = write_helper(tmp_path, GOOD_HELPER)  # writes both tracks
    monkeypatch.setattr(recorder, "ensure_helper", lambda: helper)
    client.post("/api/record/start", json={"mode": "in_person"})
    wait_for(lambda: recorder.status()["state"] == "recording")
    resp = client.post("/api/record/stop")
    name = resp.json()["file"]
    assert name.endswith("-inperson.m4a")
    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-of", "csv=p=0",
         str(tmp_path / "inbox" / name)],
        capture_output=True, text=True)
    assert probe.stdout.strip() == "1"


def test_online_mode_keeps_the_meeting_marker_and_two_channels(env,
                                                              monkeypatch):
    client, tmp_path = env
    helper = write_helper(tmp_path, MODE_AWARE_HELPER)
    monkeypatch.setattr(recorder, "ensure_helper", lambda: helper)
    client.post("/api/record/start", json={"mode": "online"})
    wait_for(lambda: recorder.status()["state"] == "recording")
    assert client.get("/api/record/status").json()["mode"] == "online"
    name = client.post("/api/record/stop").json()["file"]
    assert name.endswith("-meeting.m4a")
    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-of", "csv=p=0",
         str(tmp_path / "inbox" / name)],
        capture_output=True, text=True)
    assert probe.stdout.strip() == "2"


def test_unknown_capture_mode_is_rejected(env):
    client, _ = env
    resp = client.post("/api/record/start", json={"mode": "underwater"})
    assert resp.status_code == 422
    assert recorder.status()["state"] == "idle"


def test_double_start_is_a_no_op(env, monkeypatch):
    client, tmp_path = env
    helper = write_helper(tmp_path, GOOD_HELPER)
    monkeypatch.setattr(recorder, "ensure_helper", lambda: helper)
    client.post("/api/record/start")
    wait_for(lambda: recorder.status()["state"] == "recording")
    first_proc = recorder._proc
    client.post("/api/record/start")
    assert recorder._proc is first_proc
    client.post("/api/record/stop")


def test_stop_without_recording_is_409(env):
    client, _ = env
    assert client.post("/api/record/stop").status_code == 409


def test_permission_denied_surfaces_helper_guidance(env, monkeypatch):
    client, tmp_path = env
    helper = write_helper(tmp_path, DENIED_HELPER)
    monkeypatch.setattr(recorder, "ensure_helper", lambda: helper)

    client.post("/api/record/start")
    assert wait_for(
        lambda: client.get("/api/record/status").json()["state"] == "error")
    status = client.get("/api/record/status").json()
    assert "permission" in status["error"]
    # A stop in the error state is a 409, not a crash.
    assert client.post("/api/record/stop").status_code == 409
    # Retrying start clears the error and tries again.
    monkeypatch.setattr(recorder, "ensure_helper",
                        lambda: write_helper(tmp_path, GOOD_HELPER))
    client.post("/api/record/start")
    assert wait_for(
        lambda: client.get("/api/record/status").json()["state"]
        == "recording")
    client.post("/api/record/stop")
