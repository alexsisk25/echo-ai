import subprocess

import pytest

from app import config, sync


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    source = tmp_path / "voicememos"
    inbox = tmp_path / "inbox"
    data = tmp_path / "data"
    source.mkdir()
    monkeypatch.setattr(config, "INBOX_DIR", inbox)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(sync, "LEDGER", data / "synced_memos.txt")
    return source, inbox


def make_qta(path, seconds=1, freq=440):
    """A small real .qta (QuickTime container, AAC audio), like the ones
    newer macOS Voice Memos writes."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         f"sine=frequency={freq}:duration={seconds}", "-c:a", "aac",
         "-f", "mov", str(path)],
        check=True, capture_output=True, text=True,
    )


def test_copies_new_memos(dirs):
    source, inbox = dirs
    (source / "a.m4a").write_bytes(b"AAAA")
    (source / "b.m4a").write_bytes(b"BBBB")

    assert sync.sync_voice_memos(source) == 2
    assert sorted(p.name for p in inbox.iterdir()) == ["a.m4a", "b.m4a"]


def test_second_run_copies_nothing(dirs):
    source, inbox = dirs
    (source / "a.m4a").write_bytes(b"AAAA")
    assert sync.sync_voice_memos(source) == 1
    assert sync.sync_voice_memos(source) == 0
    assert len(list(inbox.iterdir())) == 1


def test_new_file_after_first_run_is_picked_up(dirs):
    source, inbox = dirs
    (source / "a.m4a").write_bytes(b"AAAA")
    sync.sync_voice_memos(source)
    (source / "b.m4a").write_bytes(b"BBBB")
    assert sync.sync_voice_memos(source) == 1


def test_hidden_and_non_audio_files_ignored(dirs):
    source, inbox = dirs
    (source / ".partial.m4a").write_bytes(b"X")
    (source / "notes.txt").write_text("not audio")
    assert sync.sync_voice_memos(source) == 0
    assert list(inbox.iterdir()) == []


def test_qta_is_converted_to_m4a_keeping_base_name(dirs):
    source, inbox = dirs
    make_qta(source / "20260724 110715-74C2C559.qta")
    assert sync.sync_voice_memos(source) == 1
    # Lands as .m4a with the same base name, so recorded_at parses it.
    out = inbox / "20260724 110715-74C2C559.m4a"
    assert out.exists()
    assert [p.name for p in inbox.iterdir()] == \
        ["20260724 110715-74C2C559.m4a"]
    # It is a real, decodable audio file.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True)
    assert float(probe.stdout.strip()) > 0.5


def test_qta_ledger_keys_off_original_name_no_resync(dirs):
    source, _ = dirs
    make_qta(source / "20260724 110715-74C2C559.qta")
    assert sync.sync_voice_memos(source) == 1
    # The ledger remembers the memo by stem (no extension).
    ledger = (config.DATA_DIR / "synced_memos.txt").read_text().splitlines()
    assert "20260724 110715-74C2C559" in ledger
    assert "20260724 110715-74C2C559.qta" not in ledger
    # A second run does not re-convert it.
    assert sync.sync_voice_memos(source) == 0


def test_mixed_m4a_and_qta(dirs):
    source, inbox = dirs
    (source / "20260713 111210-A.m4a").write_bytes(b"AAAA")
    make_qta(source / "20260721 122422-B.qta")
    assert sync.sync_voice_memos(source) == 2
    assert sorted(p.name for p in inbox.iterdir()) == [
        "20260713 111210-A.m4a", "20260721 122422-B.m4a"]


def test_recorded_at_parses_the_converted_qta_name():
    from app import db
    # The base name survives conversion, so recorded_at reads the memo's
    # real start time from it.
    assert db.recorded_at_for("/inbox/20260724 110715-74C2C559.m4a") == \
        "2026-07-24 11:07:15"


def test_extension_change_does_not_resync(dirs):
    # The incident: a memo synced as .m4a, then Apple renames it to .qta.
    # Stem-based dedupe must treat it as the same memo, not re-import it.
    source, inbox = dirs
    (source / "20260126 110850-ABC.m4a").write_bytes(b"AAAA")
    assert sync.sync_voice_memos(source) == 1
    (source / "20260126 110850-ABC.m4a").unlink()
    make_qta(source / "20260126 110850-ABC.qta")
    assert sync.sync_voice_memos(source) == 0
    # Still just the one file that was synced first.
    assert [p.name for p in inbox.iterdir()] == ["20260126 110850-ABC.m4a"]


def test_deleted_memo_stays_deleted_across_extension_change(dirs):
    # A memo deleted as X.m4a must not come back when the folder later
    # holds it as X.qta (the exact bug that resurrected recordings).
    source, _ = dirs
    sync.add_to_ledger("20260126 110850-ABC.m4a")  # delete recorded it
    make_qta(source / "20260126 110850-ABC.qta")
    assert sync.sync_voice_memos(source) == 0


def test_ledger_migrates_legacy_extension_entries_to_stems(dirs):
    source, _ = dirs
    ledger = config.DATA_DIR / "synced_memos.txt"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    # A pre-fix ledger full of extensioned names.
    ledger.write_text("a.m4a\nb.qta\n20260126 110850-ABC.m4a\n")
    make_qta(source / "20260126 110850-ABC.qta")   # already synced as .m4a
    (source / "a.m4a").write_bytes(b"A")            # already synced
    (source / "c.m4a").write_bytes(b"C")            # genuinely new
    assert sync.sync_voice_memos(source) == 1       # only c
    # The ledger is now stems, deduped (newline-delimited; stems may
    # contain spaces, so split on lines, not whitespace).
    stems = set(ledger.read_text().splitlines())
    assert stems == {"a", "b", "20260126 110850-ABC", "c"}


def test_add_to_ledger_stores_stem():
    # Regardless of extension passed, the ledger holds the stem.
    from app import config as cfg
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path as P
        old = sync.LEDGER
        old_data = cfg.DATA_DIR
        try:
            cfg.DATA_DIR = P(d)
            sync.LEDGER = P(d) / "synced_memos.txt"
            sync.add_to_ledger("memo.qta")
            sync.add_to_ledger("memo.m4a")   # same stem, no duplicate
            assert sync.LEDGER.read_text().split() == ["memo"]
        finally:
            sync.LEDGER = old
            cfg.DATA_DIR = old_data


def test_unreadable_folder_raises_instead_of_reporting_nothing_new(dirs):
    """The live bug: a process without Full Disk Access saw an empty
    folder and reported "Nothing new" while the folder held 28 memos.

    Path.glob swallows PermissionError, so the read failure has to be
    detected explicitly or it is indistinguishable from an empty folder.
    """
    source, _inbox = dirs
    (source / "20260810 120000-ABC.qta").write_bytes(b"x")
    (source / "20260810 130000-DEF.m4a").write_bytes(b"y")
    source.chmod(0o000)
    try:
        # The mechanism behind the bug, asserted so a regression is loud.
        assert list(source.glob("*.qta")) == []
        with pytest.raises(PermissionError):
            sync.sync_voice_memos(source)
        with pytest.raises(sync.VoiceMemosUnreadable):
            sync.list_memo_files(source)
    finally:
        source.chmod(0o755)


def test_readable_but_empty_is_not_an_error(dirs):
    """The other half of the distinction: genuinely nothing new."""
    source, _inbox = dirs
    assert sync.sync_voice_memos(source) == 0
    assert sync.list_memo_files(source) == []


def test_a_readable_folder_still_syncs_after_the_permission_check(dirs):
    source, inbox = dirs
    (source / "a.m4a").write_bytes(b"AAAA")
    (source / ".hidden.m4a").write_bytes(b"H")
    (source / "notes.txt").write_text("not audio")
    assert sync.sync_voice_memos(source) == 1
    assert [p.name for p in inbox.iterdir()] == ["a.m4a"]


def test_a_missing_folder_is_reported_as_missing_not_denied(dirs, caplog):
    source, _inbox = dirs
    gone = source / "not-there"
    with caplog.at_level("WARNING"):
        assert sync.list_memo_files(gone) == []
    assert "does not exist" in caplog.text


def test_the_two_zero_cases_log_differently(dirs, caplog):
    """A reader of the log can tell which kind of zero this was."""
    source, _inbox = dirs
    with caplog.at_level("INFO"):
        sync.sync_voice_memos(source)
    assert "0 memo(s) present, 0 new" in caplog.text

    caplog.clear()
    source.chmod(0o000)
    try:
        with caplog.at_level("INFO"):
            with pytest.raises(PermissionError):
                sync.sync_voice_memos(source)
        # The success line must NOT appear for an unreadable folder.
        assert "memo(s) present" not in caplog.text
    finally:
        source.chmod(0o755)
