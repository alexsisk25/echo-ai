"""Shared test isolation.

A real google_credentials.json in the project root must never change
test behavior: constructing the real Google provider starts an
interactive OAuth flow (a local server waiting for a browser redirect)
that hangs the whole suite. Every test sees a missing credentials file,
so the auto provider always resolves to the fake one.
"""

import pytest

from app import calendar_sync, config, sync


@pytest.fixture(autouse=True)
def no_real_google_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calendar_sync, "GOOGLE_CREDENTIALS",
        tmp_path / "google_credentials_missing.json",
    )
    # The token file too: google_connected() means "authorized", and a
    # real token on this machine must not leak into tests.
    monkeypatch.setattr(
        calendar_sync, "GOOGLE_TOKEN",
        tmp_path / "google_token_missing.json",
    )


@pytest.fixture(autouse=True)
def no_real_sync_ledger(monkeypatch, tmp_path):
    """The sync ledger is a real file in data/, and deleting a recording
    appends to it. A test that deletes a throwaway recording was writing
    its stem into the archive's own ledger (found live: entries like
    "standup" and "demo" sitting among the real memo stems). Every test
    gets its own ledger file instead."""
    monkeypatch.setattr(sync, "LEDGER", tmp_path / "test_ledger.txt")


@pytest.fixture(autouse=True)
def no_real_audio_folders(monkeypatch, tmp_path):
    """db.connect() scans the inbox and backlog to repoint recordings
    whose audio moved with the project folder. Left unpinned, a test that
    connects with seeded rows would reach Alex's real inbox, and today it
    is only the empty first database of the run that keeps that from
    happening, which makes the safety test-order dependent. Same class of
    leak as the sync ledger above."""
    inbox = tmp_path / "test_inbox"
    backlog = tmp_path / "test_backlog"
    inbox.mkdir(exist_ok=True)
    backlog.mkdir(exist_ok=True)
    monkeypatch.setattr(config, "INBOX_DIR", inbox)
    monkeypatch.setattr(config, "BACKLOG_DIR", backlog)
