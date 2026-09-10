import json

import numpy as np
import pytest

from app import commitments, db, llm, people, semantic, summarize, translate

SUMMARY = json.dumps({
    "title": "Budget Review",
    "date": "2026-07-01",
    "attendees": ["Sam"],
    "summary": "The team reviewed the budget.",
    "decisions": [],
    "action_items": [
        {"owner": "Sam", "task": "Update the forecast",
         "due_date": "2026-07-14", "priority": "high"},
        {"owner": "unassigned", "task": "Book the venue",
         "due_date": None, "priority": None},
    ],
    "risks": [],
    "topics": ["budget"],
})


@pytest.fixture
def conn(tmp_path):
    c = db.connect(db_path=tmp_path / "test.db", key="ab" * 32)
    yield c
    c.close()


def seed_meeting(conn, name="Sam", digest="h1", summary=SUMMARY):
    rec_id = db.insert_recording(conn, f"/tmp/{digest}.m4a", digest)
    tid = db.insert_transcript(
        conn, rec_id, "sam said he will update the forecast next week",
        "en", "tiny",
        [{"start": 0.0, "end": 3.0,
          "text": "I will update the forecast next week", "speaker": name},
         {"start": 3.0, "end": 6.0,
          "text": "and someone should book the venue", "speaker": name}],
    )
    if summary:
        db.upsert_summary(conn, tid, summary, "test-model")
    return rec_id, tid


# ---------- Job 1: dossiers ----------

def test_list_people_excludes_generic_labels(conn):
    seed_meeting(conn, name="Sam", digest="h1")
    rec2 = db.insert_recording(conn, "/tmp/h2.m4a", "h2")
    tid2 = db.insert_transcript(conn, rec2, "x", "en", "tiny",
                                [{"start": 0, "end": 1, "text": "hello",
                                  "speaker": "SPEAKER_00"}])
    out = people.list_people(conn)
    assert [p["name"] for p in out] == ["Sam"]
    assert out[0]["meetings"] == 1


def test_person_data_collects_their_action_items(conn):
    seed_meeting(conn)
    data = people.person_data(conn, "Sam")
    assert len(data["meetings"]) == 1
    assert [a["task"] for a in data["action_items"]] == ["Update the forecast"]
    assert data["topics"] == ["budget"]


def test_build_dossier_caches_and_stales(conn, monkeypatch):
    seed_meeting(conn)
    dossier_json = json.dumps({
        "name": "Sam", "summary": "Sam runs the budget.",
        "cares_about": ["budget"], "commitments_made": [], "follow_ups_owed": [],
    })
    calls = []
    monkeypatch.setattr(summarize, "_call_llm",
                        lambda p, privileged=False: calls.append(p) or dossier_json)
    d = people.build_dossier(conn, "Sam")
    assert d.summary == "Sam runs the budget."
    assert len(calls) == 1
    assert people.get_dossier(conn, "Sam")["summary"] == "Sam runs the budget."

    # A new meeting makes the cache stale.
    seed_meeting(conn, digest="h9")
    assert people.get_dossier(conn, "Sam") is None


# ---------- Job 2: commitments ----------

@pytest.fixture
def fake_embeddings(monkeypatch):
    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            out = []
            for t in texts:
                v = np.zeros(semantic.EMBEDDING_DIM, dtype=np.float32)
                v[0] = 1.0 if "forecast" in t.lower() else 0.1
                v[1] = 1.0 if "venue" in t.lower() else 0.1
                out.append(v / np.linalg.norm(v))
            return np.array(out)
    monkeypatch.setattr(semantic, "_model", FakeModel())


def test_extract_links_commitments_to_segments(conn, fake_embeddings):
    _, tid = seed_meeting(conn)
    assert commitments.extract(conn, tid) == 2
    rows = commitments.list_commitments(conn)
    by_task = {r["task"]: r for r in rows}
    assert by_task["Update the forecast"]["owner"] == "Sam"
    assert by_task["Update the forecast"]["segment_start"] == 0.0
    assert by_task["Book the venue"]["segment_start"] == 3.0


def test_extract_is_idempotent_and_keeps_status(conn, fake_embeddings):
    _, tid = seed_meeting(conn)
    commitments.extract(conn, tid)
    first = commitments.list_commitments(conn)[0]
    commitments.set_status(conn, first["id"], "done")
    assert commitments.extract(conn, tid) == 0
    again = [c for c in commitments.list_commitments(conn)
             if c["id"] == first["id"]][0]
    assert again["status"] == "done"


def test_status_filter_and_bad_status(conn, fake_embeddings):
    _, tid = seed_meeting(conn)
    commitments.extract(conn, tid)
    open_items = commitments.list_commitments(conn, status="open")
    assert len(open_items) == 2
    with pytest.raises(ValueError):
        commitments.set_status(conn, open_items[0]["id"], "maybe")


# ---------- Job 3: translation ----------

def test_translate_recording_text_and_summary(conn, monkeypatch):
    rec_id, tid = seed_meeting(conn)

    def fake_llm(prompt, privileged=False):
        if '"title"' in prompt:
            return SUMMARY.replace("Budget Review", "Revision del presupuesto")
        return "texto traducido"
    monkeypatch.setattr(summarize, "_call_llm", fake_llm)

    out = translate.translate_recording(conn, rec_id, "es")
    assert out["full_text"] == "texto traducido"
    assert out["summary"]["title"] == "Revision del presupuesto"

    # Cached: a second call must not hit the LLM again.
    monkeypatch.setattr(summarize, "_call_llm",
                        lambda p, privileged=False: 1 / 0)
    cached = translate.translate_recording(conn, rec_id, "es")
    assert cached["full_text"] == "texto traducido"


def test_translate_retries_malformed_notes_then_keeps_the_transcript(
        conn, monkeypatch):
    """Translation was the one LLM path with no cleanup pass: a model
    that answered the notes prompt with prose took the whole request
    down with an unhandled validation error. Now it retries once, and
    if that fails too the transcript translation still lands."""
    rec_id, tid = seed_meeting(conn)
    prompts = []

    def bad_notes(prompt, privileged=False):
        prompts.append(prompt)
        if '"title"' in prompt or "corrected JSON" in prompt:
            return "lo siento, aqui va la traduccion"   # never valid JSON
        return "texto traducido"
    monkeypatch.setattr(summarize, "_call_llm", bad_notes)

    out = translate.translate_recording(conn, rec_id, "es")
    assert out["full_text"] == "texto traducido"
    assert out["summary"] is None
    # One notes attempt plus exactly one cleanup pass.
    assert sum(1 for p in prompts if "corrected JSON" in p) == 1


def test_translate_recovers_notes_on_the_cleanup_pass(conn, monkeypatch):
    rec_id, tid = seed_meeting(conn)

    def messy_then_clean(prompt, privileged=False):
        if "corrected JSON" in prompt:
            return SUMMARY.replace("Budget Review", "Revision del presupuesto")
        if '"title"' in prompt:
            return "```\nnot json at all\n```"
        return "texto traducido"
    monkeypatch.setattr(summarize, "_call_llm", messy_then_clean)

    out = translate.translate_recording(conn, rec_id, "es")
    assert out["summary"]["title"] == "Revision del presupuesto"


def test_translate_rejects_unknown_language(conn):
    rec_id, _ = seed_meeting(conn)
    with pytest.raises(ValueError):
        translate.translate_recording(conn, rec_id, "fr")


def test_chunking_long_text():
    text = "palabra " * 2000
    chunks = translate._chunks(text)
    assert len(chunks) > 2
    assert "".join(c if i == 0 else " " + c.lstrip()
                   for i, c in enumerate(chunks)).split() == text.split()


# ---------- Job 4: privacy vault ----------

def test_privileged_flag_migration_and_helpers(conn):
    rec_id, tid = seed_meeting(conn)
    assert db.is_privileged(conn, rec_id) is False
    conn.execute("UPDATE recordings SET privileged = 1 WHERE id = ?", (rec_id,))
    conn.commit()
    assert db.is_privileged(conn, rec_id) is True
    assert db.transcript_privileged(conn, tid) is True


def test_privileged_never_calls_cloud(monkeypatch):
    """The vault guarantee: privileged prompts never reach litellm."""
    import litellm

    def cloud_forbidden(*a, **kw):
        raise AssertionError("cloud call attempted for privileged content")
    monkeypatch.setattr(litellm, "completion", cloud_forbidden)
    monkeypatch.setattr(llm, "_local_complete", lambda p, m: "local answer")

    assert llm.complete("secret prompt", privileged=True) == "local answer"


def test_privileged_fails_closed_without_local_model(monkeypatch):
    """No local model must mean failure, never a cloud fallback."""
    import litellm

    def cloud_forbidden(*a, **kw):
        raise AssertionError("cloud call attempted for privileged content")
    monkeypatch.setattr(litellm, "completion", cloud_forbidden)
    monkeypatch.setattr(llm, "_local", None)
    monkeypatch.setattr(
        llm, "_get_local",
        lambda: (_ for _ in ()).throw(llm.LocalModelUnavailable("pending")),
    )

    def local_raises(prompt, max_tokens):
        llm._get_local()
    monkeypatch.setattr(llm, "_local_complete", local_raises)
    with pytest.raises(llm.LocalModelUnavailable):
        llm.complete("secret prompt", privileged=True)


def test_ask_uses_local_when_hit_is_privileged(conn, fake_embeddings,
                                               monkeypatch):
    rec_id, tid = seed_meeting(conn)
    semantic.index_transcript(conn, tid, "we will update the forecast")
    conn.execute("UPDATE recordings SET privileged = 1 WHERE id = ?", (rec_id,))
    conn.commit()

    monkeypatch.setattr(llm, "_get_local", lambda: ("model", "tok"))
    seen = {}

    def fake_call(prompt, privileged=False):
        seen["privileged"] = privileged
        return "local synthesized answer"
    monkeypatch.setattr(summarize, "_call_llm", fake_call)

    result = semantic.ask(conn, "forecast plans")
    assert seen["privileged"] is True
    assert "local synthesized" in result["answer"]


def test_ask_excludes_privileged_when_local_unavailable(conn, fake_embeddings,
                                                        monkeypatch):
    rec_id, tid = seed_meeting(conn)
    semantic.index_transcript(conn, tid, "we will update the forecast")
    conn.execute("UPDATE recordings SET privileged = 1 WHERE id = ?", (rec_id,))
    conn.commit()

    monkeypatch.setattr(
        llm, "_get_local",
        lambda: (_ for _ in ()).throw(llm.LocalModelUnavailable("pending")),
    )
    monkeypatch.setattr(
        summarize, "_call_llm",
        lambda p, privileged=False: (_ for _ in ()).throw(
            AssertionError("no LLM call should happen with zero hits")),
    )
    result = semantic.ask(conn, "forecast plans")
    assert result["sources"] == []
    assert "excluded" in result["answer"]


# ---------- Due-date intelligence (PLAN Phase 5 "overdue alerts") ----------
# due_date was stored, sorted, and printed, but nothing ever compared it
# to today, so an overdue promise read exactly like one due next year.

def _commit(conn, tid, owner, task, due, status="open"):
    cur = conn.execute(
        "INSERT INTO commitments (transcript_id, owner, task, due_date, "
        "status) VALUES (?, ?, ?, ?, ?)", (tid, owner, task, due, status))
    conn.commit()
    return cur.lastrowid


def test_due_state_classifies_against_a_fixed_today():
    from datetime import date

    today = date(2026, 9, 9)
    assert commitments.due_state("2026-09-08", today=today) == ("overdue", -1)
    assert commitments.due_state("2026-09-09", today=today) == ("due-soon", 0)
    assert commitments.due_state("2026-09-16", today=today) == ("due-soon", 7)
    assert commitments.due_state("2026-09-17", today=today) == ("later", 8)
    assert commitments.due_state(None, today=today) == ("none", None)
    # Anything the model wrote that is not a date is "no date", not a crash.
    assert commitments.due_state("next Friday", today=today) == ("none", None)


def test_a_done_commitment_is_never_overdue():
    from datetime import date

    today = date(2026, 9, 9)
    state, days = commitments.due_state("2026-01-01", "done", today=today)
    assert state == "later"
    assert days == -251


def test_is_mine_follows_the_configured_user_name(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "USER_NAME", "Alex Sisk")
    assert commitments.is_mine("Alex Sisk")
    assert commitments.is_mine("alex sisk")
    # "Me" is what the capture pipeline pins the user's own channel to,
    # so it counts as the user whatever the configured name is.
    assert commitments.is_mine("Me")
    assert not commitments.is_mine("Felix Vivanco")
    assert not commitments.is_mine(None)
    assert not commitments.is_mine("")


def test_list_sorts_by_urgency_and_pushes_done_to_the_bottom(conn,
                                                             monkeypatch):
    from datetime import date

    from app import config
    monkeypatch.setattr(config, "USER_NAME", "Alex Sisk")
    _, tid = seed_meeting(conn, summary=None)
    later = _commit(conn, tid, "Sam", "later thing", "2026-12-01")
    undated = _commit(conn, tid, "Sam", "undated thing", None)
    soon = _commit(conn, tid, "Alex Sisk", "soon thing", "2026-09-11")
    very_overdue = _commit(conn, tid, "Sam", "very overdue", "2026-01-01")
    overdue = _commit(conn, tid, "Sam", "overdue", "2026-09-01")
    done = _commit(conn, tid, "Sam", "done thing", "2026-01-01", "done")

    rows = commitments.list_commitments(conn, today=date(2026, 9, 9))
    assert [r["id"] for r in rows] == [
        very_overdue, overdue, soon, later, undated, done]
    states = {r["id"]: r["due_state"] for r in rows}
    assert states[very_overdue] == "overdue"
    assert states[soon] == "due-soon"
    assert states[later] == "later"
    assert states[undated] == "none"
    assert states[done] == "later"


def test_due_and_mine_filters(conn, monkeypatch):
    from datetime import date

    from app import config
    monkeypatch.setattr(config, "USER_NAME", "Alex Sisk")
    _, tid = seed_meeting(conn, summary=None)
    _commit(conn, tid, "Sam", "overdue theirs", "2026-09-01")
    _commit(conn, tid, "Alex Sisk", "overdue mine", "2026-09-02")
    _commit(conn, tid, "Sam", "soon theirs", "2026-09-10")
    _commit(conn, tid, "Sam", "much later", "2026-12-01")
    today = date(2026, 9, 9)

    overdue = commitments.list_commitments(conn, due="overdue", today=today)
    assert {r["task"] for r in overdue} == {"overdue theirs", "overdue mine"}
    soon = commitments.list_commitments(conn, due="due-soon", today=today)
    assert {r["task"] for r in soon} == {"soon theirs"}

    mine = commitments.list_commitments(conn, mine=True, today=today)
    assert {r["task"] for r in mine} == {"overdue mine"}
    theirs = commitments.list_commitments(conn, mine=False, today=today)
    assert "overdue mine" not in {r["task"] for r in theirs}
    assert len(theirs) == 3

    both = commitments.list_commitments(conn, due="overdue", mine=False,
                                        today=today)
    assert {r["task"] for r in both} == {"overdue theirs"}


def test_commitments_endpoint_exposes_state_and_filters(tmp_path,
                                                        monkeypatch):
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app import main

    monkeypatch.setattr(dbmod.config, "DB_PATH", tmp_path / "c.db")
    monkeypatch.setattr(dbmod, "get_or_create_key", lambda: "ab" * 32)
    c = dbmod.connect()
    rec = dbmod.insert_recording(c, "/tmp/x.m4a", "hx")
    tid = dbmod.insert_transcript(c, rec, "x", "en", "tiny", [])
    _commit(c, tid, "Sam", "an old promise", "2020-01-01")
    _commit(c, tid, "Sam", "a distant promise", "2099-01-01")
    c.close()

    client = TestClient(main.app)
    rows = client.get("/api/commitments").json()
    assert rows[0]["task"] == "an old promise"
    assert rows[0]["due_state"] == "overdue"
    assert rows[0]["days_until"] < 0
    assert "mine" in rows[0]

    only = client.get("/api/commitments?due=overdue").json()
    assert [r["task"] for r in only] == ["an old promise"]
    assert client.get("/api/commitments?due=whenever").status_code == 422
