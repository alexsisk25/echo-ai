"""Folder digests: hierarchical assembly, incremental refresh, the
folder-scoped ask, and the privileged-folder routing (with a cloud
tripwire, like the vault tests)."""

import json
import sys
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import (config, db, digest, llm, main, manage, organize, semantic,
                 summarize)

QUOTE = "we should revisit pricing before the renewal window closes"

DIGEST_JSON = json.dumps({
    "overview": "Pricing came up in January and kept coming back.",
    "themes": [{"theme": "pricing", "first_appeared": "2026-01-05",
                "framing_then": "a cost problem",
                "framing_now": "a packaging problem",
                "how_it_changed": "reframed in February"}],
    "shifts": [{"date": "2026-02-02", "shift": "moved to annual billing",
                "note": "followed the January discussion"}],
    "open_threads": [{"thread": "who owns the renewal",
                      "raised_on": "2026-01-05",
                      "last_mentioned": "2026-02-02", "note": "never named"}],
    "commitments": [{"owner": "Sam", "commitment": "send the forecast",
                     "made_on": "2026-01-05", "status": "open",
                     "note": "not mentioned since"}],
    "quotes": [{"quote": QUOTE, "speaker": "Sam", "recording_id": 1,
                "timestamp": "09:09:09", "why": "sets up the theme"}],
})

BRIEF = "Pricing recurred through the period.\nQuotable lines:\n"


def summary_for(title: str, date: str) -> str:
    return json.dumps({
        "title": title, "date": date, "attendees": ["Sam"],
        "summary": f"{title} covered pricing and the renewal.",
        "decisions": [f"{title} decision"], "action_items": [
            {"owner": "Sam", "task": "send the forecast",
             "due_date": None, "priority": None}],
        "risks": [], "topics": ["pricing"],
    })


def seed(conn, *, date, title, folder_id=None, summary=True,
         quote=QUOTE, extra_text="", segments=True):
    """One processed recording, optionally filed and summarized."""
    key = f"{title}-{date}"
    rec = db.insert_recording(conn, f"/tmp/{key}.m4a", key, 600.0)
    conn.execute("UPDATE recordings SET recorded_at = ? WHERE id = ?",
                 (f"{date} 09:00:00", rec))
    if folder_id is not None:
        conn.execute("UPDATE recordings SET folder_id = ? WHERE id = ?",
                     (folder_id, rec))
    conn.commit()
    db.set_recording_status(conn, rec, "done")
    segs = ([{"start": 3.0, "end": 9.0, "text": quote, "speaker": "Sam"}]
            if segments else [])
    tid = db.insert_transcript(
        conn, rec, f"{quote} {extra_text}".strip(), "en", "tiny", segs)
    if summary:
        db.upsert_summary(conn, tid, summary_for(title, date), model="test")
    return rec, tid


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "digest.db")
    monkeypatch.setattr(db, "get_or_create_key", lambda: "ab" * 32)
    conn = db.connect()
    folder = organize.create_folder(conn, "Client calls")["id"]
    yield conn, folder
    conn.close()


@pytest.fixture
def llm_calls(monkeypatch):
    """Capture every prompt and answer with a valid digest."""
    calls = []

    def fake(prompt, privileged=False):
        calls.append({"prompt": prompt, "privileged": privileged})
        return BRIEF if "condensing part of" in prompt else DIGEST_JSON
    monkeypatch.setattr(summarize, "_call_llm", fake)
    return calls


@pytest.fixture
def fake_embeddings(monkeypatch):
    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            out = []
            for t in texts:
                v = np.zeros(semantic.EMBEDDING_DIM, dtype=np.float32)
                v[0] = 1.0 if "pricing" in t.lower() else 0.1
                out.append(v / np.linalg.norm(v))
            return np.array(out)
    monkeypatch.setattr(semantic, "_model", FakeModel())


# ---------- hierarchical assembly ----------

def test_material_is_summaries_in_chronological_order(env):
    conn, folder = env
    seed(conn, date="2026-02-02", title="February", folder_id=folder,
         extra_text="RAW TRANSCRIPT ONLY SENTENCE")
    seed(conn, date="2026-01-05", title="January", folder_id=folder,
         extra_text="RAW TRANSCRIPT ONLY SENTENCE")

    recs = digest.folder_recordings(conn, folder)
    assert [r["date"] for r in recs] == ["2026-01-05", "2026-02-02"]

    material = digest._material(conn, recs, privileged=False)
    assert material.index("January") < material.index("February")
    # The summary carries the meeting, not the transcript: cost stays
    # low and long folders stay possible.
    assert "covered pricing and the renewal" in material
    assert "RAW TRANSCRIPT ONLY SENTENCE" not in material
    # Real lines with timestamps ride along so quotes can be anchored.
    assert "Quotable lines:" in material
    assert "@ 00:00:03" in material


def test_missing_summary_falls_back_to_chunked_retrieval(env):
    conn, folder = env
    _, tid = seed(conn, date="2026-03-03", title="Unsummarized",
                  folder_id=folder, summary=False)
    for n in range(6):
        conn.execute("INSERT INTO chunks (transcript_id, text) VALUES (?, ?)",
                     (tid, f"chunk {n} about pricing"))
    conn.commit()

    block = digest._source_block(
        conn, digest.folder_recordings(conn, folder)[0])
    assert "No summary for this recording" in block
    assert "chunk 0 about pricing" in block


def test_long_folder_is_condensed_into_period_briefs(env, llm_calls):
    conn, folder = env
    for n in range(digest.BATCH_SIZE + 2):
        seed(conn, date=f"2026-01-{n + 1:02d}", title=f"M{n}",
             folder_id=folder)

    digest.generate(conn, folder)

    # Two chronological batches condensed, then one synthesis over the
    # briefs: the whole folder never goes to the model at once.
    briefs = [c for c in llm_calls if "condensing part of" in c["prompt"]]
    finals = [c for c in llm_calls if "longitudinal digest" in c["prompt"]]
    assert len(briefs) == 2 and len(finals) == 1
    assert "Period 1 of 2" in briefs[0]["prompt"]
    assert "Period 2 of 2" in briefs[1]["prompt"]
    assert "Period 1 of 2" in finals[0]["prompt"]
    # The synthesis reads briefs, not the raw per-recording blocks.
    assert "[recording " not in finals[0]["prompt"]


def test_short_folder_skips_the_condensing_pass(env, llm_calls):
    conn, folder = env
    seed(conn, date="2026-01-05", title="January", folder_id=folder)
    digest.generate(conn, folder)
    assert len(llm_calls) == 1
    assert "longitudinal digest" in llm_calls[0]["prompt"]


def test_prompts_ask_for_trajectory_and_stay_descriptive(env, llm_calls):
    conn, folder = env
    seed(conn, date="2026-01-05", title="January", folder_id=folder)
    digest.generate(conn, folder)
    prompt = llm_calls[0]["prompt"]
    assert "Be descriptive, not diagnostic" in prompt
    assert "do not offer advice or clinical interpretation" in prompt
    assert "TRAJECTORY, not aggregation" in prompt
    for key in ("first_appeared", "open_threads", "commitments", "shifts",
                "framing_then", "framing_now"):
        assert key in prompt


def test_generate_stores_the_digest_with_what_it_covers(env, llm_calls):
    conn, folder = env
    rec, _ = seed(conn, date="2026-01-05", title="January", folder_id=folder)

    out = digest.generate(conn, folder)
    assert out["covered_ids"] == [rec]
    assert out["recordings_count"] == 1
    assert out["stale"] is False
    assert out["digest"]["themes"][0]["theme"] == "pricing"
    assert digest.get(conn, folder)["updated_at"]


def test_empty_folder_cannot_be_digested(env, llm_calls):
    conn, folder = env
    with pytest.raises(ValueError):
        digest.generate(conn, folder)
    assert llm_calls == []


# ---------- quotes are anchored to real lines ----------

def test_quotes_link_to_a_real_segment_and_invented_ones_are_dropped(
        env, monkeypatch):
    conn, folder = env
    rec, _ = seed(conn, date="2026-01-05", title="January", folder_id=folder)
    payload = json.loads(DIGEST_JSON)
    payload["quotes"].append({
        "quote": "a line nobody in this folder ever said out loud",
        "speaker": "Sam", "recording_id": rec, "timestamp": "00:00:01",
        "why": "invented"})
    monkeypatch.setattr(summarize, "_call_llm",
                        lambda p, privileged=False: json.dumps(payload))

    quotes = digest.generate(conn, folder)["digest"]["quotes"]
    assert len(quotes) == 1
    # The recording and timestamp come from the segment, not the model.
    assert quotes[0]["recording_id"] == rec
    assert quotes[0]["timestamp"] == "00:00:03"
    assert quotes[0]["start_seconds"] == 3.0


# ---------- incremental refresh ----------

def test_unchanged_folder_returns_the_cached_digest(env, llm_calls):
    conn, folder = env
    seed(conn, date="2026-01-05", title="January", folder_id=folder)
    digest.generate(conn, folder)
    assert len(llm_calls) == 1

    again = digest.generate(conn, folder)
    assert again["cached"] is True
    assert len(llm_calls) == 1


def test_adding_a_recording_marks_stale_and_updates_incrementally(
        env, llm_calls):
    conn, folder = env
    old, _ = seed(conn, date="2026-01-05", title="January", folder_id=folder)
    digest.generate(conn, folder)
    llm_calls.clear()

    new, _ = seed(conn, date="2026-03-09", title="March")
    organize.assign_folder(conn, [new], folder)

    state = digest.get(conn, folder)
    assert state["stale"] is True
    assert state["new_since"] == 1

    out = digest.generate(conn, folder)
    assert out["incremental"] is True
    assert out["covered_ids"] == [old, new]
    assert out["stale"] is False

    prompt = llm_calls[0]["prompt"]
    assert "updating an existing digest" in prompt
    # Only the new recording's material is re-sent; the folder is not
    # rebuilt to absorb one meeting.
    assert "March covered pricing" in prompt
    assert "January covered pricing" not in prompt
    # The stored digest goes along as the base to update in place.
    assert "a packaging problem" in prompt


def test_force_rebuilds_from_every_recording(env, llm_calls):
    conn, folder = env
    seed(conn, date="2026-01-05", title="January", folder_id=folder)
    digest.generate(conn, folder)
    new, _ = seed(conn, date="2026-03-09", title="March")
    organize.assign_folder(conn, [new], folder)
    llm_calls.clear()

    out = digest.generate(conn, folder, force=True)
    assert out["incremental"] is False
    prompt = llm_calls[0]["prompt"]
    assert "January covered pricing" in prompt
    assert "March covered pricing" in prompt


def test_removing_a_recording_rebuilds_rather_than_updates(env, llm_calls):
    conn, folder = env
    keep, _ = seed(conn, date="2026-01-05", title="January",
                   folder_id=folder)
    drop, _ = seed(conn, date="2026-02-02", title="February",
                   folder_id=folder)
    digest.generate(conn, folder)
    llm_calls.clear()

    organize.assign_folder(conn, [drop], None)
    out = digest.generate(conn, folder)
    assert out["incremental"] is False
    assert out["covered_ids"] == [keep]
    assert "February covered pricing" not in llm_calls[0]["prompt"]


def test_deleting_a_recording_marks_the_digest_stale(env, llm_calls):
    conn, folder = env
    seed(conn, date="2026-01-05", title="January", folder_id=folder)
    doomed, _ = seed(conn, date="2026-02-02", title="February",
                     folder_id=folder)
    digest.generate(conn, folder)
    assert digest.get(conn, folder)["stale"] is False

    manage.delete_recording(conn, doomed)
    assert digest.get(conn, folder)["stale"] is True


def test_deleting_a_folder_takes_its_digest(env, llm_calls):
    conn, folder = env
    seed(conn, date="2026-01-05", title="January", folder_id=folder)
    digest.generate(conn, folder)
    organize.delete_folder(conn, folder)
    assert conn.execute("SELECT count(*) FROM folder_digests").fetchone()[0] \
        == 0


# ---------- folder-scoped ask ----------

def test_search_can_be_scoped_to_one_folder(env, fake_embeddings):
    conn, folder = env
    inside, tid_in = seed(conn, date="2026-01-05", title="January",
                          folder_id=folder)
    outside, tid_out = seed(conn, date="2026-01-06", title="Elsewhere")
    semantic.index_transcript(conn, tid_in, "pricing inside the folder")
    semantic.index_transcript(conn, tid_out, "pricing outside the folder")

    everywhere = {h["recording_id"] for h in semantic.search(conn, "pricing")}
    assert everywhere == {inside, outside}

    scoped = semantic.search(conn, "pricing", folder_id=folder)
    assert {h["recording_id"] for h in scoped} == {inside}


def test_folder_scoped_ask_answers_from_that_folder_only(
        env, fake_embeddings, monkeypatch):
    conn, folder = env
    inside, tid_in = seed(conn, date="2026-01-05", title="January",
                          folder_id=folder)
    _, tid_out = seed(conn, date="2026-01-06", title="Elsewhere")
    semantic.index_transcript(conn, tid_in, "pricing inside the folder")
    semantic.index_transcript(conn, tid_out, "pricing outside the folder")

    seen = {}

    def fake(prompt, privileged=False):
        seen["prompt"] = prompt
        return "Scoped answer."
    monkeypatch.setattr(summarize, "_call_llm", fake)

    out = semantic.ask(conn, "what about pricing", folder_id=folder)
    assert [s["recording_id"] for s in out["sources"]] == [inside]
    assert "pricing inside the folder" in seen["prompt"]
    assert "outside the folder" not in seen["prompt"]


def test_folder_scoped_ask_with_nothing_indexed_says_so(env, fake_embeddings):
    conn, folder = env
    out = semantic.ask(conn, "anything", folder_id=folder)
    assert out["sources"] == []
    assert "this folder" in out["answer"]


# ---------- privileged folders ----------

def test_marking_a_folder_privileged_claims_its_recordings(env):
    conn, folder = env
    rec, _ = seed(conn, date="2026-01-05", title="January", folder_id=folder)
    assert db.is_privileged(conn, rec) is False

    out = organize.set_folder_privileged(conn, folder, True)
    assert out["recordings_marked"] == 1
    assert db.is_privileged(conn, rec) is True
    assert digest.folder_is_privileged(conn, folder) is True


def test_new_arrivals_in_a_privileged_folder_are_marked_privileged(env):
    conn, folder = env
    organize.set_folder_privileged(conn, folder, True)
    arrival, _ = seed(conn, date="2026-04-04", title="April")
    assert db.is_privileged(conn, arrival) is False

    organize.assign_folder(conn, [arrival], folder)
    assert db.is_privileged(conn, arrival) is True


def test_unmarking_a_folder_leaves_existing_recordings_privileged(env):
    conn, folder = env
    rec, _ = seed(conn, date="2026-01-05", title="January", folder_id=folder)
    organize.set_folder_privileged(conn, folder, True)
    organize.set_folder_privileged(conn, folder, False)
    assert db.is_privileged(conn, rec) is True
    # The folder itself no longer claims arrivals, but it still holds a
    # privileged recording, so its digest stays local.
    assert digest.folder_is_privileged(conn, folder) is True


def test_privileged_folder_digest_never_calls_the_cloud(env, monkeypatch):
    """The vault guarantee for digests: nothing reaches litellm."""
    import litellm

    conn, folder = env
    seed(conn, date="2026-01-05", title="January", folder_id=folder)
    organize.set_folder_privileged(conn, folder, True)

    def cloud_forbidden(*a, **kw):
        raise AssertionError("cloud call attempted for a privileged folder")
    monkeypatch.setattr(litellm, "completion", cloud_forbidden)

    local_prompts = []

    def local(prompt, max_tokens):
        local_prompts.append(prompt)
        return DIGEST_JSON
    monkeypatch.setattr(llm, "_local_complete", local)

    out = digest.generate(conn, folder)
    assert local_prompts and "longitudinal digest" in local_prompts[0]
    assert out["model"] == "local"


def test_one_privileged_recording_pins_the_whole_digest_local(
        env, monkeypatch):
    import litellm

    conn, folder = env
    seed(conn, date="2026-01-05", title="January", folder_id=folder)
    secret, _ = seed(conn, date="2026-02-02", title="February",
                     folder_id=folder)
    conn.execute("UPDATE recordings SET privileged = 1 WHERE id = ?",
                 (secret,))
    conn.commit()

    monkeypatch.setattr(litellm, "completion", lambda *a, **kw: (
        _ for _ in ()).throw(AssertionError("cloud call attempted")))
    monkeypatch.setattr(llm, "_local_complete",
                        lambda prompt, max_tokens: DIGEST_JSON)

    assert digest.generate(conn, folder)["model"] == "local"


def test_privileged_folder_digest_fails_closed_without_local_model(
        env, monkeypatch):
    import litellm

    conn, folder = env
    seed(conn, date="2026-01-05", title="January", folder_id=folder)
    organize.set_folder_privileged(conn, folder, True)

    monkeypatch.setattr(litellm, "completion", lambda *a, **kw: (
        _ for _ in ()).throw(AssertionError("cloud call attempted")))
    # Nothing in the gateway is stubbed out: a stand-in mlx_lm whose
    # weights refuse to load stands where the real one would, so the
    # real _local_complete and _get_local both run. Importing the real
    # mlx_lm here would read 162MB of Metal shaders for no extra proof.
    broken = types.ModuleType("mlx_lm")
    broken.generate = lambda *a, **kw: pytest.fail(
        "generation ran without a model")

    def no_weights(*a, **kw):
        raise RuntimeError("model weights are not downloaded")
    broken.load = no_weights
    monkeypatch.setitem(sys.modules, "mlx_lm", broken)
    monkeypatch.setattr(llm, "_local", None)

    with pytest.raises(llm.LocalModelUnavailable):
        digest.generate(conn, folder)


def test_privileged_folder_ask_never_falls_back_to_the_cloud(
        env, fake_embeddings, monkeypatch):
    conn, folder = env
    _, tid = seed(conn, date="2026-01-05", title="January",
                  folder_id=folder)
    semantic.index_transcript(conn, tid, "pricing inside the folder")
    organize.set_folder_privileged(conn, folder, True)

    monkeypatch.setattr(llm, "_get_local", lambda: (_ for _ in ()).throw(
        llm.LocalModelUnavailable("pending")))
    monkeypatch.setattr(summarize, "_call_llm", lambda p, privileged=False: (
        _ for _ in ()).throw(AssertionError("no LLM call is safe here")))

    out = semantic.ask(conn, "pricing", folder_id=folder)
    assert out["sources"] == []
    assert "privileged" in out["answer"]


def test_privileged_folder_ask_uses_the_local_model(
        env, fake_embeddings, monkeypatch):
    conn, folder = env
    _, tid = seed(conn, date="2026-01-05", title="January",
                  folder_id=folder)
    semantic.index_transcript(conn, tid, "pricing inside the folder")
    organize.set_folder_privileged(conn, folder, True)
    monkeypatch.setattr(llm, "_get_local", lambda: ("model", "tok"))

    seen = {}

    def fake(prompt, privileged=False):
        seen["privileged"] = privileged
        return "local answer"
    monkeypatch.setattr(summarize, "_call_llm", fake)

    semantic.ask(conn, "pricing", folder_id=folder)
    assert seen["privileged"] is True


# ---------- API ----------

def test_digest_endpoints(env, llm_calls):
    conn, folder = env
    rec, _ = seed(conn, date="2026-01-05", title="January", folder_id=folder)
    client = TestClient(main.app)

    empty = client.get(f"/api/folders/{folder}/digest")
    assert empty.status_code == 200
    assert empty.json()["digest"] is None
    assert empty.json()["recordings_in_folder"] == 1
    assert empty.json()["folder"]["name"] == "Client calls"

    built = client.post(f"/api/folders/{folder}/digest", json={})
    assert built.status_code == 200
    body = built.json()
    assert body["cached"] is False
    assert body["digest"]["recordings_count"] == 1
    assert body["digest"]["digest"]["overview"].startswith("Pricing came up")

    assert client.post(f"/api/folders/{folder}/digest",
                       json={}).json()["cached"] is True
    assert client.get("/api/folders/9999/digest").status_code == 404
    assert client.post("/api/folders/9999/digest", json={}).status_code == 404


def test_digest_endpoint_409s_on_an_empty_folder(env, llm_calls):
    conn, folder = env
    client = TestClient(main.app)
    resp = client.post(f"/api/folders/{folder}/digest", json={})
    assert resp.status_code == 409
    assert "no processed recordings" in resp.json()["detail"]


def test_folder_privileged_endpoint(env):
    conn, folder = env
    rec, _ = seed(conn, date="2026-01-05", title="January", folder_id=folder)
    client = TestClient(main.app)

    resp = client.put(f"/api/folders/{folder}/privileged",
                      json={"privileged": True})
    assert resp.status_code == 200
    assert resp.json()["recordings_marked"] == 1
    assert client.get("/api/folders").json()[0]["privileged"] is True
    conn2 = db.connect()
    assert db.is_privileged(conn2, rec) is True
    conn2.close()

    assert client.put("/api/folders/9999/privileged",
                      json={"privileged": True}).status_code == 404


def test_ask_endpoint_takes_a_folder(env, fake_embeddings, monkeypatch):
    conn, folder = env
    inside, tid_in = seed(conn, date="2026-01-05", title="January",
                          folder_id=folder)
    _, tid_out = seed(conn, date="2026-01-06", title="Elsewhere")
    semantic.index_transcript(conn, tid_in, "pricing inside the folder")
    semantic.index_transcript(conn, tid_out, "pricing outside the folder")
    monkeypatch.setattr(summarize, "_call_llm",
                        lambda p, privileged=False: "answer")

    client = TestClient(main.app)
    scoped = client.get("/api/ask", params={"q": "pricing",
                                            "folder_id": folder}).json()
    assert [s["recording_id"] for s in scoped["sources"]] == [inside]
    everywhere = client.get("/api/ask", params={"q": "pricing"}).json()
    assert len(everywhere["sources"]) == 2


# ---------- what the audit went looking for ----------

def test_a_quote_with_invented_words_appended_is_dropped(env, monkeypatch):
    """A real line with extra words bolted on is a fabrication wearing a
    true sentence, and it must not be shown with a real timestamp."""
    conn, folder = env
    rec, _ = seed(conn, date="2026-01-05", title="January", folder_id=folder)
    payload = json.loads(DIGEST_JSON)
    payload["quotes"] = [{
        "quote": QUOTE + " and I told the board we would fire everyone",
        "speaker": "Sam", "recording_id": rec, "timestamp": "00:00:03",
        "why": "real line plus invention"}]
    monkeypatch.setattr(summarize, "_call_llm",
                        lambda p, privileged=False: json.dumps(payload))
    assert digest.generate(conn, folder)["digest"]["quotes"] == []


def test_a_quote_two_recordings_could_have_said_is_dropped(env, monkeypatch):
    """With no usable recording id and more than one candidate, there is
    no way to know who is being quoted, so nobody is credited."""
    conn, folder = env
    seed(conn, date="2026-01-05", title="January", folder_id=folder)
    two, _ = seed(conn, date="2026-02-02", title="February",
                  folder_id=folder)
    conn.execute("UPDATE segments SET speaker = 'Dana' WHERE transcript_id = "
                 "(SELECT id FROM transcripts WHERE recording_id = ?)", (two,))
    conn.commit()
    payload = json.loads(DIGEST_JSON)
    payload["quotes"] = [{"quote": QUOTE, "speaker": "Sam",
                          "recording_id": None, "timestamp": "00:00:03",
                          "why": "ambiguous"}]
    monkeypatch.setattr(summarize, "_call_llm",
                        lambda p, privileged=False: json.dumps(payload))
    assert digest.generate(conn, folder)["digest"]["quotes"] == []


def test_a_quote_with_a_wrong_recording_id_is_not_reattributed(
        env, monkeypatch):
    """The named recording is the only one consulted, so a wrong id
    fails rather than landing on whoever happens to match first."""
    conn, folder = env
    seed(conn, date="2026-01-05", title="January", folder_id=folder)
    other, _ = seed(conn, date="2026-02-02", title="February",
                    folder_id=folder, quote="a completely different line "
                    "about the renewal timetable")
    payload = json.loads(DIGEST_JSON)
    payload["quotes"] = [{"quote": QUOTE, "speaker": "Sam",
                          "recording_id": other, "timestamp": "00:00:03",
                          "why": "wrong id"}]
    monkeypatch.setattr(summarize, "_call_llm",
                        lambda p, privileged=False: json.dumps(payload))
    assert digest.generate(conn, folder)["digest"]["quotes"] == []


def test_folder_search_still_finds_its_chunks_in_a_crowded_index(
        env, fake_embeddings):
    """The folder filter runs after the vector index has chosen its
    nearest chunks, so on a big archive a small folder's chunks can sit
    outside the first window. The search must widen until it finds them,
    not report the folder as empty."""
    conn, folder = env
    inside, tid_in = seed(conn, date="2026-01-05", title="January",
                          folder_id=folder)
    _, tid_out = seed(conn, date="2026-01-06", title="Elsewhere")
    semantic.index_transcript(conn, tid_in, "pricing inside the folder")
    # 200 chunks outside the folder, all closer to the query than the
    # one inside it, which is exactly the shape of the real archive.
    semantic.index_transcript(
        conn, tid_out, " ".join(["pricing pricing pricing"] * 8000))
    outside_count = conn.execute(
        "SELECT count(*) FROM chunks WHERE transcript_id = ?",
        (tid_out,)).fetchone()[0]
    assert outside_count > 60

    hits = semantic.search(conn, "pricing", folder_id=folder)
    assert [h["recording_id"] for h in hits] == [inside]


def test_ask_separates_an_unindexed_folder_from_no_match(
        env, fake_embeddings, monkeypatch):
    conn, folder = env
    empty = semantic.ask(conn, "anything", folder_id=folder)
    assert "is indexed yet" in empty["answer"]

    _, tid = seed(conn, date="2026-01-05", title="January",
                  folder_id=folder)
    semantic.index_transcript(conn, tid, "pricing inside the folder")
    monkeypatch.setattr(semantic, "search",
                        lambda *a, **kw: [])
    monkeypatch.setattr(summarize, "_call_llm", lambda p, privileged=False: (
        _ for _ in ()).throw(AssertionError("no hits, no call")))
    nothing = semantic.ask(conn, "unrelated", folder_id=folder)
    assert nothing["answer"] == "Nothing in this folder matched that question."


def test_rewriting_a_summary_marks_the_digest_stale(env, llm_calls):
    """A reprocess rewrites the summary a digest was built from, so the
    digest is describing something that no longer exists."""
    conn, folder = env
    rec, tid = seed(conn, date="2026-01-05", title="January",
                    folder_id=folder)
    digest.generate(conn, folder)
    assert digest.get(conn, folder)["stale"] is False

    db.upsert_summary(conn, tid, summary_for("January, rewritten",
                                             "2026-01-05"), model="test")
    assert digest.get(conn, folder)["stale"] is True


def test_a_covered_recording_leaving_the_set_reads_as_stale(env, llm_calls):
    """A recording that stops being done (a reprocess in flight, say)
    drops out of the folder's set without anything marking the flag."""
    conn, folder = env
    seed(conn, date="2026-01-05", title="January", folder_id=folder)
    gone, _ = seed(conn, date="2026-02-02", title="February",
                   folder_id=folder)
    digest.generate(conn, folder)
    assert digest.get(conn, folder)["stale"] is False

    db.set_recording_status(conn, gone, "transcribing")
    state = digest.get(conn, folder)
    assert state["stale"] is True
    assert state["dropped_since"] == 1
