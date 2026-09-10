import json

import pytest

from app import summarize

VALID = json.dumps({
    "title": "Budget Review",
    "date": "2026-07-07",
    "attendees": ["Alexa", "Sam"],
    "summary": "The team reviewed the quarterly budget.",
    "decisions": ["Cut travel spend by 10 percent"],
    "action_items": [
        {"owner": "Sam", "task": "Update the forecast",
         "due_date": "2026-07-14", "priority": "high"},
    ],
    "risks": ["Vendor costs may rise"],
    "topics": ["budget"],
})


def test_valid_json_parses_on_first_try(monkeypatch):
    calls = []
    monkeypatch.setattr(summarize, "_call_llm",
                        lambda p, privileged=False: calls.append(p) or VALID)
    result = summarize.summarize("transcript text")
    assert len(calls) == 1
    assert result.title == "Budget Review"
    assert result.action_items[0].owner == "Sam"


def test_code_fences_are_stripped(monkeypatch):
    monkeypatch.setattr(summarize, "_call_llm",
                        lambda p, privileged=False: "```json\n" + VALID + "\n```")
    result = summarize.summarize("transcript text")
    assert result.title == "Budget Review"


def test_malformed_json_triggers_cleanup_pass(monkeypatch):
    responses = iter(["this is not json at all", VALID])
    calls = []

    def fake(prompt, privileged=False):
        calls.append(prompt)
        return next(responses)

    monkeypatch.setattr(summarize, "_call_llm", fake)
    result = summarize.summarize("transcript text")
    assert len(calls) == 2
    assert "failed to parse" in calls[1]
    assert result.title == "Budget Review"


def test_cleanup_failure_raises(monkeypatch):
    monkeypatch.setattr(summarize, "_call_llm", lambda p, privileged=False: "still not json")
    with pytest.raises(Exception):
        summarize.summarize("transcript text")


def test_schema_rejects_missing_required_fields(monkeypatch):
    bad = json.dumps({"date": "2026-07-07"})  # no title or summary
    responses = iter([bad, bad])
    monkeypatch.setattr(summarize, "_call_llm", lambda p, privileged=False: next(responses))
    with pytest.raises(Exception):
        summarize.summarize("transcript text")


def test_transcript_text_reaches_the_prompt(monkeypatch):
    seen = {}

    def fake(prompt, privileged=False):
        seen["prompt"] = prompt
        return VALID

    monkeypatch.setattr(summarize, "_call_llm", fake)
    summarize.summarize("the llama walked into the office")
    assert "the llama walked into the office" in seen["prompt"]


# PLAN.md's routing rule: routine per-meeting calls go to a cheap model,
# and the model name is config, not a hard-coded string. Nothing asserted
# either half until now.

def test_default_model_is_the_cheap_tier():
    import importlib

    from app import config
    importlib.reload(config)
    assert config.LLM_MODEL == "anthropic/claude-haiku-4-5"
    assert "haiku" in config.LLM_MODEL.lower()


def test_env_var_overrides_the_model(monkeypatch):
    import importlib

    from app import config
    monkeypatch.setenv("OTTER_LLM_MODEL", "anthropic/claude-sonnet-5")
    importlib.reload(config)
    try:
        assert config.LLM_MODEL == "anthropic/claude-sonnet-5"
    finally:
        monkeypatch.delenv("OTTER_LLM_MODEL")
        importlib.reload(config)
    assert config.LLM_MODEL == "anthropic/claude-haiku-4-5"


def test_cloud_call_sends_the_configured_model(monkeypatch):
    """The gateway must pass config.LLM_MODEL through, not its own string."""
    import sys
    import types

    from app import config, llm

    seen = {}

    class _Msg:
        content = "ok"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    fake = types.ModuleType("litellm")

    def completion(**kwargs):
        seen.update(kwargs)
        return _Resp()

    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)
    monkeypatch.setattr(config, "LLM_MODEL", "anthropic/claude-haiku-4-5")

    assert llm.complete("hi") == "ok"
    assert seen["model"] == "anthropic/claude-haiku-4-5"
