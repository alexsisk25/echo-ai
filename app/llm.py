"""The single gateway for every LLM call. The privacy vault lives here.

Every feature (summaries, ask, dossiers, translation) calls complete().
When privileged is True the prompt goes to a small local model via
mlx-lm and NEVER to the cloud; if the local model cannot load, the call
fails with LocalModelUnavailable rather than falling back to the cloud.
"""

from app import config


class LocalModelUnavailable(RuntimeError):
    """The local model is not installed or failed to load. Callers must
    surface this, never downgrade to a cloud call."""


_local = None


def _get_local():
    global _local
    if _local is None:
        try:
            from mlx_lm import load
        except ImportError as err:
            raise LocalModelUnavailable(
                "mlx-lm is not installed; local analysis pending"
            ) from err
        try:
            _local = load(config.LOCAL_LLM_MODEL)
        except Exception as err:
            raise LocalModelUnavailable(
                f"Local model {config.LOCAL_LLM_MODEL} failed to load: {err}"
            ) from err
    return _local


def _local_complete(prompt: str, max_tokens: int) -> str:
    from mlx_lm import generate

    model, tokenizer = _get_local()
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    return generate(model, tokenizer, prompt=chat, max_tokens=max_tokens)


def complete(prompt: str, privileged: bool = False,
             max_tokens: int = 4096) -> str:
    if privileged:
        return _local_complete(prompt, max_tokens)

    import litellm

    response = litellm.completion(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
