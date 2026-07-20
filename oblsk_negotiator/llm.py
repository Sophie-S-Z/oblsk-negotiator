"""
The LLM layer. One thin wrapper around the Anthropic SDK that the rest of the
agent talks to; nothing else imports anthropic.

Three call sites use it:
    prose.write_message   drafts the outbound message for a decision
    qa.answer_question    answers open-ended creator questions from the brief
    replay.interpret      reads a real creator message into intent + ask

Every caller has a deterministic offline fallback (templates, keyword rules),
so the agent runs with no key and no network — the LLM makes it sound human and
read messy real-world text; the calculator and the behavior tree still make
every pricing decision. The model never invents a price or picks the move.

Configuration (environment):
    ANTHROPIC_API_KEY   the key (or an `ant auth login` profile)
    OBLSK_LLM_MODEL     model override (default claude-opus-4-8)
    OBLSK_NO_LLM=1      force offline mode (tests, CI, demos)
    OBLSK_LLM_MAX_RETRIES  in-call retries on a transient error (default 1)

Failure handling: a *hard* failure (no anthropic package, bad/absent
credentials) latches the LLM off for the rest of the process — retrying it
would just pay a timeout on every message. A *transient* failure (timeout,
rate limit, 5xx, dropped connection) does not latch: the call retries a couple
of times, and if it still fails the LLM goes on a short exponential-backoff
cooldown and re-enables itself afterwards. One bad network moment must not
downgrade a long-running service to templates for its whole lifetime.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

DEFAULT_MODEL = "claude-opus-4-8"

_client = None
_disabled = False        # hard failure: off for the rest of the run
_cooldown_until = 0.0    # transient failure: off until this monotonic-ish time
_consecutive_failures = 0
_MAX_BACKOFF_S = 60.0

# Exception classes (by name) and HTTP status codes that mean "this key/setup
# will never work" — latch off. Everything else caught is treated as transient.
_HARD_STATUS = {401, 403}
_HARD_NAMES = {"AuthenticationError", "PermissionDeniedError"}


def model_name() -> str:
    return os.environ.get("OBLSK_LLM_MODEL", DEFAULT_MODEL)


def _max_retries() -> int:
    try:
        return max(0, int(os.environ.get("OBLSK_LLM_MAX_RETRIES", "1")))
    except ValueError:
        return 1


def _is_hard(exc: Exception) -> bool:
    """Hard failures latch the LLM off; transient ones only cool it down. We
    classify by status code / class name so this module never imports anthropic
    types (it must import even when the package is absent)."""
    if getattr(exc, "status_code", None) in _HARD_STATUS:
        return True
    return type(exc).__name__ in _HARD_NAMES


def _get_client():
    """The shared client, or None when the LLM path is off. Off means: the
    kill-switch is set, the anthropic package is not installed, a previous call
    failed hard (no credentials, no network), or we are mid-cooldown after a
    transient failure."""
    global _client, _disabled
    if _disabled or os.environ.get("OBLSK_NO_LLM"):
        return None
    if time.time() < _cooldown_until:
        return None
    if _client is None:
        try:
            import anthropic
            _client = anthropic.Anthropic()
        except Exception:
            _disabled = True
            return None
    return _client


def llm_available() -> bool:
    return _get_client() is not None


def _mark_failed(exc: Exception) -> None:
    """Record a failed call. Hard failures latch off for the run; transient
    ones start (or extend) a short exponential-backoff cooldown."""
    global _disabled, _cooldown_until, _consecutive_failures
    if _is_hard(exc):
        _disabled = True
        return
    _consecutive_failures += 1
    backoff = min(2.0 ** _consecutive_failures, _MAX_BACKOFF_S)
    _cooldown_until = time.time() + backoff


def _reset_failures() -> None:
    global _consecutive_failures, _cooldown_until
    _consecutive_failures = 0
    _cooldown_until = 0.0


def _call(make_response):
    """Run one LLM call with in-call retry on transient errors. `make_response`
    takes the client and returns the raw response; returns the response or
    None (LLM off, refusal handled by caller, or all attempts failed)."""
    client = _get_client()
    if client is None:
        return None
    attempts = _max_retries() + 1
    for i in range(attempts):
        try:
            resp = make_response(client)
            _reset_failures()
            return resp
        except Exception as exc:  # noqa: BLE001 - classified in _mark_failed
            if _is_hard(exc) or i == attempts - 1:
                _mark_failed(exc)
                return None
            time.sleep(min(2.0 ** i, _MAX_BACKOFF_S))
    return None


def complete(system: str, user: str, *, max_tokens: int = 1024) -> Optional[str]:
    """One completion. Returns the text, or None when the LLM path is off or
    the call fails — the caller falls back to its offline path."""
    response = _call(lambda client: client.messages.create(
        model=model_name(), max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}]))
    if response is None or response.stop_reason == "refusal":
        return None
    return next((b.text for b in response.content if b.type == "text"), None)


def complete_json(system: str, user: str, schema: dict,
                  *, max_tokens: int = 1024) -> Optional[dict]:
    """One completion constrained to a JSON schema (structured outputs).
    Returns the parsed dict, or None on any failure."""
    response = _call(lambda client: client.messages.create(
        model=model_name(), max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema}}))
    if response is None or response.stop_reason == "refusal":
        return None
    text = next((b.text for b in response.content if b.type == "text"), None)
    return json.loads(text) if text else None
