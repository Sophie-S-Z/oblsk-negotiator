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
"""

from __future__ import annotations

import json
import os
from typing import Optional

DEFAULT_MODEL = "claude-opus-4-8"

_client = None
_disabled = False   # set on first hard failure so we do not retry every call


def model_name() -> str:
    return os.environ.get("OBLSK_LLM_MODEL", DEFAULT_MODEL)


def _get_client():
    """The shared client, or None when the LLM path is off. Off means: the
    kill-switch is set, the anthropic package is not installed, or a previous
    call failed hard (no credentials, no network)."""
    global _client, _disabled
    if _disabled or os.environ.get("OBLSK_NO_LLM"):
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


def complete(system: str, user: str, *, max_tokens: int = 1024) -> Optional[str]:
    """One completion. Returns the text, or None when the LLM path is off or
    the call fails — the caller falls back to its offline path."""
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.messages.create(
            model=model_name(), max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}])
        if response.stop_reason == "refusal":
            return None
        return next((b.text for b in response.content if b.type == "text"), None)
    except Exception:
        _mark_failed()
        return None


def complete_json(system: str, user: str, schema: dict,
                  *, max_tokens: int = 1024) -> Optional[dict]:
    """One completion constrained to a JSON schema (structured outputs).
    Returns the parsed dict, or None on any failure."""
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.messages.create(
            model=model_name(), max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}})
        if response.stop_reason == "refusal":
            return None
        text = next((b.text for b in response.content if b.type == "text"), None)
        return json.loads(text) if text else None
    except Exception:
        _mark_failed()
        return None


def _mark_failed():
    """A call failed hard (auth, network). Go offline for the rest of the run
    rather than paying a timeout on every message."""
    global _disabled
    _disabled = True
