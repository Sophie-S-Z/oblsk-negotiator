"""LLM resilience: a transient blip must not permanently downgrade the process
to templates the way a hard auth failure does.

These tests drive the module-level failure classification directly with fake
exceptions and a fake client, monkeypatching time so the cooldown is
observable without sleeping.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from oblsk_negotiator import llm  # noqa: E402


class _StatusError(Exception):
    """Stand-in for an anthropic APIStatusError-like exception."""

    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class RateLimitError(Exception):
    """Name-classified transient error (no status_code attribute)."""


class AuthenticationError(Exception):
    """Name-classified hard error."""


class _FakeResponse:
    stop_reason = "end_turn"

    class _Block:
        type = "text"
        text = "ok"

    content = [_Block()]


class _FakeClient:
    """Raises `exc` for the first `fail_times` calls, then returns text."""

    def __init__(self, exc, fail_times):
        self._exc = exc
        self._fail_times = fail_times
        self.calls = 0
        self.messages = self

    def create(self, **_kwargs):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return _FakeResponse()


def _reset(monkeypatch, *, now=1000.0):
    """Fresh module state with the kill-switch off and a controllable clock."""
    monkeypatch.delenv("OBLSK_NO_LLM", raising=False)
    monkeypatch.setenv("OBLSK_LLM_MAX_RETRIES", "0")
    llm._disabled = False
    llm._cooldown_until = 0.0
    llm._consecutive_failures = 0
    clock = {"t": now}
    monkeypatch.setattr(llm.time, "time", lambda: clock["t"])
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    return clock


def test_transient_failure_cools_down_then_re_enables(monkeypatch):
    clock = _reset(monkeypatch)
    llm._client = _FakeClient(RateLimitError("429"), fail_times=1)

    # First call fails transiently -> None, and the LLM goes on cooldown.
    assert llm.complete("s", "u") is None
    assert llm._cooldown_until > clock["t"]
    assert llm.llm_available() is False   # mid-cooldown

    # After the cooldown elapses, the LLM is available again and the next call
    # (the fake now succeeds) returns text.
    clock["t"] = llm._cooldown_until + 1
    assert llm.llm_available() is True
    assert llm.complete("s", "u") == "ok"
    assert llm._consecutive_failures == 0   # reset on success


def test_status_5xx_is_transient(monkeypatch):
    _reset(monkeypatch)
    llm._client = _FakeClient(_StatusError(503), fail_times=1)
    assert llm.complete("s", "u") is None
    assert llm._disabled is False           # not latched
    assert llm._cooldown_until > 0


def test_hard_auth_failure_latches_off(monkeypatch):
    _reset(monkeypatch)
    llm._client = _FakeClient(AuthenticationError("bad key"), fail_times=1)
    assert llm.complete("s", "u") is None
    assert llm._disabled is True            # latched for the rest of the run
    assert llm.llm_available() is False


def test_status_401_is_hard(monkeypatch):
    _reset(monkeypatch)
    llm._client = _FakeClient(_StatusError(401), fail_times=1)
    assert llm.complete("s", "u") is None
    assert llm._disabled is True


def test_in_call_retry_recovers(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("OBLSK_LLM_MAX_RETRIES", "2")
    client = _FakeClient(RateLimitError("429"), fail_times=1)
    llm._client = client
    # One transient failure then success, within the retry budget -> text, no cooldown.
    assert llm.complete("s", "u") == "ok"
    assert client.calls == 2
    assert llm._cooldown_until == 0.0
