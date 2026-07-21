"""
Integration-service tests: the /suggest contract the Oblsk platform calls.
All offline; one test runs a real HTTP round trip against the stdlib server.
"""

from __future__ import annotations
import json
import os
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from oblsk_negotiator.campaign import load_campaign
from oblsk_negotiator.service import _Handler, suggest

os.environ["OBLSK_NO_LLM"] = "1"

CAMP = load_campaign(os.path.join(os.path.dirname(__file__), "..", "examples",
                                  "unest_campaign.yaml"))


def _msg(role, content, at):
    return {"role": role, "receivedAt": at, "content": content}


def test_fresh_thread_opens_with_the_anchor():
    out = suggest({
        "creatorHandle": "@thecollinskidsfamily",
        "medianViewsPerVideo": 150000,
        "messages": [_msg("creator", "Hi! Karissa would love to partner. "
                                     "What did you have in mind?", 1)],
    }, CAMP)
    assert out["agent"]["action"] == "opening_flat"
    assert out["strategy"]["kind"] == "initial_offer"
    assert "$" in out["message"]
    lad = out["agent"]["ladder"]
    assert lad["anchorUsd"] <= lad["targetUsd"] <= lad["walkAwayUsd"]


def test_state_derived_from_thread_no_reopening():
    # A human already offered $2,000; the creator counters just inside our
    # acceptable window. The agent must pick the thread up mid-flight (accept),
    # not re-open at the anchor.
    from oblsk_negotiator.pricing import price_ladder
    from oblsk_negotiator.service import _view_model_from
    vm = _view_model_from({"medianViewsPerVideo": 150000})
    lad = price_ladder(vm, CAMP.econ, CAMP.ctx.pricing_policy())
    ask = round(min(lad.target * CAMP.ctx.accept_margin, lad.walk_away) * 0.98)
    out = suggest({
        "medianViewsPerVideo": 150000,
        "messages": [
            _msg("creator", "What were you thinking budget-wise?", 1),
            _msg("brand", "We could allocate $2,000 for one cross-posted video.", 2),
            _msg("creator", f"Could you do ${ask:,.0f}?", 3),
        ],
    }, CAMP)
    assert out["agent"]["action"] == "accept"
    assert out["strategy"]["kind"] == "accept"
    assert f"${ask:,.0f}" in out["message"]


def test_walk_away_holds_over_http_contract_too():
    # An ask above the ceiling is never accepted, whatever the thread history.
    out = suggest({
        "medianViewsPerVideo": 150000,
        "messages": [
            _msg("brand", "We were thinking $2,500 for the video.", 1),
            _msg("creator", "I need $3,600 for a single video.", 2),
        ],
    }, CAMP)
    walk = out["agent"]["ladder"]["walkAwayUsd"]
    assert 3600 > walk
    assert out["agent"]["action"] != "accept"


def test_bundle_reply_carries_package_terms():
    out = suggest({
        "medianViewsPerVideo": 150000,
        "messages": [
            _msg("brand", "We were thinking $1,900 for the video.", 1),
            _msg("creator", "Could you do $3,100 a video?", 2),
            _msg("brand", "We can stretch to $2,500 for the video.", 3),
            _msg("creator", "I really can't go below $3,100 a video.", 4),
        ],
    }, CAMP)
    assert out["agent"]["action"] == "escalate_bundle"
    pkg = out["strategy"]["packageTerms"]
    assert pkg and pkg["videoCount"] > 1
    assert pkg["effectivePerVideoUsd"] <= out["agent"]["ladder"]["walkAwayUsd"] * 1.01


def test_own_bundle_message_reconstructs_and_holds():
    # Regression: the agent's own "$7,000 total for 3 videos ($2,333 each)"
    # used to re-read as a $2,333 offer, so a creator follow-up got a hold_firm
    # reply restating a slashed price. The standing offer must survive the
    # thread -> state round trip intact.
    out = suggest({
        "creatorHandle": "Devin",
        "medianViewsPerVideo": 150000,
        "messages": [
            _msg("brand", "We were thinking $1,450 for the video.", 1),
            _msg("creator", "Could you do $2,450 a video?", 2),
            _msg("brand", "Here is something that might work better for both "
                          "of us: $7,000 total for 3 videos ($2,333 each).", 3),
            _msg("creator", "Hey, just following up on this! Still interested.", 4),
        ],
    }, CAMP)
    assert out["agent"]["action"] == "hold_firm"
    deal = out["agent"]["deal"]
    assert deal["video_count"] == 3
    assert abs(deal["flat_per_video"] * 3 - 7000) <= 1
    assert "$7,000" in out["message"]


def test_ev_snapshot_freezes_pricing_across_turns():
    # First turn builds and echoes the snapshot.
    out1 = suggest({
        "creatorHandle": "@fam",
        "medianViewsPerVideo": 150000,
        "messages": [_msg("creator", "What did you have in mind?", 1)],
    }, CAMP)
    snap = out1["agent"]["evSnapshot"]
    assert snap and "viewModel" in snap

    # A later turn with a very different reach signal, but resending the snapshot,
    # prices off the frozen model -> identical ladder.
    later = [_msg("creator", "What did you have in mind?", 1),
             _msg("brand", "We were thinking $1,500 for one video.", 2),
             _msg("creator", "Hmm, can you share more on usage?", 3)]
    out2 = suggest({"medianViewsPerVideo": 999999, "evSnapshot": snap,
                    "messages": later}, CAMP)
    assert out2["agent"]["ladder"] == out1["agent"]["ladder"]

    # Without the snapshot, the inflated reach signal moves the ladder — proving
    # the snapshot is what pinned it.
    out2b = suggest({"medianViewsPerVideo": 999999, "messages": later}, CAMP)
    assert out2b["agent"]["ladder"] != out1["agent"]["ladder"]


def test_history_reconstruction_is_deterministic_and_heuristic(monkeypatch):
    # History is rebuilt with the deterministic heuristic, not the LLM: the same
    # thread reconstructs to the same offer every time (no per-request LLM drift
    # on recorded positions), and the interpreter is called at most once — for
    # the single live message, never once-per-history-turn.
    import oblsk_negotiator.service as svc

    real = svc.interpret_message
    calls = {"n": 0}

    def spy(text):
        calls["n"] += 1
        return real(text)

    monkeypatch.setattr(svc, "interpret_message", spy)
    payload = {
        "medianViewsPerVideo": 150000,
        "messages": [
            _msg("creator", "What were you thinking budget-wise?", 1),
            _msg("brand", "We could allocate $2,000 total for one cross-posted video.", 2),
            _msg("creator", "Following up — still keen!", 3),
        ],
    }
    out1 = suggest(payload, CAMP)
    assert calls["n"] == 1                       # live message only, not history
    out2 = suggest(payload, CAMP)
    assert out1["agent"]["deal"] == out2["agent"]["deal"]   # reproducible
    assert out1["agent"]["deal"]["flat_per_video"] == 2000.0


def test_malformed_ev_snapshot_is_client_error():
    # A corrupt resent snapshot is a bad client payload (-> 400 via ValueError),
    # not an unhandled 500.
    base = [_msg("creator", "What did you have in mind?", 1)]
    for bad in ({"econ": {"conversionRate": 0.0005, "ltvUsd": 80}},        # no viewModel
                {"viewModel": {"family": "lognormal"}},                    # no econ / params
                {"viewModel": {"family": "lognormal", "params": {}}, "econ": {}}):
        with pytest.raises(ValueError, match="evSnapshot"):
            suggest({"medianViewsPerVideo": 150000, "messages": base,
                     "evSnapshot": bad}, CAMP)


def test_bad_payloads_raise():
    with pytest.raises(ValueError, match="messages"):
        suggest({"medianViewsPerVideo": 150000, "messages": []}, CAMP)
    with pytest.raises(ValueError, match="ours"):
        suggest({"medianViewsPerVideo": 150000,
                 "messages": [_msg("brand", "hello!", 1)]}, CAMP)
    with pytest.raises(ValueError, match="medianViewsPerVideo"):
        suggest({"messages": [_msg("creator", "hi", 1)]}, CAMP)


def test_http_round_trip_and_auth():
    from http.server import ThreadingHTTPServer
    _Handler.campaign = CAMP
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        os.environ["OBLSK_SERVICE_TOKEN"] = "s3cret"
        body = json.dumps({
            "creatorHandle": "@maya",
            "medianViewsPerVideo": 150000,
            "messages": [_msg("creator", "Interested! What did you have in mind?", 1)],
        }).encode()

        # No token -> 401.
        req = urllib.request.Request(base + "/suggest", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req)
            assert False, "expected 401"
        except urllib.error.HTTPError as e:
            assert e.code == 401

        # With the token -> a full suggestion.
        req = urllib.request.Request(base + "/suggest", data=body, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer s3cret"})
        with urllib.request.urlopen(req) as resp:
            out = json.loads(resp.read())
        assert out["agent"]["action"] == "opening_flat"
        assert out["strategy"]["kind"] == "initial_offer"

        # Health check works unauthenticated is wrong — GET requires no auth.
        with urllib.request.urlopen(base + "/health") as resp:
            health = json.loads(resp.read())
        assert health["ok"] is True
    finally:
        os.environ.pop("OBLSK_SERVICE_TOKEN", None)
        server.shutdown()
