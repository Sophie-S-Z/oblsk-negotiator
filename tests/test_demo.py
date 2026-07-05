"""
Guards for the terminal entry points people actually use: the plain-English
calculation breakdown, the interactive --chat loop (the same read -> price ->
draft the platform runs), and --random. --chat/--random go through a
subprocess so the real CLI wiring is exercised, not just the functions.
"""

from __future__ import annotations
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV = {**os.environ, "OBLSK_NO_LLM": "1", "PYTHONIOENCODING": "utf-8"}


def test_explain_ladder_shows_the_derivation():
    import demo
    from oblsk_negotiator import CreatorEconomics, fit_view_model, CampaignContext
    import numpy as np
    vm = fit_view_model(np.random.default_rng(1).lognormal(np.log(150000), 0.6, 24))
    text = demo.explain_ladder(vm, CreatorEconomics(0.0005, 80), CampaignContext())
    for token in ("REACH", "DOLLARS", "PRICES", "anchor", "target", "walk-away"):
        assert token in text, f"explain_ladder missing {token!r}"


def _run(args, stdin=""):
    return subprocess.run([sys.executable, "demo.py", *args], cwd=ROOT, env=ENV,
                          input=stdin, capture_output=True, text=True, timeout=120)


def test_chat_closes_a_deal_over_stdin():
    # Play the creator: greet, counter high, then meet the target -> accept.
    script = ("Hi, what did you have in mind?\n"
              "I usually get $4,000 a video, can you do $3,500?\n"
              "Could you meet me at $2,000?\n"
              "quit\n")
    r = _run(["--chat", "--campaign", "examples/unest_campaign.yaml",
              "--median", "150000", "--name", "Karissa"], stdin=script)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "opening_flat" in out          # opened at the anchor
    assert "accept" in out                # closed the deal
    assert "thread accepted" in out
    assert "ceiling" in out               # priced against the walk-away


def test_random_runs_a_full_negotiation():
    r = _run(["--random", "--seed", "12345"])
    assert r.returncode == 0, r.stderr
    assert "random creator:" in r.stdout
    assert "--- result ---" in r.stdout
    assert "How these prices are calculated" in r.stdout
