"""
Replay tests: parsing a pasted thread, reading creator messages offline, and
shadow-running the agent over a real (sample) negotiation. All offline — the
heuristic interpreter is exercised directly, so no key or network is needed.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from oblsk_negotiator.behavior_tree import Action, CampaignContext, Intent
from oblsk_negotiator.campaign import load_campaign
from oblsk_negotiator.ev_engine import CreatorEconomics, fit_view_model
from oblsk_negotiator.replay import (heuristic_interpret, parse_email_thread,
                                     parse_thread, parse_transcript,
                                     replay_thread)

os.environ["OBLSK_NO_LLM"] = "1"   # replay must be fully offline in tests

ECON = CreatorEconomics(conversion_rate=0.0016, ltv_usd=80)
EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
SAMPLE = os.path.join(EXAMPLES, "sample_thread.txt")
UNEST_THREAD = os.path.join(EXAMPLES, "unest_thread.txt")
UNEST_CAMPAIGN = os.path.join(EXAMPLES, "unest_campaign.yaml")


def _vm(median=50000, seed=0):
    rng = np.random.default_rng(seed)
    return fit_view_model(rng.lognormal(np.log(median), 0.6, 24))


def test_parse_transcript_turns_and_continuations():
    text = ("Creator: hey there!\n"
            "still me on a second line\n"
            "Us: hi! rate is $1,800.\n"
            "Creator: can you do $2,500?\n")
    turns = parse_transcript(text)
    assert [t.sender for t in turns] == ["creator", "us", "creator"]
    assert "second line" in turns[0].text


def test_parse_transcript_aliases_and_garbage():
    turns = parse_transcript("Maya: I'm in!\nUs: great",
                             creator_aliases=["Maya"])
    assert turns[0].sender == "creator"
    with pytest.raises(ValueError):
        parse_transcript("no prefixes anywhere")


def test_heuristic_reads_asks():
    m = heuristic_interpret("I'm usually at $3k but could you do $2,500?")
    assert m.intent == Intent.NEGOTIATING and m.ask_total_usd == 2500
    m = heuristic_interpret("I charge $1,200 per video for 3 videos")
    assert m.ask_total_usd == 3600 and m.ask_video_count == 3
    assert heuristic_interpret("That works for me, let's do it.").intent == Intent.ACCEPTED
    assert heuristic_interpret("No thanks, I'll pass.").intent == Intent.REJECTING
    assert heuristic_interpret("What's the timeline?").intent == Intent.QUESTION


def test_email_thread_parses_real_gmail_paste():
    with open(UNEST_THREAD, encoding="utf-8") as f:
        text = f.read()
    turns = parse_email_thread(text, us_aliases=["unest.co"])
    assert [t.sender for t in turns] == ["creator", "us", "creator", "creator",
                                         "us", "creator", "us", "creator"]
    # Signatures survive but confidentiality footers are cut.
    assert not any("CONFIDENTIALITY" in t.text for t in turns)
    # The quoted rate card in the first message is preserved and readable.
    m = heuristic_interpret(turns[0].text)
    assert m.intent == Intent.NEGOTIATING and m.ask_total_usd == 9000
    # parse_thread auto-detects the email format.
    assert len(parse_thread(text, us_aliases=["unest.co"])) == len(turns)


def test_contract_terms_route_to_human():
    m = heuristic_interpret("Overall the contract looks great, but Section 9's "
                            "exclusivity provision needs $2,000 per deliverable.")
    assert m.contract_terms
    import numpy as np
    from oblsk_negotiator.behavior_tree import decide
    from oblsk_negotiator.state import NegotiationState
    s = NegotiationState("c", "k", "t")
    d = decide(m, s, _vm(), ECON, CampaignContext())
    assert d.action == Action.ESCALATE_HUMAN
    assert "contract terms" in d.rationale


def test_follow_up_holds_position():
    from oblsk_negotiator.behavior_tree import CreatorMessage, decide
    from oblsk_negotiator.state import NegotiationState
    s = NegotiationState("c", "k", "t")
    vm = _vm()
    d0 = decide(CreatorMessage(Intent.INTERESTED, raw_text="interested"),
                s, vm, ECON, CampaignContext())
    d1 = decide(CreatorMessage(Intent.INTERESTED,
                               raw_text="just following up on this!"),
                s, vm, ECON, CampaignContext())
    assert d1.action == Action.HOLD_FIRM
    assert d1.deal.total_usd == d0.deal.total_usd   # no unforced concession
    assert s.round_count == 1                        # a nudge costs no round


def test_unest_campaign_loads_and_replays():
    camp = load_campaign(UNEST_CAMPAIGN)
    assert camp.brief.brand == "UNest"
    assert camp.ctx.target_videos == 3
    assert "unest.co" in camp.us_aliases
    with open(UNEST_THREAD, encoding="utf-8") as f:
        thread = f.read()
    report = replay_thread(thread, _vm(150000), camp.econ, camp.ctx,
                           brief=camp.brief, us_aliases=camp.us_aliases)
    assert len(report.steps) == 5
    # The nudge emails hold the standing offer instead of conceding...
    assert Action.HOLD_FIRM in [s.decision.action for s in report.steps]
    # ...and the contract-revisions email goes to a person.
    assert report.steps[-1].decision.action == Action.ESCALATE_HUMAN
    assert report.final_status == "escalated"


def test_campaign_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("economics: {conversion_rate: 0.001, ltv_usd: 50}\n"
                   "negotiation: {roi_targett: 3}\n")
    with pytest.raises(ValueError, match="roi_targett"):
        load_campaign(str(bad))


def test_unanswerable_question_asks_the_team():
    # "Ingredient questions": outside the brief -> ask the team, never bluff.
    from oblsk_negotiator.behavior_tree import decide
    from oblsk_negotiator.state import NegotiationState
    s = NegotiationState("c", "k", "t")
    m = heuristic_interpret("Does the serum contain fragrance or parabens?")
    assert m.intent == Intent.QUESTION
    d = decide(m, s, _vm(), ECON, CampaignContext())
    assert d.action == Action.ASK_HUMAN
    assert "fragrance" in d.human_prompt
    # An answerable question still gets answered by the agent itself.
    d2 = decide(heuristic_interpret("What's the timeline looking like?"),
                s, _vm(), ECON, CampaignContext())
    assert d2.action == Action.ANSWER_QUESTION


def test_call_reference_without_context_asks_the_team():
    from oblsk_negotiator.behavior_tree import decide
    from oblsk_negotiator.state import NegotiationState
    m = heuristic_interpret("As discussed on our call, we're aligned on next steps.")
    assert m.refers_to_call
    s = NegotiationState("c", "k", "t")
    d = decide(m, s, _vm(), ECON, CampaignContext())
    assert d.action == Action.ASK_HUMAN
    assert "call" in d.human_prompt.lower()


def test_call_request_proposes_call_once():
    from oblsk_negotiator.behavior_tree import CreatorMessage, decide
    from oblsk_negotiator.state import NegotiationState
    s = NegotiationState("c", "k", "t")
    ctx = CampaignContext(call_windows="Friday after 11 AM ET")
    decide(CreatorMessage(Intent.INTERESTED, raw_text="interested"),
           s, _vm(), ECON, ctx)                                   # offer opens
    m = heuristic_interpret("Would love to hop on a quick call this week to "
                            "discuss!")
    assert m.wants_call
    d = decide(m, s, _vm(), ECON, ctx)
    assert d.action == Action.PROPOSE_CALL
    assert "Friday after 11 AM ET" in d.answer_text
    assert d.human_prompt and "run the call" in d.human_prompt
    assert s.call_proposed
    # A second call-ish nudge doesn't loop another proposal.
    d2 = decide(heuristic_interpret("Sounds good, happy to grab a time."),
                s, _vm(), ECON, ctx)
    assert d2.action != Action.PROPOSE_CALL


def test_accept_with_nothing_on_table_opens_instead():
    # A first message misread (or genuinely eager) as "accepted" has no terms
    # to lock; the agent opens with a number instead of confirming nothing.
    from oblsk_negotiator.behavior_tree import CreatorMessage, decide
    from oblsk_negotiator.state import NegotiationState
    s = NegotiationState("c", "k", "t")
    d = decide(CreatorMessage(Intent.ACCEPTED, raw_text="sounds great, I'm in!"),
               s, _vm(), ECON, CampaignContext())
    assert d.action == Action.OPENING_FLAT
    # A yes to call-agreed terms the agent never saw asks the team instead.
    s2 = NegotiationState("c", "k", "t2")
    m = heuristic_interpret("As discussed on our call, let's do it.")
    assert m.intent == Intent.ACCEPTED and m.refers_to_call
    d2 = decide(m, s2, _vm(), ECON, CampaignContext())
    assert d2.action == Action.ASK_HUMAN


def test_budget_question_after_offer_restates_it():
    from oblsk_negotiator.behavior_tree import CreatorMessage, decide
    from oblsk_negotiator.state import NegotiationState
    s = NegotiationState("c", "k", "t")
    d0 = decide(CreatorMessage(Intent.INTERESTED, raw_text="interested"),
                s, _vm(), ECON, CampaignContext())
    d1 = decide(heuristic_interpret("Sorry, what was the rate again?"),
                s, _vm(), ECON, CampaignContext())
    assert d1.action == Action.HOLD_FIRM
    assert d1.deal.total_usd == d0.deal.total_usd


def test_llm_drafts_are_used_and_number_guard_holds(monkeypatch):
    # Prove the LLM path is live: when a model is available, the outbound
    # message is its draft (given the numbers survive), not the template —
    # and a draft that drops the calculator's price is rejected.
    from oblsk_negotiator import prose as prose_mod
    from oblsk_negotiator.behavior_tree import CreatorMessage, decide
    from oblsk_negotiator.state import NegotiationState
    s = NegotiationState("c", "k", "t")
    d = decide(CreatorMessage(Intent.INTERESTED, raw_text="interested"),
               s, _vm(), ECON, CampaignContext())
    total = f"${d.deal.total_usd:,.0f}"

    good = f"Hey! Loved the recent posts. We can do {total} for one video."
    monkeypatch.setattr(prose_mod.llm, "complete", lambda *a, **k: good)
    assert prose_mod.write_message(d, "Maya") == good

    bad = "Hey! Loved the recent posts. Let's talk numbers soon."
    monkeypatch.setattr(prose_mod.llm, "complete", lambda *a, **k: bad)
    fallback = prose_mod.write_message(d, "Maya", seed=1)
    assert fallback != bad and total in fallback   # template kept the price


def test_spar_requires_llm():
    from oblsk_negotiator.spar import spar
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        spar(_vm(), ECON, CampaignContext(), counterpart_profile="floor $2,600")


def test_spar_with_scripted_counterpart(monkeypatch):
    # Script the "Claude-played manager" so the loop runs offline: the agent
    # should open, then accept the in-ladder counter, producing a judgeable
    # transcript.
    import importlib
    spar_mod = importlib.import_module("oblsk_negotiator.spar")
    replies = iter([
        "Hi! I manage Jordan. Rates are $4,500 per video. Excited to connect.",
        "Appreciate the offer — could you do $2,000 per video?",
    ])

    def fake_complete(system, user, **kwargs):
        # Only the counterpart persona is scripted; prose falls back to templates.
        return next(replies, None) if "role-playing" in system else None

    monkeypatch.setattr(spar_mod.llm, "llm_available", lambda: True)
    monkeypatch.setattr(spar_mod.llm, "complete", fake_complete)
    result = spar_mod.spar(_vm(), ECON, CampaignContext(),
                           counterpart_profile="floor $2,000/video")
    senders = [t.sender for t in result.turns]
    assert senders == ["creator", "us", "creator", "us"]
    assert result.decisions[-1].action == Action.ACCEPT
    assert result.final_status == "accepted"
    assert "AGENT [accept]" in result.report()


def test_replay_sample_thread_offline():
    with open(SAMPLE, encoding="utf-8") as f:
        thread = f.read()
    report = replay_thread(thread, _vm(), ECON, CampaignContext())
    # Every creator turn produced a decision and a drafted message.
    assert len(report.steps) == 5
    assert all(s.agent_message for s in report.steps)
    # The first turn is a question; the last is the creator accepting.
    assert report.steps[0].decision.action == Action.ANSWER_QUESTION
    assert report.steps[-1].decision.action == Action.ACCEPT
    assert report.final_status == "accepted"
    # The team's actual replies were lined up for comparison.
    assert report.steps[0].human_reply is not None
    assert "step" in report.report()
