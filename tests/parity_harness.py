"""
Characterization harness for the behavior tree.

Builds a fixed battery of negotiation scenarios (targeted ones that hit each
branch, plus many seeded-random ones) and runs every message through decide(),
recording the Decision and the resulting state at each step. The output is a
plain, fully-rounded structure that serializes to JSON.

Both sides of the parity check import this: generate_golden.py captures the
battery from the known-good tree into golden_decisions.json, and
test_tree_parity.py replays the same battery against the current tree and asserts
the two match. The battery is independent of the tree internals, so the message
sequences are identical on both sides and only the decide() implementation
varies.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np

from oblsk_negotiator.ev_engine import (CreatorEconomics, ViewModel, FormatSpec,
                                        FormatCatalog, deal_to_dict)
from oblsk_negotiator.rate_card import (RateCard, IG_REEL, IG_STORY, TIKTOK,
                                        YT_SHORT)
from oblsk_negotiator.behavior_tree import (decide, CreatorMessage, Intent,
                                           CampaignContext)
from oblsk_negotiator.qa import CampaignBrief
from oblsk_negotiator.state import NegotiationState, AutonomyLevel


# ---- fixtures (deterministic) ----------------------------------------------

def _make_vm() -> ViewModel:
    rng = np.random.default_rng(1)
    mu = float(np.log(50000))
    return ViewModel(family="lognormal", params={"mu": mu, "sigma": 0.6},
                     n_posts_fit=24, median_views=50000.0, notes="parity vm")


def _vm_for(median: float) -> ViewModel:
    return ViewModel(family="lognormal", params={"mu": float(np.log(median)),
                                                 "sigma": 1.0},
                     n_posts_fit=20, median_views=median,
                     notes=f"reach median {median:,.0f}")


VM = _make_vm()
ECON = CreatorEconomics(conversion_rate=0.0016, ltv_usd=80)
ECON_MF = CreatorEconomics(conversion_rate=0.001, ltv_usd=50.0)

CATALOG = FormatCatalog({
    IG_REEL:  FormatSpec(IG_REEL,  _vm_for(1_200_000), ECON_MF),
    IG_STORY: FormatSpec(IG_STORY, _vm_for(800_000),  ECON_MF),
    TIKTOK:   FormatSpec(TIKTOK,   _vm_for(1_500_000), ECON_MF),
    YT_SHORT: FormatSpec(YT_SHORT, _vm_for(900_000),  ECON_MF),
})
RATE_CARD = RateCard(prices={IG_REEL: 16500, IG_STORY: 5850, TIKTOK: 16750,
                             YT_SHORT: 3500}, creator="parity creator")

BRIEF = CampaignBrief()
BRIEF_MF = CampaignBrief(primary_format=IG_REEL,
                         allowed_formats=[IG_REEL, IG_STORY, TIKTOK])


# ---- scenario construction --------------------------------------------------

def _q(text):
    return CreatorMessage(intent=Intent.QUESTION, raw_text=text)


def _interested():
    return CreatorMessage(intent=Intent.INTERESTED, raw_text="interested")


def _ask(total, videos=1):
    return CreatorMessage(intent=Intent.NEGOTIATING, ask_total_usd=total,
                          ask_video_count=videos, raw_text=f"could you do ${total}")


def _rejecting():
    return CreatorMessage(intent=Intent.REJECTING, raw_text="too low for me")


def _accepted():
    return CreatorMessage(intent=Intent.ACCEPTED, raw_text="let's do it")


def _targeted_scenarios():
    """Hand-built sequences, one per branch we want to guarantee is exercised."""
    return [
        # pre-offer rejection -> soft close
        {"name": "soft_close", "msgs": [_rejecting()]},
        # open flat from interest
        {"name": "open_flat", "msgs": [_interested()]},
        # budget question -> open flat
        {"name": "budget_q", "msgs": [_q("what does this pay?")]},
        # one then two answered questions (nudges), then interest -> flat
        {"name": "qa_then_offer",
         "msgs": [_q("what are the deliverables?"), _q("and the timeline?"),
                  _interested()]},
        # three questions -> proactive offer on the third
        {"name": "qa_threshold",
         "msgs": [_q("what are the deliverables?"), _q("timeline?"),
                  _q("usage rights?")]},
        # flat rejected -> revised flat; high counters -> bundle -> escalate
        {"name": "flat_revise_bundle",
         "msgs": [_interested(), _rejecting(), _ask(4000), _ask(4000)]},
        # full ladder incl. sweetener, needs a higher round budget than the default
        {"name": "sweetener_ladder", "max_rounds": 5,
         "msgs": [_interested(), _rejecting(), _ask(4000), _ask(4000),
                  _ask(4000)]},
        # two straight rejections with no counter -> one salvage, then soft close
        {"name": "reject_twice_soft_close",
         "msgs": [_interested(), _rejecting(), _rejecting()]},
        # low counter accepted
        {"name": "accept_low_counter", "msgs": [_interested(), _ask(1500)]},
        # extreme ask -> escalate human
        {"name": "extreme_ask", "msgs": [_interested(), _ask(200000)]},
        # accept on explicit confirmation after a flat
        {"name": "accept_on_confirm", "msgs": [_interested(), _accepted()]},
        # multi-format: flat, counter, composed bundle, accept
        {"name": "multiformat_close", "mf": True,
         "msgs": [_interested(), _ask(21200), _accepted()]},
        # multi-format: flat then immediate high counter -> compose bundle
        {"name": "multiformat_bundle", "mf": True,
         "msgs": [_interested(), _ask(30000)]},
        # a teammate steps into the thread -> pause
        {"name": "human_pause", "human": True, "msgs": [_interested()]},
    ]


def _random_scenarios(n=160, seed=7):
    rng = random.Random(seed)
    intents = ["question", "interested", "ask", "rejecting", "accepted"]
    questions = ["what are the deliverables?", "timeline?", "usage rights?",
                 "do I keep creative control?", "what does it pay?",
                 "any exclusivity?", "how many revisions?"]
    out = []
    for i in range(n):
        length = rng.randint(1, 6)
        msgs = []
        for _ in range(length):
            kind = rng.choice(intents)
            if kind == "question":
                msgs.append(_q(rng.choice(questions)))
            elif kind == "interested":
                msgs.append(_interested())
            elif kind == "ask":
                msgs.append(_ask(rng.choice([1200, 1800, 2400, 3200, 5000,
                                             9000, 21200, 50000, 200000]),
                                 rng.choice([1, 1, 2, 3])))
            elif kind == "rejecting":
                msgs.append(_rejecting())
            else:
                msgs.append(_accepted())
        out.append({"name": f"rand_{i}", "mf": bool(rng.getrandbits(1)),
                    "msgs": msgs, "autonomous": bool(rng.getrandbits(1))})
    return out


def build_scenarios():
    return _targeted_scenarios() + _random_scenarios()


# ---- running and serialization ----------------------------------------------

def _round_floats(obj, ndigits=4):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def _ev_digest(ev):
    if ev is None:
        return None
    return _round_floats({
        "net_mean": ev.net_mean, "roi_mean": ev.roi_mean,
        "prob_net_positive": ev.prob_net_positive, "cost_total": ev.cost_total,
        "revenue_mean": ev.revenue_mean})


def _deal_digest(deal):
    return None if deal is None else _round_floats(deal_to_dict(deal))


def _step_record(decision, state):
    return {
        "decision": {
            "action": decision.action.value,
            "requires_approval": decision.requires_approval,
            "non_price_lever": decision.non_price_lever,
            "answer_text": decision.answer_text,
            "rationale": decision.rationale,
            "deal": _deal_digest(decision.deal),
            "ev": _ev_digest(decision.ev),
            "reserve_deal": _deal_digest(decision.reserve_deal),
            "reserve_ev": _ev_digest(decision.reserve_ev),
        },
        "state": {
            "phase": state.phase.value,
            "round_count": state.round_count,
            "questions_answered": state.questions_answered,
            "flat_offer_made": state.flat_offer_made,
            "flat_revised": state.flat_revised,
            "bundle_offered": state.bundle_offered,
            "status": state.status.value,
            "creator_first_ask": _round_floats(state.creator_first_ask),
            "final_total": _round_floats(state.final_total),
            "final_deal": _round_floats(state.final_deal),
        },
    }


def run_battery():
    """Run every scenario through decide(), returning a list of records, one per
    scenario, each holding the per-step decision and state snapshots."""
    records = []
    for sc in build_scenarios():
        mf = sc.get("mf", False)
        autonomy = (AutonomyLevel.AUTONOMOUS if sc.get("autonomous")
                    else AutonomyLevel.HUMAN_APPROVAL)
        state = NegotiationState("c", "k", "t", max_rounds=sc.get("max_rounds", 3),
                                 autonomy_level=autonomy)
        brief = BRIEF_MF if mf else BRIEF
        catalog = CATALOG if mf else None
        card = RATE_CARD if mf else None
        sender = "human" if sc.get("human") else "creator"
        ts = 1e9 if sc.get("human") else 0.0
        steps = []
        for msg in sc["msgs"]:
            decision = decide(msg, state, VM, ECON, CampaignContext(),
                              brief=brief, catalog=catalog, rate_card=card,
                              thread_last_sender=sender, thread_last_ts=ts)
            steps.append(_step_record(decision, state))
        records.append({"name": sc["name"], "steps": steps})
    return records
