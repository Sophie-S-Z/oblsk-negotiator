"""Test suite. Run from the repo root: python -m pytest, or python tests/test_negotiator.py"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from oblsk_negotiator.ev_engine import (CreatorEconomics, Deal, fit_view_model,
                                        ev_distribution, synthetic_tier_prior)
from oblsk_negotiator.bundles import flat_offer, bundle_from_flat
from oblsk_negotiator.state import NegotiationState, AutonomyLevel, Phase
from oblsk_negotiator.behavior_tree import (decide, CreatorMessage, Intent,
                                           CampaignContext, Action,
                                           _requires_approval, _round_money,
                                           _flat_total, _bundle_total,
                                           _max_pay_per_video)
from oblsk_negotiator.pricing import price_ladder, authenticity
from oblsk_negotiator.qa import (CampaignBrief, classify_question_topic,
                                answer_from_brief, signals_ready_for_offer)
from oblsk_negotiator.creator_sim import SimCreator
from oblsk_negotiator.prose import render_template
from oblsk_negotiator.runner import run_negotiation, reject_action

ECON = CreatorEconomics(conversion_rate=0.0016, ltv_usd=80)
CTX = CampaignContext()


def _sim(floor=2450, opens="interest", ask=0.0, counter=0.6, bulk=0.06,
         max_videos=4, questions=None, walks=False, seed=0):
    return SimCreator(
        reservation_per_video=floor, opening_ask=ask, opens_with=opens,
        questions=questions or [], counter_ratio=counter, bulk_tolerance=bulk,
        max_videos=max_videos, walks_if_lowballed=walks, rng_seed=seed)


# A spread of temperaments for end-to-end runs: floor, how they open, how hard
# they counter, whether volume moves them, whether they walk when lowballed.
SIM_CONFIGS = [
    dict(floor=2450, opens="question", counter=0.6, bulk=0.06,
         questions=["What would the deliverables be for this?",
                    "And what's the timeline looking like?"]),
    dict(floor=2450, counter=0.35, bulk=0.0, max_videos=3),
    dict(floor=2450, counter=0.6, bulk=0.10, max_videos=5),
    dict(floor=2700, opens="ask", ask=3500, counter=0.45, bulk=0.0,
         max_videos=3, walks=True),
    dict(floor=1700, counter=0.7, bulk=0.08, max_videos=5),
]


def _views(median, sigma, n, hits=None, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.lognormal(np.log(median), sigma, n)
    return np.concatenate([p, hits]) if hits else p


VM = fit_view_model(_views(50000, 0.6, 24, [300000, 650000], seed=1))


def test_ev_determinism():
    d = flat_offer(2000, 2)
    assert ev_distribution(VM, d, ECON, seed=42).net_mean == \
           ev_distribution(VM, d, ECON, seed=42).net_mean


def test_flat_cost_is_fixed():
    assert abs(ev_distribution(VM, flat_offer(2400, 3), ECON, seed=1).cost_total - 2400) < 1e-6


def test_volume_monotonic_net():
    nets = [ev_distribution(VM, Deal(video_count=k, flat_per_video=800),
                            ECON, n_samples=20000, seed=3).net_mean for k in (1, 2, 3, 4)]
    assert all(x < y for x, y in zip(nets, nets[1:]))


def test_heavy_tail_detected():
    vm = fit_view_model(_views(40000, 0.5, 30, [2_000_000, 3_500_000], seed=2))
    assert vm.family == "pareto_lognorm"


def test_new_creator_fallback():
    prior = synthetic_tier_prior(40000, size=1500)
    assert fit_view_model([38000, 52000], min_posts=6, prior_samples=prior).family == "empirical_prior"


def test_bundle_raises_total_and_videos():
    flat = flat_offer(2000, 1)
    b = bundle_from_flat(flat)
    assert b.total_usd > flat.total_usd and b.video_count > flat.video_count


def test_offers_are_clean_numbers():
    # Opening flat rounds to $50; bundle to $100.
    assert _flat_total(VM, ECON, CTX) % 50 == 0
    assert _bundle_total(VM, ECON, CTX) % 100 == 0
    assert _round_money(2478.3, 50) == 2500.0


def test_state_roundtrip():
    s = NegotiationState("c", "k", "t", autonomy_level=AutonomyLevel.AUTONOMOUS)
    s.flat_offer_made = True
    s.phase = Phase.NEGOTIATING
    s.record_concession({"video_count": 3, "flat_per_video": 700}, 2100, 2400)
    s.record_approval("opening_flat", 2000, "approved", "ok")
    blob = s.to_json()
    r = NegotiationState.from_json(blob)
    assert r.to_json() == blob and r.autonomy_level == AutonomyLevel.AUTONOMOUS
    assert len(r.concession_history) == 1 and len(r.approval_history) == 1


def test_human_guard_pauses():
    import time
    s = NegotiationState("c", "k", "t")
    s.last_agent_message_at = time.time() - 50
    d = decide(CreatorMessage(Intent.NEGOTIATING, ask_total_usd=2000), s, VM, ECON, CTX,
               thread_last_sender="human", thread_last_ts=time.time())
    assert d.action == Action.PAUSE_HUMAN


def test_qa_topic_and_answer():
    assert classify_question_topic("what are the deliverables here?") == "deliverables"
    assert len(answer_from_brief("what are the deliverables here?", CampaignBrief())) > 0


def test_qa_budget_routes_to_offer():
    assert signals_ready_for_offer("how much does this pay?")


def test_opening_is_single_flat():
    s = NegotiationState("c", "k", "t")
    d = decide(CreatorMessage(Intent.INTERESTED, raw_text="interested"), s, VM, ECON, CTX)
    assert d.action == Action.OPENING_FLAT and d.deal.video_count == 1 and s.flat_offer_made


def test_flat_rejected_revises_then_bundles():
    # First rejection gets one revised single-video flat at our ROI ceiling; only
    # a second rejection moves to a bundle.
    s = NegotiationState("c", "k", "t")
    d0 = decide(CreatorMessage(Intent.INTERESTED), s, VM, ECON, CTX)
    d1 = decide(CreatorMessage(Intent.REJECTING, raw_text="too low"), s, VM, ECON, CTX)
    assert d1.action == Action.OPENING_FLAT and d1.deal.video_count == 1
    assert s.flat_revised and not s.bundle_offered
    assert d1.deal.total_usd >= d0.deal.total_usd            # a real concession up
    assert d1.deal.total_usd <= _max_pay_per_video(VM, ECON, CTX) + 50  # still ROI-capped
    d2 = decide(CreatorMessage(Intent.REJECTING, raw_text="still too low"),
                s, VM, ECON, CTX)
    assert d2.action == Action.ESCALATE_BUNDLE and d2.deal.video_count > 1
    assert s.bundle_offered


def test_extreme_ask_escalates_to_human():
    # An ask far above the walk-away ceiling goes straight to a person.
    s = NegotiationState("c", "k", "t")
    decide(CreatorMessage(Intent.INTERESTED), s, VM, ECON, CTX)        # flat
    ceiling = _max_pay_per_video(VM, ECON, CTX)
    d = decide(CreatorMessage(Intent.NEGOTIATING, ask_total_usd=ceiling * 3),
               s, VM, ECON, CTX)
    assert d.action == Action.ESCALATE_HUMAN and "x what" in d.rationale


def test_accept_never_crosses_walk_away():
    # Regression (screenshot bug): an ask above the walk-away ceiling but
    # inside willingness*accept_margin was accepted ($2,450 over a $2,372
    # ceiling). The margin closes thin gaps below the ceiling; it never
    # crosses it — at any rung of the ladder.
    from oblsk_negotiator.behavior_tree import _max_pay_per_video
    s = NegotiationState("c", "k", "t")
    walk = _max_pay_per_video(VM, ECON, CTX)
    decide(CreatorMessage(Intent.INTERESTED), s, VM, ECON, CTX)          # open
    over = walk * 1.03    # above the ceiling, within the old margin window
    d1 = decide(CreatorMessage(Intent.NEGOTIATING, ask_total_usd=over),
                s, VM, ECON, CTX)
    assert d1.action != Action.ACCEPT                                    # revise
    d2 = decide(CreatorMessage(Intent.NEGOTIATING, ask_total_usd=over),
                s, VM, ECON, CTX)
    assert d2.action != Action.ACCEPT                                    # bundle
    if d2.deal is not None:   # any offer we do make stays under the ceiling
        assert d2.deal.total_usd / d2.deal.video_count <= walk * 1.01
    # And an ask genuinely under the ceiling still closes.
    s2 = NegotiationState("c", "k", "t2")
    decide(CreatorMessage(Intent.INTERESTED), s2, VM, ECON, CTX)
    ok = decide(CreatorMessage(Intent.NEGOTIATING, ask_total_usd=walk * 0.95),
                s2, VM, ECON, CTX)
    assert ok.action == Action.ACCEPT


def test_accept_rationale_ceiling_is_consistent():
    # Regression: the accept rationale used to display raw willingness (the
    # target) as "our max", so an ask between target and target*margin was
    # accepted while the text claimed it was "at or below" a smaller number.
    # The stated ceiling must never be below the per-video rate accepted.
    import re
    from oblsk_negotiator.behavior_tree import _ladder
    # A low-variance creator makes target strictly below walk-away, the case
    # that exposed the gap.
    vm = fit_view_model(np.random.default_rng(3).lognormal(np.log(50000), 0.25, 40))
    lad = _ladder(vm, ECON, CTX)
    assert not lad.downside_limited            # target < walk-away here
    s = NegotiationState("c", "k", "t")
    decide(CreatorMessage(Intent.INTERESTED), s, vm, ECON, CTX)
    ask = round(lad.target * 1.03)             # above target, under the gate
    d = decide(CreatorMessage(Intent.NEGOTIATING, ask_total_usd=ask), s, vm, ECON, CTX)
    assert d.action == Action.ACCEPT
    dollars = [float(x.replace(",", "")) for x in re.findall(r"\$([\d,]+)", d.rationale)]
    per_video = d.deal.total_usd / d.deal.video_count
    ceiling = max(dollars)                      # the stated ceiling is the largest figure
    assert ceiling >= per_video - 1             # never accept above the stated ceiling
    assert ceiling <= lad.walk_away + 1         # and the ceiling itself respects walk-away


def test_ladder_orders_and_reproduces():
    lad = price_ladder(VM, ECON, CTX.pricing_policy())
    assert 0 < lad.anchor <= lad.target <= lad.walk_away
    assert lad.value_down <= lad.value_exp <= lad.value_up
    again = price_ladder(VM, ECON, CTX.pricing_policy())
    assert again.target == lad.target and again.walk_away == lad.walk_away


def test_ladder_cpm_cap_binds():
    from oblsk_negotiator.pricing import PricingPolicy
    tight = PricingPolicy(cpm_cap_usd=1.0)   # $1 per 1,000 views: far below ROI room
    lad = price_ladder(VM, ECON, tight)
    assert lad.binding == "CPM-cap"
    assert lad.walk_away < price_ladder(VM, ECON, PricingPolicy()).walk_away


def test_risk_discount_tightens_ceiling():
    from oblsk_negotiator.pricing import PricingPolicy
    base = price_ladder(VM, ECON, PricingPolicy())
    risky = price_ladder(VM, ECON, PricingPolicy(risk_discount=0.5))
    assert risky.walk_away < base.walk_away


def test_authenticity_flags_inflated_account():
    good = authenticity([{"views": 50000, "likes": 4000, "comments": 90}] * 10,
                        followers=100000)
    bad = authenticity([{"views": 1500, "likes": 300, "comments": 0}] * 10,
                       followers=900000)
    assert good["score"] > bad["score"]
    assert bad["score"] < 0.8


def test_recency_weighted_fit_tracks_recent_posts():
    # Old posts at 20k views, recent posts at 80k: the weighted fit should sit
    # far closer to the recent level than the unweighted one.
    now = 1_750_000_000
    views = [20000] * 12 + [80000] * 12
    ts = [now - 400 * 86400] * 12 + [now - 5 * 86400] * 12
    flat = fit_view_model(views)
    weighted = fit_view_model(views, timestamps=ts)
    assert weighted.median_views > flat.median_views * 1.3


def test_sponsored_haircut_scales_model():
    full = fit_view_model(_views(50000, 0.6, 24, seed=3))
    cut = fit_view_model(_views(50000, 0.6, 24, seed=3), sponsored_factor=0.8)
    assert abs(cut.median_views - 0.8 * full.median_views) < 1e-6


def test_prose_leads_with_one_offer():
    s = NegotiationState("c", "k", "t")
    d = decide(CreatorMessage(Intent.INTERESTED), s, VM, ECON, CTX)
    t1 = render_template(d, "Sam", seed=1)
    d2 = decide(CreatorMessage(Intent.REJECTING), s, VM, ECON, CTX)
    t2 = render_template(d2, "Sam", seed=1)
    assert all("Option A" not in t and "Option B" not in t for t in (t1, t2))


def test_approval_required_in_human_mode():
    s = NegotiationState("c", "k", "t", autonomy_level=AutonomyLevel.HUMAN_APPROVAL)
    assert decide(CreatorMessage(Intent.INTERESTED), s, VM, ECON, CTX).requires_approval


def test_autonomous_small_deal_no_approval():
    s = NegotiationState("c", "k", "t", autonomy_level=AutonomyLevel.AUTONOMOUS)
    assert _requires_approval(Action.OPENING_FLAT, 1500, s, CTX) is False


def test_human_rejection_escalates():
    o = run_negotiation(VM, ECON, CTX, _sim(bulk=0.10, max_videos=5, seed=1),
                        approver=reject_action(Action.OPENING_FLAT))
    assert o.human_rejected and o.status == "escalated"


def test_round_cap_escalates_human():
    o = run_negotiation(VM, ECON, CTX,
                        _sim(**SIM_CONFIGS[3], seed=4))
    assert o.status in ("accepted", "escalated")


def test_e2e_creators_close_or_escalate():
    for cfg in SIM_CONFIGS:
        for seed in range(8):
            o = run_negotiation(
                fit_view_model(_views(50000, 0.6, 22, [300000, 650000], seed=seed)),
                ECON, CTX, _sim(**cfg, seed=seed))
            assert o.status in ("accepted", "escalated", "rejected")
            if o.discount_from_ask is not None:
                assert o.discount_from_ask >= -0.01


def test_autonomous_closes_without_approvals():
    o = run_negotiation(VM, ECON, CTX, _sim(floor=1700, counter=0.7, bulk=0.08,
                                            max_videos=5, seed=1),
                        autonomy=AutonomyLevel.AUTONOMOUS)
    assert o.status == "accepted" and o.approvals_requested == 0


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); passed += 1; print(f"PASS {name}")
            except Exception as e:
                failed += 1; print(f"FAIL {name}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
