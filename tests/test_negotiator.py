"""Test suite. Run from the repo root: python -m pytest, or python tests/test_negotiator.py"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from oblsk_negotiator.ev_engine import (CreatorEconomics, Deal, ViewModel,
                                        fit_view_model, ev_distribution,
                                        synthetic_tier_prior)
from oblsk_negotiator.bundles import flat_offer, bundle_from_flat
from oblsk_negotiator.state import NegotiationState, AutonomyLevel, Phase
from oblsk_negotiator.behavior_tree import (decide, CreatorMessage, Intent,
                                           CampaignContext, Action,
                                           _requires_approval, _round_money,
                                           _flat_total, _bundle_total,
                                           _max_pay_per_video)
from oblsk_negotiator.pricing import price_ladder, authenticity
from oblsk_negotiator.qa import (CampaignBrief, classify_question_topic,
                                answer_from_brief, signals_ready_for_offer,
                                can_answer)
from oblsk_negotiator.creator_sim import SimCreator
from oblsk_negotiator.prose import render_template, write_message
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


def test_qa_content_style_is_answered_not_escalated():
    # "style of content" questions used to fall into `general` and get paged to
    # a human; they now classify as content_style and answer from the brief.
    for q in ("what style of content are you looking for?",
              "what's the vibe / editing style you want?",
              "any particular tone or aesthetic in mind?"):
        assert classify_question_topic(q) == "content_style", q
        assert can_answer(q, CampaignBrief()) is True, q
    ans = answer_from_brief("what style of content are you looking for?", CampaignBrief())
    assert "style" in ans.lower() and "$" not in ans

    # Guard against collisions with neighbouring topics.
    assert classify_question_topic("do i get a round of revisions?") == "revisions"
    assert classify_question_topic("how many videos do i make?") == "deliverables"


def test_qa_budget_routes_to_offer():
    assert signals_ready_for_offer("how much does this pay?")


def test_opening_is_single_flat():
    s = NegotiationState("c", "k", "t")
    d = decide(CreatorMessage(Intent.INTERESTED, raw_text="interested"), s, VM, ECON, CTX)
    assert d.action == Action.OPENING_FLAT and d.deal.video_count == 1 and s.flat_offer_made


def test_flat_rejected_revises_then_bundles():
    # First rejection gets one revised single-video flat at our ROI ceiling; a
    # counter still above it moves to a bundle.
    s = NegotiationState("c", "k", "t")
    d0 = decide(CreatorMessage(Intent.INTERESTED), s, VM, ECON, CTX)
    d1 = decide(CreatorMessage(Intent.REJECTING, raw_text="too low"), s, VM, ECON, CTX)
    assert d1.action == Action.OPENING_FLAT and d1.deal.video_count == 1
    assert s.flat_revised and not s.bundle_offered
    assert d1.deal.total_usd >= d0.deal.total_usd            # a real concession up
    assert d1.deal.total_usd <= _max_pay_per_video(VM, ECON, CTX)  # still ROI-capped
    high = _max_pay_per_video(VM, ECON, CTX) * 1.5           # too high, not extreme
    d2 = decide(CreatorMessage(Intent.NEGOTIATING, ask_total_usd=high,
                               raw_text="I really need more"), s, VM, ECON, CTX)
    assert d2.action == Action.ESCALATE_BUNDLE and d2.deal.video_count > 1
    assert s.bundle_offered


def test_second_straight_rejection_soft_closes():
    # One salvage attempt after a rejection with no counter; a second straight
    # rejection closes gently instead of pushing a bundle at someone who is done.
    from oblsk_negotiator.state import Status
    s = NegotiationState("c", "k", "t")
    decide(CreatorMessage(Intent.INTERESTED), s, VM, ECON, CTX)
    d1 = decide(CreatorMessage(Intent.REJECTING, raw_text="not interested"),
                s, VM, ECON, CTX)
    assert d1.action == Action.OPENING_FLAT                  # the one salvage
    d2 = decide(CreatorMessage(Intent.REJECTING, raw_text="again: not interested"),
                s, VM, ECON, CTX)
    assert d2.action == Action.SOFT_CLOSE and s.status == Status.REJECTED
    # A counter in between resets the streak.
    s2 = NegotiationState("c", "k", "t2")
    decide(CreatorMessage(Intent.INTERESTED), s2, VM, ECON, CTX)
    decide(CreatorMessage(Intent.REJECTING, raw_text="too low"), s2, VM, ECON, CTX)
    decide(CreatorMessage(Intent.NEGOTIATING, ask_total_usd=4000), s2, VM, ECON, CTX)
    assert s2.consecutive_rejections == 0


def test_rounding_never_crosses_walk_away():
    # Rounding to clean numbers floors at the ceiling instead of crossing it.
    from oblsk_negotiator.behavior_tree import _revised_flat_total, _bundle_total
    walk = _max_pay_per_video(VM, ECON, CTX)
    assert _revised_flat_total(VM, ECON, CTX) <= walk
    assert _bundle_total(VM, ECON, CTX, ask_per_video=walk * 1.9) / CTX.target_videos \
        <= walk


def test_extreme_ask_escalates_to_human():
    # An ask far above the walk-away ceiling goes straight to a person.
    s = NegotiationState("c", "k", "t")
    decide(CreatorMessage(Intent.INTERESTED), s, VM, ECON, CTX)        # flat
    ceiling = _max_pay_per_video(VM, ECON, CTX)
    d = decide(CreatorMessage(Intent.NEGOTIATING, ask_total_usd=ceiling * 3),
               s, VM, ECON, CTX)
    assert d.action == Action.ESCALATE_HUMAN and "x what" in d.rationale


def test_implausibly_low_ask_asks_human_not_accept():
    # A mis-parsed $0/near-zero ask used to sail through ask_acceptable and lock
    # the deal at that number. It now goes to a person to confirm the figure.
    s = NegotiationState("c", "k", "t")
    decide(CreatorMessage(Intent.INTERESTED), s, VM, ECON, CTX)          # open
    d = decide(CreatorMessage(Intent.NEGOTIATING, ask_total_usd=0.0,
                              ask_video_count=1, raw_text="sounds good"),
               s, VM, ECON, CTX)
    assert d.action == Action.ASK_HUMAN and d.human_prompt
    d2 = decide(CreatorMessage(Intent.NEGOTIATING, ask_total_usd=50.0,
                               ask_video_count=1, raw_text="can you do 50"),
                s, VM, ECON, CTX)
    assert d2.action == Action.ASK_HUMAN
    # A genuine low-but-plausible counter (above the floor) still accepts.
    s2 = NegotiationState("c", "k", "t2")
    decide(CreatorMessage(Intent.INTERESTED), s2, VM, ECON, CTX)
    ok = decide(CreatorMessage(Intent.NEGOTIATING, ask_total_usd=1500.0,
                               ask_video_count=1), s2, VM, ECON, CTX)
    assert ok.action == Action.ACCEPT


def test_bundle_offer_clamped_to_prior_concession():
    # Safety net: if a drifted recompute would price a bundle far above what we
    # already offered (the $2,200-after-$400 shape), the per-video rate is pinned
    # to the last concession plus the anomaly margin.
    s = NegotiationState("c", "k", "t")
    decide(CreatorMessage(Intent.INTERESTED), s, VM, ECON, CTX)          # open
    decide(CreatorMessage(Intent.REJECTING, raw_text="too low"), s, VM, ECON, CTX)  # revise
    # Simulate a corrupted/low recorded concession (a misread we should not
    # bid away from): our last position reads as $400 for one video.
    s.concession_history[-1].offered_total = 400.0
    s.concession_history[-1].offered_video_count = 1
    d = decide(CreatorMessage(Intent.NEGOTIATING, ask_total_usd=4000,
                              raw_text="I need a lot more"), s, VM, ECON, CTX)
    assert d.action == Action.ESCALATE_BUNDLE
    per_video = d.deal.total_usd / d.deal.video_count
    ceiling = 400.0 * (1.0 + CTX.max_concession_step)
    assert per_video <= ceiling + CTX.bundle_money_step   # pinned, allowing rounding


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


# A thin, low-conversion creator whose ROI-justified target falls below the
# opening floor — the exact shape that used to open at $0/video.
_THIN_VM = fit_view_model([50, 40, 60, 45, 55, 30])
_THIN_ECON = CreatorEconomics(conversion_rate=0.001, ltv_usd=20)


def test_sub_minimum_escalates_by_default_never_opens_at_zero():
    from oblsk_negotiator.behavior_tree import _ladder
    lad = _ladder(_THIN_VM, _THIN_ECON, CTX)
    assert lad.target < CTX.min_offer_usd            # precondition: thin creator
    s = NegotiationState("c", "k", "t")
    d = decide(CreatorMessage(Intent.INTERESTED, raw_text="sounds interesting"),
               s, _THIN_VM, _THIN_ECON, CTX)
    assert d.action == Action.ESCALATE_HUMAN
    assert "floor" in (s.escalation_reason or "").lower()
    # The bug symptom — a $0 opening flat — is never produced.
    assert d.action != Action.OPENING_FLAT


def test_sub_minimum_floor_mode_opens_at_least_at_the_floor():
    ctx = CampaignContext(sub_minimum_policy="floor")
    s = NegotiationState("c", "k", "t")
    d = decide(CreatorMessage(Intent.INTERESTED, raw_text="sounds interesting"),
               s, _THIN_VM, _THIN_ECON, ctx)
    assert d.action == Action.OPENING_FLAT
    assert d.deal.total_usd >= ctx.min_offer_usd     # the floor actually holds


def test_floor_mode_bundle_never_undercuts_the_opening_floor():
    # Regression: floor mode opened at the min_offer floor but then a rejected
    # flat escalated to a *cheaper* bundle (priced off the tiny target), undercutting
    # our own opening. The bundle per-video must stay at/above the floor.
    ctx = CampaignContext(sub_minimum_policy="floor")
    s = NegotiationState("c", "k", "t")
    d0 = decide(CreatorMessage(Intent.INTERESTED), s, _THIN_VM, _THIN_ECON, ctx)
    open_pv = d0.deal.total_usd / d0.deal.video_count
    d1 = decide(CreatorMessage(Intent.REJECTING, raw_text="too low"),
                s, _THIN_VM, _THIN_ECON, ctx)     # revised flat
    d2 = decide(CreatorMessage(Intent.REJECTING, raw_text="still too low"),
                s, _THIN_VM, _THIN_ECON, ctx)     # bundle
    for d in (d1, d2):
        if d.deal is not None:
            assert d.deal.total_usd / d.deal.video_count >= ctx.min_offer_usd - 1
    assert open_pv >= ctx.min_offer_usd - 1


def test_sub_minimum_policy_validated():
    import pytest
    with pytest.raises(ValueError):
        CampaignContext(sub_minimum_policy="nonsense")


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


def test_all_outbound_drafts_require_human_send():
    """No creator-facing text ever auto-sends: every outbound action (and
    ESCALATE_HUMAN) requires approval regardless of autonomy level or total;
    only the internal ASK_HUMAN/PAUSE_HUMAN moves proceed on their own."""
    outbound = [Action.ANSWER_QUESTION, Action.OPENING_FLAT, Action.ESCALATE_BUNDLE,
                Action.ADD_SWEETENER, Action.HOLD_FIRM, Action.PROPOSE_CALL,
                Action.ACCEPT, Action.SOFT_CLOSE]
    for level in (AutonomyLevel.HUMAN_APPROVAL, AutonomyLevel.AUTONOMOUS):
        s = NegotiationState("c", "k", "t", autonomy_level=level)
        for action in outbound:
            for total in (None, 50, 1500, 10_000_000):
                assert _requires_approval(action, total, s, CTX) is True, (action, total, level)
        assert _requires_approval(Action.ESCALATE_HUMAN, None, s, CTX) is True
        assert _requires_approval(Action.ASK_HUMAN, None, s, CTX) is False
        assert _requires_approval(Action.PAUSE_HUMAN, None, s, CTX) is False


def test_outbound_decision_still_carries_a_draft():
    """The human gets a pre-filled box, not an empty one: an auto-mode opening
    still produces both a draft and requires_approval=True."""
    s = NegotiationState("c", "k", "t", autonomy_level=AutonomyLevel.AUTONOMOUS)
    d = decide(CreatorMessage(Intent.INTERESTED, raw_text="interested"), s, VM, ECON, CTX)
    assert d.action == Action.OPENING_FLAT and d.requires_approval is True
    assert len(write_message(d, creator_name="Ana")) > 0


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


def test_autonomous_still_requests_approval_on_every_draft():
    # The autonomy level no longer bypasses sending: even in AUTONOMOUS mode
    # every creator-facing draft is put up for human approval before it goes
    # out. It still closes (the approver here approves), but approvals are
    # requested rather than skipped.
    o = run_negotiation(VM, ECON, CTX, _sim(floor=1700, counter=0.7, bulk=0.08,
                                            max_videos=5, seed=1),
                        autonomy=AutonomyLevel.AUTONOMOUS)
    assert o.status == "accepted" and o.approvals_requested > 0


def test_composed_bundle_accept_above_singleformat_walkaway():
    """Composed-bundle exception: the bundle's own blended per-video rate is its
    own ceiling (it prices off each format's own reach), so a counter under the
    bundle's bundled-rate plus the accept margin closes, even though it sits
    well above the single-format walk_away. Without the exception, the cap
    would refuse the deal and we'd lose it."""
    from oblsk_negotiator.ev_engine import (FormatCatalog, FormatSpec,
                                            deal_to_dict)
    from oblsk_negotiator.rate_card import RateCard, IG_REEL, IG_STORY, TIKTOK
    from oblsk_negotiator.bundles import compose_bundle

    # High-reach formats so the composed bundle clears the 3x ROI hurdle at a
    # per-video rate well above the single-format VM's walk_away.
    def _format_vm(median):
        return ViewModel(family="lognormal",
                         params={"mu": float(np.log(median)), "sigma": 1.0},
                         n_posts_fit=20, median_views=median,
                         notes=f"reach median {median:,.0f}")
    catalog = FormatCatalog({
        IG_REEL:  FormatSpec(IG_REEL,  _format_vm(1_200_000), ECON),
        IG_STORY: FormatSpec(IG_STORY, _format_vm(800_000),  ECON),
        TIKTOK:   FormatSpec(TIKTOK,   _format_vm(1_500_000), ECON),
    })
    rate_card = RateCard(prices={IG_REEL: 16500, IG_STORY: 5850, TIKTOK: 16750},
                         creator="test creator")
    composed = compose_bundle(rate_card, catalog, roi_hurdle=CTX.roi_target,
                              primary_fmt=IG_REEL,
                              allowed_formats=[IG_REEL, IG_STORY, TIKTOK])
    assert composed is not None and composed.video_count == 3
    composed_pv = composed.flat_per_video

    # Sanity: this test only exercises the exception if the composed bundle's
    # blended rate is genuinely above the single-format walk_away.
    singleformat_walk = _max_pay_per_video(VM, ECON, CTX)
    assert composed_pv > singleformat_walk, \
        "test setup: composed_pv must exceed walk_away to exercise the exception"

    # State: a composed bundle is already on the table (the negotiation has
    # flattened the standalone offer and escalated into a multi-format package).
    s = NegotiationState("c", "k", "t")
    s.flat_offer_made = s.flat_revised = s.bundle_offered = True
    s.phase = Phase.NEGOTIATING
    s.round_count = 3
    s.record_concession(deal_to_dict(composed), composed.total_usd, None,
                        video_count=composed.video_count)

    ask_pv = composed_pv * 1.04            # just under accept_margin (1.05)
    ask_total = ask_pv * composed.video_count
    msg = CreatorMessage(intent=Intent.NEGOTIATING,
                         ask_total_usd=ask_total,
                         ask_video_count=composed.video_count,
                         raw_text=f"Could you do ${ask_pv:,.0f} a video?")
    d = decide(msg, s, VM, ECON, CTX, catalog=catalog, rate_card=rate_card)
    assert d.action == Action.ACCEPT, (
        f"composed-bundle on table + counter under bundle-pv*1.05 should "
        f"accept, got {d.action.value}; ask_pv={ask_pv:,.0f}, "
        f"composed_pv={composed_pv:,.0f}, walk_away={singleformat_walk:,.0f}")
    assert d.deal.video_count == composed.video_count
    assert d.deal.total_usd == ask_total   # locked at their ask
    # And the symmetry: above the bundle margin escalates, doesn't silently accept.
    high = composed_pv * 1.10
    d_high = decide(CreatorMessage(intent=Intent.NEGOTIATING,
                                   ask_total_usd=high * composed.video_count,
                                   ask_video_count=composed.video_count,
                                   raw_text="too high"),
                    s, VM, ECON, CTX, catalog=catalog, rate_card=rate_card)
    assert d_high.action != Action.ACCEPT


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
