"""
Event-log tests.

The point of the log is that the negotiation row is reconstructable from it. So
the core test runs real negotiations across a spread of creator temperaments and
seeds, folds the log, and asserts the fold agrees with the outcome the runner
reported: same rounds, same questions answered, same status, same final total.
A second test drives a multi-format deal to a bundle close and checks the
structural flags the fold derives. The rest check append-only integrity and
JSON round-tripping.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from oblsk_negotiator.events import EventLog, fold_events, DECISION
from oblsk_negotiator.runner import run_negotiation
from oblsk_negotiator.behavior_tree import CampaignContext
from oblsk_negotiator.creator_sim import SimCreator
from oblsk_negotiator.ev_engine import (CreatorEconomics, FormatSpec, FormatCatalog,
                                        fit_view_model)
from oblsk_negotiator.rate_card import RateCard, IG_REEL, IG_STORY, TIKTOK
from oblsk_negotiator.qa import CampaignBrief

CTX = CampaignContext()
ECON = CreatorEconomics(conversion_rate=0.0016, ltv_usd=80)

SIM_CONFIGS = [
    dict(reservation_per_video=2450, opens_with="question",
         questions=["What would the deliverables be for this?",
                    "And what's the timeline looking like?"],
         counter_ratio=0.6, bulk_tolerance=0.06),
    dict(reservation_per_video=2450, counter_ratio=0.35, bulk_tolerance=0.0,
         max_videos=3),
    dict(reservation_per_video=2450, counter_ratio=0.6, bulk_tolerance=0.10,
         max_videos=5),
    dict(reservation_per_video=2700, opening_ask=3500, opens_with="ask",
         counter_ratio=0.45, bulk_tolerance=0.0, max_videos=3,
         walks_if_lowballed=True),
    dict(reservation_per_video=1700, counter_ratio=0.7, bulk_tolerance=0.08,
         max_videos=5),
]


def _vm(median, seed=0):
    rng = np.random.default_rng(seed)
    return fit_view_model(rng.lognormal(np.log(median), 0.6, 24).tolist())


def test_fold_matches_outcome_across_creators():
    vm = _vm(50000)
    for i, cfg in enumerate(SIM_CONFIGS):
        for seed in range(6):
            o = run_negotiation(vm, ECON, CTX, SimCreator(**cfg, rng_seed=seed))
            f = fold_events(o.event_log)
            assert f["round_count"] == o.rounds, (i, seed, "rounds")
            assert f["questions_answered"] == o.questions_answered, (i, seed, "qa")
            assert f["status"] == o.status, (i, seed, "status")
            assert f["final_total"] == o.final_total, (i, seed, "final_total")


def test_fold_matches_outcome_multiformat():
    # The fold is format-blind (a bundle is a bundle), but check it holds on a
    # multi-format run end to end too.
    econ_mf = CreatorEconomics(conversion_rate=0.001, ltv_usd=50.0)
    catalog = FormatCatalog({
        IG_REEL:  FormatSpec(IG_REEL,  _vm(1_200_000), econ_mf),
        IG_STORY: FormatSpec(IG_STORY, _vm(800_000),  econ_mf),
        TIKTOK:   FormatSpec(TIKTOK,   _vm(1_500_000), econ_mf),
    })
    card = RateCard(prices={IG_REEL: 16500, IG_STORY: 5850, TIKTOK: 16750},
                    creator="mf creator")
    brief = CampaignBrief(primary_format=IG_REEL,
                          allowed_formats=[IG_REEL, IG_STORY, TIKTOK])
    for seed in range(4):
        o = run_negotiation(_vm(1_200_000), econ_mf, CTX,
                            SimCreator(reservation_per_video=2450,
                                       counter_ratio=0.6, bulk_tolerance=0.10,
                                       max_videos=5, rng_seed=seed),
                            brief=brief, catalog=catalog, rate_card=card)
        f = fold_events(o.event_log)
        assert f["round_count"] == o.rounds
        assert f["status"] == o.status
        assert f["final_total"] == o.final_total


def test_fold_reconstructs_structural_flags():
    # A creator whose floor sits above our target rejects through the whole
    # ladder: open, revised flat, bundle. The fold should recover each of those
    # structural facts from the log alone.
    o = run_negotiation(_vm(50000), ECON, CTX,
                        SimCreator(reservation_per_video=4200, counter_ratio=0.2,
                                   bulk_tolerance=0.0, max_videos=4, rng_seed=0))
    actions = [e.data["action"] for e in o.event_log if e.kind == DECISION]
    assert actions.count("opening_flat") >= 2 and "escalate_bundle" in actions
    f = fold_events(o.event_log)
    assert f["flat_offer_made"]
    assert f["flat_revised"]        # the second flat is the revised offer
    assert f["bundle_offered"]
    assert f["round_count"] == o.rounds
    assert f["status"] == o.status


def test_log_is_append_only_and_serializes():
    o = run_negotiation(_vm(50000), ECON, CTX,
                        SimCreator(reservation_per_video=2700, opening_ask=3500,
                                   opens_with="ask", counter_ratio=0.45,
                                   bulk_tolerance=0.0, max_videos=3,
                                   walks_if_lowballed=True, rng_seed=1))
    events = o.event_log.events
    assert len(events) >= 2
    # Sequence numbers are contiguous from zero and timestamps never go backwards:
    # the marks of an append-only log.
    assert [e.seq for e in events] == list(range(len(events)))
    assert all(b.ts >= a.ts for a, b in zip(events, events[1:]))
    # At least one decision was recorded.
    assert any(e.kind == DECISION for e in events)
    # Round-trip through JSON preserves the fold.
    restored = EventLog.from_json(o.event_log.to_json())
    assert fold_events(restored) == fold_events(o.event_log)
