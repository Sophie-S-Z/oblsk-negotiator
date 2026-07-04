"""Oblsk creator-negotiation agent.

Two parts that meet at one handoff:
  the calculator (ev_engine) prices any deal as a distribution of net value,
  the negotiator (behavior_tree + prose + qa) runs the conversation.
"""

from .ev_engine import (CreatorEconomics, Deal, ViewModel, EVResult,
                        fit_view_model, ev_distribution, synthetic_tier_prior)
from .pricing import PricingPolicy, PriceLadder, price_ladder, authenticity
from .bundles import (flat_offer, bundle_from_flat, same_total_repackage,
                     add_non_price_sweetener)
from .state import (NegotiationState, Status, Phase, AutonomyLevel,
                   ConcessionRecord, ApprovalRecord)
from .qa import (CampaignBrief, answer_from_brief, answer_question,
                classify_question_topic)
from .behavior_tree import (decide, CreatorMessage, Intent, CampaignContext,
                           Decision, Action)
from .prose import render_template, write_message, build_llm_prompt
from .creator_sim import SimCreator
from .campaign import Campaign, load_campaign
from .replay import (replay_thread, parse_thread, parse_transcript,
                    parse_email_thread, interpret_message, ReplayReport)
from .spar import spar, SparResult
from .runner import (run_negotiation, batch_metrics, NegotiationOutcome,
                    BatchMetrics, always_approve, reject_action, ApprovalResult)

__all__ = [
    "CreatorEconomics", "Deal", "ViewModel", "EVResult", "fit_view_model",
    "ev_distribution", "synthetic_tier_prior",
    "PricingPolicy", "PriceLadder", "price_ladder", "authenticity",
    "flat_offer", "bundle_from_flat", "same_total_repackage",
    "add_non_price_sweetener",
    "NegotiationState", "Status", "Phase", "AutonomyLevel", "ConcessionRecord",
    "ApprovalRecord",
    "CampaignBrief", "answer_from_brief", "answer_question",
    "classify_question_topic",
    "decide", "CreatorMessage", "Intent", "CampaignContext", "Decision", "Action",
    "render_template", "write_message", "build_llm_prompt",
    "SimCreator",
    "Campaign", "load_campaign",
    "replay_thread", "parse_thread", "parse_transcript", "parse_email_thread",
    "interpret_message", "ReplayReport",
    "spar", "SparResult",
    "run_negotiation", "batch_metrics", "NegotiationOutcome", "BatchMetrics",
    "always_approve", "reject_action", "ApprovalResult",
]
