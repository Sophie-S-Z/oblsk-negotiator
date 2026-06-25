"""Oblsk creator-negotiation agent.

Two parts that meet at one handoff:
  the calculator (ev_engine) prices any deal as a distribution of net value,
  the negotiator (behavior_tree + prose + qa) runs the conversation.
"""

from .ev_engine import (CreatorEconomics, Deal, ViewModel, EVResult,
                        fit_view_model, ev_distribution, synthetic_tier_prior)
from .bundles import (flat_offer, bundle_from_flat, same_total_repackage,
                     add_non_price_sweetener)
from .state import (NegotiationState, Status, Phase, AutonomyLevel,
                   ConcessionRecord, ApprovalRecord)
from .qa import CampaignBrief, answer_from_brief, classify_question_topic
from .behavior_tree import (decide, CreatorMessage, Intent, CampaignContext,
                           Decision, Action)
from .prose import render_template, build_llm_prompt
from .creator_sim import SimCreator, Personality, PERSONAS
from .runner import (run_negotiation, batch_metrics, NegotiationOutcome,
                    BatchMetrics, always_approve, reject_action, ApprovalResult)

__all__ = [
    "CreatorEconomics", "Deal", "ViewModel", "EVResult", "fit_view_model",
    "ev_distribution", "synthetic_tier_prior",
    "flat_offer", "bundle_from_flat", "same_total_repackage",
    "add_non_price_sweetener",
    "NegotiationState", "Status", "Phase", "AutonomyLevel", "ConcessionRecord",
    "ApprovalRecord",
    "CampaignBrief", "answer_from_brief", "classify_question_topic",
    "decide", "CreatorMessage", "Intent", "CampaignContext", "Decision", "Action",
    "render_template", "build_llm_prompt",
    "SimCreator", "Personality", "PERSONAS",
    "run_negotiation", "batch_metrics", "NegotiationOutcome", "BatchMetrics",
    "always_approve", "reject_action", "ApprovalResult",
]
