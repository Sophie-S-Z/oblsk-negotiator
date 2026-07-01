"""
The loop, the approval gate, and the metrics.

Every message the agent wants to send is a proposal. When it needs approval, the
runner calls an approver (a stand-in for the human reviewer) which approves or
rejects it. Approved messages go out; a rejected one does not, and the thread is
handed to a person. Swap the approver for always_approve, or put the campaign in
autonomous mode, and the human comes out of the loop with no other change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
import time
import numpy as np

from .ev_engine import (CreatorEconomics, ViewModel, ev_distribution, ev_bundle,
                        Deal, Bundle, FormatCatalog, deal_from_dict)
from .state import NegotiationState, Status, AutonomyLevel
from .behavior_tree import decide, Intent, CampaignContext, Decision, Action
from .creator_sim import SimCreator
from .prose import render_template
from .qa import CampaignBrief
from .rate_card import RateCard
from .events import (EventLog, MESSAGE_RECEIVED, DECISION, APPROVAL, SENT,
                     HUMAN_REJECTED)


@dataclass
class ApprovalResult:
    outcome: str                # approved | rejected
    note: str = ""


Approver = Callable[[Decision, NegotiationState], ApprovalResult]


def always_approve(decision: Decision, state: NegotiationState) -> ApprovalResult:
    return ApprovalResult("approved", "looks good")


def reject_action(action: Action, note: str = "human took over") -> Approver:
    """An approver that rejects one specific action once, then approves."""
    fired = {"done": False}

    def _approver(decision, state):
        if decision.action == action and not fired["done"]:
            fired["done"] = True
            return ApprovalResult("rejected", note)
        return ApprovalResult("approved", "looks good")
    return _approver


@dataclass
class TurnLog:
    role: str          # creator | agent | human
    label: str
    total: Optional[float]
    text: str


@dataclass
class NegotiationOutcome:
    status: str
    creator_first_ask: Optional[float]
    final_total: Optional[float]
    final_video_count: Optional[int]
    rounds: int
    questions_answered: int
    approvals_requested: int
    human_rejected: bool
    runs_autonomously: bool
    escalated: bool
    final_ev_net_mean: Optional[float]
    final_ev_roi: Optional[float]
    transcript: list[TurnLog] = field(default_factory=list)
    event_log: Optional[EventLog] = None

    @property
    def discount_from_ask(self) -> Optional[float]:
        # The creator's first ask is a per-video number; the final total scales with
        # the video count, so compare per-video to per-video, not ask to total.
        if self.creator_first_ask and self.final_total and self.final_video_count:
            final_per_video = self.final_total / self.final_video_count
            return 1.0 - (final_per_video / self.creator_first_ask)
        return None


def _deal_total(d: Optional[Deal]) -> Optional[float]:
    return None if d is None else d.total_usd


def run_negotiation(vm: ViewModel,
                    econ: CreatorEconomics,
                    ctx: CampaignContext,
                    creator: SimCreator,
                    *,
                    brief: Optional[CampaignBrief] = None,
                    catalog: Optional[FormatCatalog] = None,
                    rate_card: Optional[RateCard] = None,
                    autonomy: AutonomyLevel = AutonomyLevel.HUMAN_APPROVAL,
                    approver: Approver = always_approve,
                    creator_id="c1", campaign_id="camp1", thread_id="t1",
                    creator_name="there",
                    max_turns=14,
                    verbose=False) -> NegotiationOutcome:
    brief = brief or CampaignBrief()
    state = NegotiationState(creator_id=creator_id, campaign_id=campaign_id,
                             thread_id=thread_id, max_rounds=3,
                             autonomy_level=autonomy)
    transcript: list[TurnLog] = []
    approvals_requested = 0
    human_rejected = False
    log = EventLog()

    msg = creator.first_message()
    transcript.append(TurnLog("creator", msg.intent.value, msg.ask_total_usd, msg.raw_text))
    log.append(MESSAGE_RECEIVED, intent=msg.intent.value, ask_total=msg.ask_total_usd)
    if verbose:
        print(f"CREATOR: {msg.raw_text}")

    for _ in range(max_turns):
        decision = decide(msg, state, vm, econ, ctx, brief=brief,
                          catalog=catalog, rate_card=rate_card,
                          thread_last_sender="creator", thread_last_ts=time.time())
        state.last_agent_message_at = time.time()
        total = _deal_total(decision.deal)
        log.append(DECISION, action=decision.action.value, total=total,
                   video_count=(decision.deal.video_count if decision.deal else None),
                   requires_approval=decision.requires_approval,
                   qa_count=state.questions_answered)

        approved = True
        if decision.requires_approval:
            approvals_requested += 1
            result = approver(decision, state)
            state.record_approval(decision.action.value, total, result.outcome, result.note)
            approved = result.outcome == "approved"
            if verbose:
                print(f"   [human reviews -> {'approved' if approved else 'REJECTED'}: {result.note}]")
            if not approved:
                human_rejected = True
                state.status = Status.ESCALATED
                state.escalation_reason = "human rejected the proposed message"
                transcript.append(TurnLog("human", "rejected", total, result.note))
                log.append(HUMAN_REJECTED, action=decision.action.value, total=total)
                break
            log.append(APPROVAL, action=decision.action.value, outcome="approved")
        else:
            state.record_approval(decision.action.value, total, "auto", "autonomous send")
            log.append(APPROVAL, action=decision.action.value, outcome="auto")

        email = render_template(decision, creator_name=creator_name, seed=state.round_count)
        transcript.append(TurnLog("agent", decision.action.value, total, email))
        log.append(SENT, action=decision.action.value, total=total)
        if verbose:
            who = "AGENT (approved, sent)" if decision.requires_approval else "AGENT (sent)"
            print(f"\n{who} [{decision.action.value}]:\n{email}")
            print(f"   why: {decision.rationale}\n")

        if decision.action in (Action.ACCEPT, Action.SOFT_CLOSE,
                                Action.ESCALATE_HUMAN, Action.PAUSE_HUMAN):
            break

        msg = creator.respond_to(decision)
        transcript.append(TurnLog("creator", msg.intent.value, msg.ask_total_usd, msg.raw_text))
        log.append(MESSAGE_RECEIVED, intent=msg.intent.value, ask_total=msg.ask_total_usd)
        if verbose:
            print(f"CREATOR: {msg.raw_text}")

        if msg.intent == Intent.ACCEPTED:
            decision = decide(msg, state, vm, econ, ctx, brief=brief,
                              catalog=catalog, rate_card=rate_card)
            total = _deal_total(decision.deal)
            log.append(DECISION, action=decision.action.value, total=total,
                       video_count=(decision.deal.video_count if decision.deal else None),
                       requires_approval=decision.requires_approval,
                       qa_count=state.questions_answered)
            if decision.requires_approval:
                approvals_requested += 1
                result = approver(decision, state)
                state.record_approval(decision.action.value, total, result.outcome, result.note)
                if verbose:
                    print(f"   [human reviews -> {result.outcome}: {result.note}]")
                if result.outcome != "approved":
                    human_rejected = True
                    state.status = Status.ESCALATED
                    log.append(HUMAN_REJECTED, action=decision.action.value, total=total)
                    break
                log.append(APPROVAL, action=decision.action.value, outcome="approved")
            transcript.append(TurnLog("agent", decision.action.value, total,
                                      render_template(decision, creator_name)))
            log.append(SENT, action=decision.action.value, total=total)
            if verbose:
                print(f"\nAGENT [{decision.action.value}]: locking terms.")
            break

    final_deal = state.final_deal
    if final_deal is None and state.concession_history:
        final_deal = state.concession_history[-1].deal
    final_total = state.final_total
    if final_total is None and state.concession_history:
        final_total = state.concession_history[-1].offered_total

    final_net = final_roi = final_vc = None
    if final_deal:
        d = deal_from_dict(final_deal)
        final_vc = d.video_count
        if state.status == Status.ACCEPTED:
            if isinstance(d, Bundle) and catalog is not None:
                ev = ev_bundle(catalog, d, n_samples=8000, seed=99)
            else:
                ev = ev_distribution(vm, d, econ, n_samples=8000, seed=99)
            final_net, final_roi = ev.net_mean, ev.roi_mean

    runs_autonomously = (state.status == Status.ACCEPTED and
                         final_total is not None and
                         final_total <= ctx.auto_send_dollar_ceiling and
                         not human_rejected)

    return NegotiationOutcome(
        status=state.status.value,
        creator_first_ask=state.creator_first_ask or (creator.opening_ask or None),
        final_total=final_total, final_video_count=final_vc,
        rounds=state.round_count, questions_answered=state.questions_answered,
        approvals_requested=approvals_requested, human_rejected=human_rejected,
        runs_autonomously=runs_autonomously,
        escalated=(state.status == Status.ESCALATED),
        final_ev_net_mean=final_net, final_ev_roi=final_roi, transcript=transcript,
        event_log=log)


@dataclass
class BatchMetrics:
    n: int
    win_rate: float
    autonomous_close_rate: float
    escalation_rate: float
    avg_rounds: float
    avg_discount_from_ask: float
    avg_final_total: float
    avg_ev_net_mean: float
    avg_ev_roi: float

    def report(self) -> str:
        return (f"n={self.n}\n"
                f"  win rate                 {self.win_rate:.0%}\n"
                f"  autonomous-close rate    {self.autonomous_close_rate:.0%}\n"
                f"  escalation rate          {self.escalation_rate:.0%}\n"
                f"  avg rounds to close      {self.avg_rounds:.2f}\n"
                f"  avg discount from ask    {self.avg_discount_from_ask:.0%}\n"
                f"  avg final total          ${self.avg_final_total:,.0f}\n"
                f"  avg EV net (mean)        ${self.avg_ev_net_mean:,.0f}\n"
                f"  avg EV ROI               {self.avg_ev_roi:.1f}x")


def batch_metrics(outcomes: list[NegotiationOutcome]) -> BatchMetrics:
    n = len(outcomes)
    wins = [o for o in outcomes if o.status == "accepted"]
    disc = [o.discount_from_ask for o in wins if o.discount_from_ask is not None]
    net = [o.final_ev_net_mean for o in wins if o.final_ev_net_mean is not None]
    roi = [o.final_ev_roi for o in wins if o.final_ev_roi is not None]
    tot = [o.final_total for o in wins if o.final_total is not None]
    return BatchMetrics(
        n=n,
        win_rate=len(wins) / n if n else 0.0,
        autonomous_close_rate=sum(o.runs_autonomously for o in outcomes) / n if n else 0.0,
        escalation_rate=sum(o.escalated for o in outcomes) / n if n else 0.0,
        avg_rounds=float(np.mean([o.rounds for o in outcomes])) if n else 0.0,
        avg_discount_from_ask=float(np.mean(disc)) if disc else 0.0,
        avg_final_total=float(np.mean(tot)) if tot else 0.0,
        avg_ev_net_mean=float(np.mean(net)) if net else 0.0,
        avg_ev_roi=float(np.mean(roi)) if roi else 0.0)
