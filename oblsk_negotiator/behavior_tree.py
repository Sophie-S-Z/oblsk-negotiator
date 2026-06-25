"""
Behavior tree for the conversation side.

One pass per inbound message. The first branch that matches fires. The tree
reads the calculator's output and never does pricing math itself. The diagram in
figures/ is drawn from this file, so keep them in step.

Order of branches:
  human stepped in            -> pause, hand back
  creator accepted            -> lock terms
  creator asked a question    -> answer, stay in Q&A
  no offer yet                -> open with one flat number
  their ask is fine for us    -> accept
  their ask is wildly high     -> escalate to a person
  out of rounds               -> escalate to a person
  flat was rejected           -> escalate to a bundle (one offer)
  bundle on the table         -> add a sweetener, else escalate
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

from .ev_engine import CreatorEconomics, Deal, ViewModel, ev_distribution, EVResult
from .bundles import flat_offer, bundle_from_flat, add_non_price_sweetener
from .state import NegotiationState, Status, Phase, AutonomyLevel
from .qa import CampaignBrief, answer_from_brief, classify_question_topic


class Intent(str, Enum):
    QUESTION = "question"
    INTERESTED = "interested"        # engaged, no number yet
    NEGOTIATING = "negotiating"      # has a concrete ask
    ACCEPTED = "accepted"
    REJECTING = "rejecting"
    UNCLEAR = "unclear"


@dataclass
class CreatorMessage:
    intent: Intent
    ask_total_usd: Optional[float] = None
    ask_video_count: Optional[int] = None
    raw_text: str = ""


@dataclass
class CampaignContext:
    """Per-campaign knobs."""
    time_pressure: bool = False
    alternatives_available: bool = True
    auto_send_dollar_ceiling: float = 3000.0  # above this, a person signs off
    roi_hurdle: float = 3.0                    # min revenue/cost multiple
    pay_fraction: float = 0.30                 # opening flat as share of EV/video
    accept_margin: float = 1.05                # accept asks within this of our max
    extreme_ask_multiple: float = 2.0          # above this multiple, escalate


class Action(str, Enum):
    ANSWER_QUESTION = "answer_question"
    OPENING_FLAT = "opening_flat"
    ESCALATE_BUNDLE = "escalate_bundle"
    ADD_SWEETENER = "add_sweetener"
    ACCEPT = "accept"
    SOFT_CLOSE = "soft_close"
    ESCALATE_HUMAN = "escalate_human"     # hand the thread to a person
    PAUSE_HUMAN = "pause_human"           # a person stepped in mid-thread


# Actions that send a message to the creator (so they pass through approval).
_OUTBOUND = {
    Action.ANSWER_QUESTION, Action.OPENING_FLAT, Action.ESCALATE_BUNDLE,
    Action.ADD_SWEETENER, Action.ACCEPT, Action.SOFT_CLOSE,
}


@dataclass
class Decision:
    action: Action
    deal: Optional[Deal]
    ev: Optional[EVResult]
    rationale: str                       # internal, for the reviewer
    requires_approval: bool = False
    answer_text: Optional[str] = None    # for ANSWER_QUESTION
    non_price_lever: Optional[str] = None
    reserve_deal: Optional[Deal] = None  # what we would reveal next if rejected
    reserve_ev: Optional[EVResult] = None


def _requires_approval(action: Action, total: Optional[float],
                       state: NegotiationState, ctx: CampaignContext) -> bool:
    if action == Action.ESCALATE_HUMAN:
        return True
    if action not in _OUTBOUND:
        return False
    if state.autonomy_level == AutonomyLevel.HUMAN_APPROVAL:
        return True
    return total is not None and total > ctx.auto_send_dollar_ceiling


# ---- valuation: read the calculator, decide what we will pay ----------------

def _round_money(x: float, step: float = 50.0) -> float:
    """Round an offer to a clean number, the way a person would write it."""
    return float(round(x / step) * step)


def _fair_flat_total(vm: ViewModel, econ: CreatorEconomics,
                     ctx: CampaignContext) -> float:
    """What we will pay for one video: a slice of the expected revenue it drives,
    floored so it never goes silly-low on a small creator."""
    exp_rev_per_video = vm.median_views * econ.revenue_per_view
    return max(250.0, ctx.pay_fraction * exp_rev_per_video)


def _flat_total(vm, econ, ctx) -> float:
    return _round_money(_fair_flat_total(vm, econ, ctx), 50.0)


def _max_pay_per_video(vm, econ, ctx) -> float:
    """Oblsk's ceiling: the most per video that still clears the ROI hurdle. Uses
    the same conservative (median-based) revenue proxy as the opening flat, so the
    agent never pays a rate that the campaign cannot justify on the numbers."""
    exp_rev_per_video = vm.median_views * econ.revenue_per_view
    if ctx.roi_hurdle <= 0:
        return exp_rev_per_video
    return exp_rev_per_video / ctx.roi_hurdle


# A small bulk discount the creator grants for committing to several videos at
# once. Off our own fair rate when we anchor there; a lighter touch when we anchor
# to the creator's revealed ask, so the rate still clears their (private) floor.
_BUNDLE_BULK_DISCOUNT = 0.10
_ASK_BULK_DISCOUNT = 0.05


def _bundle_per_video(vm, econ, ctx, ask_per_video: Optional[float] = None) -> float:
    """The per-video rate for a bundle. Pay our fair rate (less a bulk discount);
    if the creator has revealed a higher per-video ask, meet them just under it so
    the rate is one they will actually accept. Never exceed the ROI ceiling."""
    fair = _fair_flat_total(vm, econ, ctx)
    floor = fair * (1.0 - _BUNDLE_BULK_DISCOUNT)
    target = floor
    if ask_per_video:
        target = max(floor, ask_per_video * (1.0 - _ASK_BULK_DISCOUNT))
    return min(target, _max_pay_per_video(vm, econ, ctx))


def _bundle_total(vm, econ, ctx, ask_per_video: Optional[float] = None,
                  target_videos: int = 3) -> float:
    per_video = _bundle_per_video(vm, econ, ctx, ask_per_video)
    return _round_money(per_video * target_videos, 100.0)


def _willingness_to_pay(state, vm, econ, ctx) -> float:
    """The most we would happily pay right now. Once a bundle is on the table we
    judge an ask against the bundle total, anchored to the creator's first ask so
    the comparison is like-for-like."""
    if state.bundle_offered:
        return _bundle_total(vm, econ, ctx, ask_per_video=state.creator_first_ask)
    return _flat_total(vm, econ, ctx)


def _score(deal: Deal, vm: ViewModel, econ: CreatorEconomics,
           seed: int = 11) -> EVResult:
    return ev_distribution(vm, deal, econ, n_samples=8000, seed=seed)


def _is_high_value(vm: ViewModel, econ: CreatorEconomics) -> bool:
    return vm.median_views * econ.revenue_per_view > 5000.0


# ---- the tree ---------------------------------------------------------------

def decide(msg: CreatorMessage,
           state: NegotiationState,
           vm: ViewModel,
           econ: CreatorEconomics,
           ctx: CampaignContext,
           *,
           brief: Optional[CampaignBrief] = None,
           thread_last_sender: str = "creator",
           thread_last_ts: float = 0.0) -> Decision:
    """One pass through the tree. Mutates state, returns a Decision."""
    brief = brief or CampaignBrief()

    if state.human_intervened(thread_last_sender, thread_last_ts):
        state.notes.append("human message detected mid-thread; pausing")
        return Decision(
            action=Action.PAUSE_HUMAN, deal=None, ev=None,
            rationale="A teammate sent a message in this thread. Pause and resync "
                      "before acting again.",
            requires_approval=False)

    if msg.intent == Intent.ACCEPTED:
        state.status = Status.ACCEPTED
        state.phase = Phase.CLOSING
        last = state.concession_history[-1] if state.concession_history else None
        if last:
            state.final_total = last.offered_total
            state.final_deal = last.deal
        return Decision(
            action=Action.ACCEPT, deal=None, ev=None,
            rationale="Creator accepted. Lock terms and hand to execution.",
            requires_approval=_requires_approval(Action.ACCEPT, state.final_total,
                                                 state, ctx))

    if msg.intent == Intent.QUESTION:
        topic = classify_question_topic(msg.raw_text)
        if topic == "budget" and not state.flat_offer_made:
            return _opening_flat(state, vm, econ, ctx, reason="asked about budget")
        state.questions_answered += 1
        state.phase = Phase.QA
        return Decision(
            action=Action.ANSWER_QUESTION, deal=None, ev=None,
            answer_text=answer_from_brief(msg.raw_text, brief),
            rationale=f"Question about {topic}. Answer from the brief, stay in Q&A.",
            requires_approval=_requires_approval(Action.ANSWER_QUESTION, None,
                                                 state, ctx))

    if not state.flat_offer_made:
        if msg.intent == Intent.REJECTING:
            return _soft_close(state, vm, econ, ctx)
        reason = "engaged, no number yet" if msg.intent == Intent.INTERESTED \
            else "opening the deal"
        return _opening_flat(state, vm, econ, ctx, reason=reason)

    # An offer is on the table.
    ask = msg.ask_total_usd
    if ask is not None and state.creator_first_ask is None:
        state.creator_first_ask = ask

    willingness = _willingness_to_pay(state, vm, econ, ctx)

    if ask is not None and ask <= willingness * ctx.accept_margin:
        return _accept_ask(msg, state, vm, econ, ctx, willingness)

    # A wildly high ask is not worth negotiating through. Hand it to a person.
    if ask is not None and ask > willingness * ctx.extreme_ask_multiple:
        return _escalate_human(state, vm, econ, ctx, ask, extreme=True)

    if state.round_count >= state.max_rounds:
        return _escalate_human(state, vm, econ, ctx, ask)

    if not state.bundle_offered:
        return _escalate_bundle(state, vm, econ, ctx, ask)
    return _add_sweetener_or_escalate(state, vm, econ, ctx, ask)


# ---- nodes ------------------------------------------------------------------

def _opening_flat(state, vm, econ, ctx, *, reason: str) -> Decision:
    deal = flat_offer(_flat_total(vm, econ, ctx), video_count=1)
    ev = _score(deal, vm, econ, seed=3)

    state.flat_offer_made = True
    state.phase = Phase.OPENING
    state.round_count += 1
    state.record_concession(asdict(deal), deal.total_usd, None)

    reserve = bundle_from_flat(deal)
    reserve_ev = _score(reserve, vm, econ, seed=4)
    return Decision(
        action=Action.OPENING_FLAT, deal=deal, ev=ev,
        rationale=f"Open with one flat number, ${deal.total_usd:,.0f} for a single "
                  f"video ({reason}). ROI {ev.roi_mean:.1f}x. If rejected, go to a "
                  f"{reserve.video_count}-video bundle.",
        reserve_deal=reserve, reserve_ev=reserve_ev,
        requires_approval=_requires_approval(Action.OPENING_FLAT, deal.total_usd,
                                             state, ctx))


def _escalate_bundle(state, vm, econ, ctx, ask) -> Decision:
    # The flat counter is a per-single-video number, so it anchors the bundle's
    # per-video rate. Hold that rate near their ask and add videos: each one is an
    # independent draw, so the total return scales while the rate stays fair.
    target_videos = 3
    per_video = _bundle_per_video(vm, econ, ctx, ask_per_video=ask)
    total = _round_money(per_video * target_videos, 100.0)
    bundle = Deal(video_count=target_videos, flat_per_video=total / target_videos)
    ev = _score(bundle, vm, econ, seed=7)

    state.bundle_offered = True
    state.phase = Phase.NEGOTIATING
    state.round_count += 1
    state.record_concession(asdict(bundle), bundle.total_usd, ask)

    flat_ev = _score(flat_offer(_flat_total(vm, econ, ctx)), vm, econ, seed=3)
    gain = ev.net_mean - flat_ev.net_mean
    ask_str = f"${ask:,.0f}" if ask is not None else "the flat rate"
    return Decision(
        action=Action.ESCALATE_BUNDLE, deal=bundle, ev=ev,
        rationale=f"Flat rejected ({ask_str}). Hold a fair "
                  f"${bundle.flat_per_video:,.0f}/video and offer {target_videos} "
                  f"videos (${bundle.total_usd:,.0f} total). Net EV up about "
                  f"${gain:,.0f} from {target_videos - 1} more independent draws, "
                  f"ROI {ev.roi_mean:.1f}x. One offer, no menu.",
        requires_approval=_requires_approval(Action.ESCALATE_BUNDLE,
                                             bundle.total_usd, state, ctx))


def _add_sweetener_or_escalate(state, vm, econ, ctx, ask) -> Decision:
    already_sweetened = any(c.deal.get("net_terms_days", 30) < 30
                            for c in state.concession_history)
    if already_sweetened:
        return _escalate_human(state, vm, econ, ctx, ask)

    base = Deal(**state.concession_history[-1].deal)
    deal = add_non_price_sweetener(base, "fast_pay")
    ev = _score(deal, vm, econ, seed=9)

    state.round_count += 1
    state.record_concession(asdict(deal), deal.total_usd, ask)
    return Decision(
        action=Action.ADD_SWEETENER, deal=deal, ev=ev, non_price_lever="fast_pay",
        rationale=f"Bundle is on the table and they still want more. Hold "
                  f"${deal.total_usd:,.0f} and add faster payment to close the gap.",
        requires_approval=_requires_approval(Action.ADD_SWEETENER, deal.total_usd,
                                             state, ctx))


def _accept_ask(msg, state, vm, econ, ctx, willingness) -> Decision:
    ask = msg.ask_total_usd
    videos = msg.ask_video_count or (
        state.concession_history[-1].offered_video_count
        if state.concession_history else 1)
    deal = flat_offer(ask, video_count=max(videos, 1))
    ev = _score(deal, vm, econ, seed=5)

    state.status = Status.ACCEPTED
    state.phase = Phase.CLOSING
    state.round_count += 1
    state.record_concession(asdict(deal), deal.total_usd, ask)
    state.final_total = deal.total_usd
    state.final_deal = asdict(deal)
    return Decision(
        action=Action.ACCEPT, deal=deal, ev=ev,
        rationale=f"Accept ${ask:,.0f} across {deal.video_count} video(s). At or "
                  f"below our max (${willingness:,.0f}), ROI {ev.roi_mean:.1f}x.",
        requires_approval=_requires_approval(Action.ACCEPT, deal.total_usd,
                                             state, ctx))


def _soft_close(state, vm, econ, ctx) -> Decision:
    state.status = Status.REJECTED
    high_value = _is_high_value(vm, econ)
    note = " High-value creator, flag for follow-up." if high_value else ""
    return Decision(
        action=Action.SOFT_CLOSE, deal=None, ev=None,
        rationale="Not interested. Close gently, keep the door open." + note,
        requires_approval=_requires_approval(Action.SOFT_CLOSE, None, state, ctx) or high_value)


def _escalate_human(state, vm, econ, ctx, ask, *, extreme=False) -> Decision:
    state.status = Status.ESCALATED
    ask_str = f"${ask:,.0f}" if ask is not None else "their position"
    if extreme:
        state.escalation_reason = (
            f"Ask {ask_str} is more than {ctx.extreme_ask_multiple:.0f}x what the "
            f"numbers support. A person should decide whether to walk or stretch.")
    else:
        state.escalation_reason = (
            f"Round {state.round_count} of {state.max_rounds} with no agreement on "
            f"{ask_str}. Handing to a person.")
    last = state.concession_history[-1] if state.concession_history else None
    deal = Deal(**last.deal) if last else None
    ev = _score(deal, vm, econ, seed=15) if deal else None
    return Decision(action=Action.ESCALATE_HUMAN, deal=deal, ev=ev,
                    rationale=state.escalation_reason, requires_approval=True)
