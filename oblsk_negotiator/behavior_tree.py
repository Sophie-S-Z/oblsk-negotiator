"""
Behavior tree for the conversation side.

One pass per inbound message. The first branch that matches fires. The tree
reads the calculator's output and never does pricing math itself. The branch
order is data: it lives in tree_spec.yaml as (guard, handler) name pairs, walked
by the interpreter at the bottom of this file. Adding a branch is a config edit
plus the guard and handler it names; the diagram in figures/ is drawn from the
same order, so keep them in step.

Order of branches (see tree_spec.yaml for the source of truth):
  human stepped in            -> pause, hand back
  creator accepted            -> lock terms
  creator asked a question    -> answer, stay in Q&A
  no offer yet                -> open with one flat number
  their ask is fine for us    -> accept
  their ask is wildly high     -> escalate to a person
  out of rounds               -> escalate to a person
  flat was rejected           -> revise the flat once, up to our ROI ceiling
  revised flat also rejected  -> escalate to a bundle (one offer)
  bundle on the table         -> add a sweetener, else escalate
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .ev_engine import (CreatorEconomics, Deal, ViewModel, ev_distribution, EVResult,
                        Bundle, FormatCatalog, ev_bundle, deal_to_dict, deal_from_dict)
from .bundles import (flat_offer, bundle_from_flat, add_non_price_sweetener,
                      compose_bundle, bundle_format_summary)
from .rate_card import RateCard
from .state import NegotiationState, Status, Phase, AutonomyLevel
from .qa import CampaignBrief, answer_from_brief, classify_question_topic, forward_nudge


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
    qa_offer_after: int = 3                     # answered questions before we open an offer


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


def _revised_flat_total(vm, econ, ctx, ask_per_video: Optional[float] = None) -> float:
    """Our revised single-video number after the opening flat is rejected: move up
    to the ROI ceiling, the most the numbers allow, but never above it and never
    above what the creator asked. This is the last single-video offer before we
    restructure the deal into a bundle."""
    ceiling = _max_pay_per_video(vm, econ, ctx)
    target = ceiling if not ask_per_video else min(ceiling, ask_per_video)
    return _round_money(target, 50.0)


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
    judge an ask against the bundle total: for a composed multi-format bundle that
    total already clears ROI, so it is our ceiling; for a single-format bundle we
    anchor to the creator's first ask so the comparison is like-for-like."""
    if state.bundle_offered and state.concession_history:
        last = deal_from_dict(state.concession_history[-1].deal)
        if isinstance(last, Bundle):
            return last.total_usd
    if state.bundle_offered:
        return _bundle_total(vm, econ, ctx, ask_per_video=state.creator_first_ask)
    return _flat_total(vm, econ, ctx)


def _score(deal: Deal, vm: ViewModel, econ: CreatorEconomics,
           seed: int = 11) -> EVResult:
    return ev_distribution(vm, deal, econ, n_samples=8000, seed=seed)


def _score_any(deal, vm: ViewModel, econ: CreatorEconomics,
               catalog: Optional[FormatCatalog] = None,
               seed: int = 11) -> EVResult:
    """Score a Deal or a Bundle. A bundle draws each line from its own format's
    reach, so it needs the catalog; a flat deal uses the single view model."""
    if isinstance(deal, Bundle):
        if catalog is None:
            raise ValueError("scoring a bundle needs a format catalog")
        return ev_bundle(catalog, deal, n_samples=8000, seed=seed)
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
           catalog: Optional[FormatCatalog] = None,
           rate_card: Optional[RateCard] = None,
           thread_last_sender: str = "creator",
           thread_last_ts: float = 0.0) -> Decision:
    """One pass through the tree. Mutates state, returns a Decision.

    The branch order lives in tree_spec.yaml as (guard, handler) name pairs; this
    function builds the context and walks that spec. Guards and handlers resolve
    against the registries at the bottom of this file, and all pricing math stays
    in their Python functions. When a `catalog` (the value side) and `rate_card`
    (the creator's asking side) are both supplied, the bundle escalation composes
    a multi-format package; with neither it falls back to the single-format
    flat-plus-videos bundle.
    """
    brief = brief or CampaignBrief()
    dctx = DecisionContext(
        msg=msg, state=state, vm=vm, econ=econ, ctx=ctx, brief=brief,
        catalog=catalog, rate_card=rate_card,
        thread_last_sender=thread_last_sender, thread_last_ts=thread_last_ts)
    return run_tree(dctx, _load_tree_spec())


# ---- nodes ------------------------------------------------------------------

def _opening_flat(state, vm, econ, ctx, *, reason: str) -> Decision:
    deal = flat_offer(_flat_total(vm, econ, ctx), video_count=1)
    ev = _score(deal, vm, econ, seed=3)

    state.flat_offer_made = True
    state.phase = Phase.OPENING
    state.round_count += 1
    state.record_concession(deal_to_dict(deal), deal.total_usd, None,
                            video_count=deal.video_count)

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


def _escalate_bundle(state, vm, econ, ctx, ask, *,
                     brief: Optional[CampaignBrief] = None,
                     catalog: Optional[FormatCatalog] = None,
                     rate_card: Optional[RateCard] = None) -> Decision:
    """Flat was rejected. Escalate to one bundle offer.

    With a rate card and a format catalog in hand, compose a multi-format package:
    keep the formats whose reach justifies their price, discount them the way the
    market already discounts combos, and offer that. Without them, or when no
    ROI-positive package exists at a discount a creator would accept, fall back to
    the single-format path: hold a fair per-video rate and add videos, each one an
    independent draw, so the total scales while the rate stays fair.
    """
    brief = brief or CampaignBrief()
    if catalog is not None and rate_card is not None:
        composed = compose_bundle(
            rate_card, catalog, roi_hurdle=ctx.roi_hurdle,
            primary_fmt=brief.primary_format,
            allowed_formats=brief.allowed_formats)
        if composed is not None:
            return _escalate_multiformat(state, vm, econ, ctx, ask, catalog,
                                         composed)

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
    state.record_concession(deal_to_dict(bundle), bundle.total_usd, ask,
                            video_count=bundle.video_count)

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


def _escalate_multiformat(state, vm, econ, ctx, ask, catalog, bundle) -> Decision:
    """Escalate to a composed multi-format bundle. The package already clears the
    ROI hurdle by construction (compose_bundle drops any line that would sink it).
    Score it with the real Monte Carlo over each format's reach, record it, and
    write the single offer."""
    ev = ev_bundle(catalog, bundle, n_samples=8000, seed=7)

    state.bundle_offered = True
    state.phase = Phase.NEGOTIATING
    state.round_count += 1
    state.record_concession(deal_to_dict(bundle), bundle.total_usd, ask,
                            video_count=bundle.video_count)

    formats = bundle_format_summary(bundle)
    ask_str = f"${ask:,.0f}" if ask is not None else "the flat rate"
    return Decision(
        action=Action.ESCALATE_BUNDLE, deal=bundle, ev=ev,
        rationale=f"Flat rejected ({ask_str}). Offer a {len(bundle.lines)}-format "
                  f"package ({formats}) at ${bundle.total_usd:,.0f}: each line is "
                  f"the creator's own list price less the standard combo discount, "
                  f"so it reads as a normal package to them. Clears ROI "
                  f"{ev.roi_mean:.1f}x on reach, P(net>0) "
                  f"{ev.prob_net_positive:.0%}. One offer, no menu.",
        requires_approval=_requires_approval(Action.ESCALATE_BUNDLE,
                                             bundle.total_usd, state, ctx))


def _add_sweetener_or_escalate(state, vm, econ, ctx, ask, *,
                               catalog: Optional[FormatCatalog] = None) -> Decision:
    already_sweetened = any(c.deal.get("net_terms_days", 30) < 30
                            for c in state.concession_history)
    if already_sweetened:
        return _escalate_human(state, vm, econ, ctx, ask, catalog=catalog)

    base = deal_from_dict(state.concession_history[-1].deal)
    deal = add_non_price_sweetener(base, "fast_pay")
    ev = _score_any(deal, vm, econ, catalog=catalog, seed=9)

    state.round_count += 1
    state.record_concession(deal_to_dict(deal), deal.total_usd, ask,
                            video_count=deal.video_count)
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
    state.record_concession(deal_to_dict(deal), deal.total_usd, ask,
                            video_count=deal.video_count)
    state.final_total = deal.total_usd
    state.final_deal = deal_to_dict(deal)
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


def _escalate_human(state, vm, econ, ctx, ask, *, extreme=False,
                    catalog: Optional[FormatCatalog] = None) -> Decision:
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
    deal = deal_from_dict(last.deal) if last else None
    ev = _score_any(deal, vm, econ, catalog=catalog, seed=15) if deal else None
    return Decision(action=Action.ESCALATE_HUMAN, deal=deal, ev=ev,
                    rationale=state.escalation_reason, requires_approval=True)


# ---- config-driven dispatch -------------------------------------------------
# The branches above are walked from a spec (tree_spec.yaml) rather than a fixed
# if/elif chain. The spec is an ordered list of (guard, handler) name pairs; the
# interpreter fires the handler of the first guard that holds. Guards are pure
# predicates over a DecisionContext and handlers return a Decision; both resolve
# against the registries below. The point is that branch *order and wiring* are
# data while the pricing *logic* stays in the Python functions above, so adding a
# node is a config edit, not a rewrite, without turning the config into a DSL.

@dataclass
class DecisionContext:
    """Everything a guard or handler needs for one pass, in one object. `ask` and
    `willingness` are computed lazily so the early branches never pay for them and
    the ask-capture side effect fires exactly where the old chain put it: once an
    offer is on the table (the pre-offer branches return before reaching here) and
    only for the first revealed ask."""
    msg: CreatorMessage
    state: NegotiationState
    vm: ViewModel
    econ: CreatorEconomics
    ctx: CampaignContext
    brief: CampaignBrief
    catalog: Optional[FormatCatalog] = None
    rate_card: Optional[RateCard] = None
    thread_last_sender: str = "creator"
    thread_last_ts: float = 0.0
    _willingness: Optional[float] = field(default=None, repr=False)

    @property
    def ask(self) -> Optional[float]:
        a = self.msg.ask_total_usd
        if (a is not None and self.state.flat_offer_made
                and self.state.creator_first_ask is None):
            self.state.creator_first_ask = a
        return a

    @property
    def willingness(self) -> float:
        if self._willingness is None:
            self._willingness = _willingness_to_pay(
                self.state, self.vm, self.econ, self.ctx)
        return self._willingness


# ---- handlers extracted from the old chain (one move each) ------------------

def _pause_human(dctx: DecisionContext) -> Decision:
    dctx.state.notes.append("human message detected mid-thread; pausing")
    return Decision(
        action=Action.PAUSE_HUMAN, deal=None, ev=None,
        rationale="A teammate sent a message in this thread. Pause and resync "
                  "before acting again.",
        requires_approval=False)


def _accept_on_confirm(dctx: DecisionContext) -> Decision:
    state = dctx.state
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
                                             state, dctx.ctx))


def _question(dctx: DecisionContext) -> Decision:
    msg, state, ctx, brief = dctx.msg, dctx.state, dctx.ctx, dctx.brief
    topic = classify_question_topic(msg.raw_text)
    if topic == "budget" and not state.flat_offer_made:
        return _opening_flat(state, dctx.vm, dctx.econ, ctx,
                             reason="asked about budget")
    state.questions_answered += 1
    state.phase = Phase.QA
    # Conversation-aware: do not answer in place forever. Once several questions
    # are answered and no number is on the table, the next step is a concrete
    # offer, so open it instead of fielding another question.
    if not state.flat_offer_made and state.questions_answered >= ctx.qa_offer_after:
        return _opening_flat(state, dctx.vm, dctx.econ, ctx,
                             reason="several questions answered, time for a number")
    answer = answer_from_brief(msg.raw_text, brief)
    nudge = forward_nudge(state.questions_answered, state.flat_offer_made)
    answer_text = f"{answer} {nudge}".strip() if nudge else answer
    return Decision(
        action=Action.ANSWER_QUESTION, deal=None, ev=None,
        answer_text=answer_text,
        rationale=f"Question about {topic}. Answer from the brief and nudge "
                  f"toward the next step (answered {state.questions_answered}).",
        requires_approval=_requires_approval(Action.ANSWER_QUESTION, None,
                                             state, ctx))


def _open_or_soft_close(dctx: DecisionContext) -> Decision:
    msg, state = dctx.msg, dctx.state
    if msg.intent == Intent.REJECTING:
        return _soft_close(state, dctx.vm, dctx.econ, dctx.ctx)
    reason = "engaged, no number yet" if msg.intent == Intent.INTERESTED \
        else "opening the deal"
    return _opening_flat(state, dctx.vm, dctx.econ, dctx.ctx, reason=reason)


def _revise_flat(dctx: DecisionContext) -> Decision:
    """The opening flat was rejected. Before restructuring into a bundle, make one
    revised single-video offer at the top of what the numbers allow: meet the
    creator up to our ROI ceiling. Only if this is rejected too do we move to a
    bundle, so we have tried hardest on the simplest deal first."""
    state, vm, econ, ctx = dctx.state, dctx.vm, dctx.econ, dctx.ctx
    deal = flat_offer(_revised_flat_total(vm, econ, ctx, dctx.ask), video_count=1)
    ev = _score(deal, vm, econ, seed=6)

    state.flat_revised = True
    state.phase = Phase.NEGOTIATING
    state.round_count += 1
    state.record_concession(deal_to_dict(deal), deal.total_usd, dctx.ask,
                            video_count=deal.video_count)

    reserve = bundle_from_flat(deal)
    reserve_ev = _score(reserve, vm, econ, seed=8)
    ask_str = f"${dctx.ask:,.0f}" if dctx.ask is not None else "the flat rate"
    return Decision(
        action=Action.OPENING_FLAT, deal=deal, ev=ev,
        rationale=f"Flat rejected ({ask_str}). Hold the single video but revise up "
                  f"to ${deal.total_usd:,.0f}, the most the ROI hurdle allows "
                  f"({ev.roi_mean:.1f}x). If rejected again, restructure into a "
                  f"bundle.",
        reserve_deal=reserve, reserve_ev=reserve_ev,
        requires_approval=_requires_approval(Action.OPENING_FLAT, deal.total_usd,
                                             state, ctx))


# ---- registries: names the spec resolves against ---------------------------

GUARDS = {
    "human_intervened": lambda d: d.state.human_intervened(
        d.thread_last_sender, d.thread_last_ts),
    "creator_accepted": lambda d: d.msg.intent == Intent.ACCEPTED,
    "creator_asked": lambda d: d.msg.intent == Intent.QUESTION,
    "no_offer_yet": lambda d: not d.state.flat_offer_made,
    "ask_acceptable": lambda d: (
        d.ask is not None and d.ask <= d.willingness * d.ctx.accept_margin),
    "ask_extreme": lambda d: (
        d.ask is not None and d.ask > d.willingness * d.ctx.extreme_ask_multiple),
    "out_of_rounds": lambda d: d.state.round_count >= d.state.max_rounds,
    "no_revision_yet": lambda d: (
        not d.state.flat_revised and not d.state.bundle_offered),
    "no_bundle_yet": lambda d: not d.state.bundle_offered,
    "always": lambda d: True,
}

HANDLERS = {
    "pause_human": _pause_human,
    "accept_on_confirm": _accept_on_confirm,
    "answer_question": _question,
    "open_or_soft_close": _open_or_soft_close,
    "revise_flat": _revise_flat,
    "accept_ask": lambda d: _accept_ask(
        d.msg, d.state, d.vm, d.econ, d.ctx, d.willingness),
    "escalate_human_extreme": lambda d: _escalate_human(
        d.state, d.vm, d.econ, d.ctx, d.ask, extreme=True, catalog=d.catalog),
    "escalate_human": lambda d: _escalate_human(
        d.state, d.vm, d.econ, d.ctx, d.ask, catalog=d.catalog),
    "escalate_bundle": lambda d: _escalate_bundle(
        d.state, d.vm, d.econ, d.ctx, d.ask,
        brief=d.brief, catalog=d.catalog, rate_card=d.rate_card),
    "sweeten_or_escalate": lambda d: _add_sweetener_or_escalate(
        d.state, d.vm, d.econ, d.ctx, d.ask, catalog=d.catalog),
}


# ---- interpreter ------------------------------------------------------------

def run_tree(dctx: DecisionContext, spec: list) -> Decision:
    """Fire the handler of the first guard that holds. The spec must end in a
    total guard (`always`); if none match, that is a malformed spec."""
    for step in spec:
        if GUARDS[step["guard"]](dctx):
            return HANDLERS[step["handler"]](dctx)
    raise RuntimeError(
        "no tree branch matched; tree_spec.yaml must end in an 'always' guard")


def _validate_spec(spec) -> None:
    if not spec:
        raise ValueError("tree_spec.yaml is empty")
    for step in spec:
        if step.get("guard") not in GUARDS:
            raise ValueError(f"unknown guard {step.get('guard')!r} in tree_spec.yaml")
        if step.get("handler") not in HANDLERS:
            raise ValueError(
                f"unknown handler {step.get('handler')!r} in tree_spec.yaml")


_TREE_SPEC_CACHE = None


def _load_tree_spec() -> list:
    """Load and cache the branch order from tree_spec.yaml, validating that every
    name resolves so a bad edit fails loudly at load rather than mid-negotiation."""
    global _TREE_SPEC_CACHE
    if _TREE_SPEC_CACHE is None:
        import yaml
        path = os.path.join(os.path.dirname(__file__), "tree_spec.yaml")
        with open(path) as f:
            spec = yaml.safe_load(f)
        _validate_spec(spec)
        _TREE_SPEC_CACHE = spec
    return _TREE_SPEC_CACHE
