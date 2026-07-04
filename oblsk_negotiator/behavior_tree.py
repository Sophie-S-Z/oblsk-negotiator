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
  creator accepted our offer  -> lock terms
  creator asked a question    -> answer from the brief, or ask the team
  negotiating contract terms  -> escalate to a person (never negotiate paper)
  references a call we missed -> ask the team for that context
  wants a call, no counter    -> propose call windows, a human runs the call
  no offer yet                -> open with one flat number (the anchor)
  follow-up with no counter   -> hold the standing offer, concede nothing
  their ask is fine for us    -> accept
  their ask is wildly high    -> escalate to a person
  out of rounds               -> escalate to a person
  flat was rejected           -> revise once, up to the ladder's target
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
from .pricing import PricingPolicy, PriceLadder, price_ladder
from .rate_card import RateCard
from .state import NegotiationState, Status, Phase, AutonomyLevel
from .qa import (CampaignBrief, answer_question, can_answer,
                classify_question_topic, forward_nudge)


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
    # The message negotiates contract/legal terms (exclusivity compensation,
    # equity structure, kill fees, indemnities) rather than just price. The
    # agent never negotiates paper: these go to a person.
    contract_terms: bool = False
    # They propose, request, or confirm availability for a call. Calls are run
    # by real people; the agent only gets one on the calendar.
    wants_call: bool = False
    # They reference a call or conversation the agent was not part of. That
    # context lives with the team, so the agent asks before acting on it.
    refers_to_call: bool = False


@dataclass
class CampaignContext:
    """Per-campaign knobs. Every number the negotiator uses lives here, so a
    campaign tunes its stance in config instead of editing pricing code."""
    auto_send_dollar_ceiling: float = 3000.0  # above this, a person signs off

    # The pricing ladder (see pricing.py): where we open, aim, and stop.
    roi_target: float = 3.0                    # revenue/cost multiple we aim for
    roi_min: float = 1.0                       # multiple the p10 downside must clear
    anchor_factor: float = 0.62                # opening offer as fraction of target
    cpm_cap_usd: Optional[float] = None        # optional $ per 1,000 expected views cap
    risk_discount: float = 1.0                 # authenticity multiplier on the downside
    min_offer_usd: float = 250.0               # never open below this

    # Negotiation stance.
    accept_margin: float = 1.05                # accept asks within this of our max
    extreme_ask_multiple: float = 2.0          # asks above this multiple of the
                                               # walk-away ceiling go to a person
    qa_offer_after: int = 3                    # answered questions before we open an offer
    target_videos: int = 3                     # videos in the single-format bundle
    bundle_bulk_discount: float = 0.10         # per-video discount for committing to several
    ask_bulk_discount: float = 0.05            # lighter discount when anchored to their ask
    high_value_threshold_usd: float = 5000.0   # expected revenue/video that flags follow-up
    money_step: float = 50.0                   # round flat offers to this
    bundle_money_step: float = 100.0           # round bundle totals to this
    call_windows: str = ""                     # when our team can take intro calls,
                                               # e.g. "Thursday after 3 PM ET"; the
                                               # agent proposes these, a human runs
                                               # the call

    def pricing_policy(self) -> PricingPolicy:
        return PricingPolicy(
            roi_target=self.roi_target, roi_min=self.roi_min,
            anchor_factor=self.anchor_factor, cpm_cap_usd=self.cpm_cap_usd,
            risk_discount=self.risk_discount, min_offer_usd=self.min_offer_usd)


class Action(str, Enum):
    ANSWER_QUESTION = "answer_question"
    OPENING_FLAT = "opening_flat"
    ESCALATE_BUNDLE = "escalate_bundle"
    ADD_SWEETENER = "add_sweetener"
    HOLD_FIRM = "hold_firm"               # restate the standing offer, concede nothing
    PROPOSE_CALL = "propose_call"         # get a call on the calendar (a human runs it)
    ACCEPT = "accept"
    SOFT_CLOSE = "soft_close"
    ASK_HUMAN = "ask_human"               # pause; ask the team for missing context
    ESCALATE_HUMAN = "escalate_human"     # hand the thread to a person
    PAUSE_HUMAN = "pause_human"           # a person stepped in mid-thread


# Actions that send a message to the creator (so they pass through approval).
_OUTBOUND = {
    Action.ANSWER_QUESTION, Action.OPENING_FLAT, Action.ESCALATE_BUNDLE,
    Action.ADD_SWEETENER, Action.HOLD_FIRM, Action.PROPOSE_CALL,
    Action.ACCEPT, Action.SOFT_CLOSE,
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
    # What the agent needs from the team: the specific question to answer
    # (ASK_HUMAN) or the follow-through to do (PROPOSE_CALL: run the call).
    human_prompt: Optional[str] = None


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
# All dollar positions come from the pricing ladder (pricing.py): open at the
# anchor, revise toward the target, never cross the walk-away ceiling.

def _round_money(x: float, step: float = 50.0) -> float:
    """Round an offer to a clean number, the way a person would write it."""
    return float(round(x / step) * step)


def _ladder(vm: ViewModel, econ: CreatorEconomics,
            ctx: CampaignContext) -> PriceLadder:
    """The single-video ladder for this creator under this campaign's policy."""
    return price_ladder(vm, econ, ctx.pricing_policy(), video_count=1,
                        n_samples=8000, seed=11)


def _flat_total(vm, econ, ctx) -> float:
    """The opening flat: the ladder's anchor, rounded to a clean number."""
    return _round_money(_ladder(vm, econ, ctx).anchor, ctx.money_step)


def _max_pay_per_video(vm, econ, ctx) -> float:
    """Oblsk's ceiling: the ladder's walk-away, where the p10 downside still
    clears the minimum ROI (or the CPM cap, whichever binds first)."""
    return _ladder(vm, econ, ctx).walk_away


def _revised_flat_total(vm, econ, ctx, ask_per_video: Optional[float] = None) -> float:
    """Our revised single-video number after the opening flat is rejected: move up
    to the ladder's target — the fee that still hits the campaign's ROI goal at
    expected delivery — or meet the creator's ask if it sits below the target.
    The walk-away above the target is held in reserve for judging their asks,
    never offered proactively."""
    lad = _ladder(vm, econ, ctx)
    up_to = lad.target if not ask_per_video else min(lad.target, ask_per_video)
    return max(_round_money(up_to, ctx.money_step), _flat_total(vm, econ, ctx))


def _bundle_per_video(vm, econ, ctx, ask_per_video: Optional[float] = None) -> float:
    """The per-video rate for a bundle. Start from the ladder's target less a bulk
    discount (standard for committing to several videos at once); if the creator
    has revealed a higher per-video ask, meet them just under it so the rate is
    one they will actually accept. Never exceed the walk-away ceiling."""
    lad = _ladder(vm, econ, ctx)
    floor = lad.target * (1.0 - ctx.bundle_bulk_discount)
    per_video = floor
    if ask_per_video:
        per_video = max(floor, ask_per_video * (1.0 - ctx.ask_bulk_discount))
    return min(per_video, lad.walk_away)


def _bundle_total(vm, econ, ctx, ask_per_video: Optional[float] = None,
                  target_videos: Optional[int] = None) -> float:
    videos = target_videos or ctx.target_videos
    per_video = _bundle_per_video(vm, econ, ctx, ask_per_video)
    return _round_money(per_video * videos, ctx.bundle_money_step)


def _willingness_per_video(state, vm, econ, ctx) -> float:
    """The most per video we would happily pay right now. We concede up the
    ladder as the thread progresses: the target while the simple flat deal is
    live, the walk-away once we have already revised (closing under the ceiling
    beats losing the deal), and once a bundle is on the table, the rate of the
    bundle we put there (a composed bundle already clears ROI by construction)."""
    if state.bundle_offered and state.concession_history:
        last = deal_from_dict(state.concession_history[-1].deal)
        if isinstance(last, Bundle):
            return last.flat_per_video
    if state.bundle_offered:
        return _bundle_per_video(vm, econ, ctx,
                                 ask_per_video=state.creator_first_ask)
    lad = _ladder(vm, econ, ctx)
    return lad.walk_away if state.flat_revised else lad.target


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


def _is_high_value(vm: ViewModel, econ: CreatorEconomics,
                   ctx: CampaignContext) -> bool:
    return vm.median_views * econ.revenue_per_view > ctx.high_value_threshold_usd


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

    reserve = bundle_from_flat(deal, target_videos=ctx.target_videos,
                               bulk_discount=ctx.bundle_bulk_discount)
    reserve_ev = _score(reserve, vm, econ, seed=4)
    return Decision(
        action=Action.OPENING_FLAT, deal=deal, ev=ev,
        rationale=f"Open with one flat number, ${deal.total_usd:,.0f} for a single "
                  f"video ({reason}). ROI {ev.roi_mean:.1f}x. If rejected, go to a "
                  f"{reserve.video_count}-video bundle.",
        reserve_deal=reserve, reserve_ev=reserve_ev,
        requires_approval=_requires_approval(Action.OPENING_FLAT, deal.total_usd,
                                             state, ctx))


def _escalate_bundle(state, vm, econ, ctx, ask, ask_per_video=None, *,
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
            rate_card, catalog, roi_hurdle=ctx.roi_target,
            primary_fmt=brief.primary_format,
            allowed_formats=brief.allowed_formats)
        if composed is not None:
            return _escalate_multiformat(state, vm, econ, ctx, ask, catalog,
                                         composed)

    # The flat counter is a per-single-video number, so it anchors the bundle's
    # per-video rate. Hold that rate near their ask and add videos: each one is an
    # independent draw, so the total return scales while the rate stays fair.
    target_videos = ctx.target_videos
    per_video = _bundle_per_video(vm, econ, ctx, ask_per_video=ask_per_video)
    total = _round_money(per_video * target_videos, ctx.bundle_money_step)
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


def _accept_ask(msg, state, vm, econ, ctx, accept_ceiling_pv) -> Decision:
    """Accept the creator's ask. `accept_ceiling_pv` is the exact per-video gate
    the tree measured the ask against (willingness plus the thin-gap margin,
    capped at the walk-away) — display that, not a looser number, so the
    rationale never reads as accepting above its own stated ceiling."""
    ask = msg.ask_total_usd
    videos = max(msg.ask_video_count or (
        state.concession_history[-1].offered_video_count
        if state.concession_history else 1), 1)
    deal = flat_offer(ask, video_count=videos)
    ev = _score(deal, vm, econ, seed=5)

    state.status = Status.ACCEPTED
    state.phase = Phase.CLOSING
    state.round_count += 1
    state.record_concession(deal_to_dict(deal), deal.total_usd, ask,
                            video_count=deal.video_count)
    state.final_total = deal.total_usd
    state.final_deal = deal_to_dict(deal)
    per_video = deal.total_usd / deal.video_count
    return Decision(
        action=Action.ACCEPT, deal=deal, ev=ev,
        rationale=f"Accept ${ask:,.0f} across {deal.video_count} video(s) "
                  f"(${per_video:,.0f}/video) — within our ${accept_ceiling_pv:,.0f}"
                  f"/video ceiling, ROI {ev.roi_mean:.1f}x.",
        requires_approval=_requires_approval(Action.ACCEPT, deal.total_usd,
                                             state, ctx))


def _soft_close(state, vm, econ, ctx) -> Decision:
    state.status = Status.REJECTED
    high_value = _is_high_value(vm, econ, ctx)
    note = " High-value creator, flag for follow-up." if high_value else ""
    return Decision(
        action=Action.SOFT_CLOSE, deal=None, ev=None,
        rationale="Not interested. Close gently, keep the door open." + note,
        requires_approval=_requires_approval(Action.SOFT_CLOSE, None, state, ctx) or high_value)


def _escalate_human(state, vm, econ, ctx, ask, *, extreme=False,
                    reason: Optional[str] = None,
                    catalog: Optional[FormatCatalog] = None) -> Decision:
    state.status = Status.ESCALATED
    ask_str = f"${ask:,.0f}" if ask is not None else "their position"
    if reason:
        state.escalation_reason = reason
    elif extreme:
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
    _ladder: Optional[PriceLadder] = field(default=None, repr=False)

    @property
    def ask(self) -> Optional[float]:
        """The creator's ask as a total. Capturing the first ask normalizes it to
        per-video, which is the unit every consumer (bundle anchoring, the
        discount metric) reads it in."""
        a = self.msg.ask_total_usd
        if (a is not None and self.state.flat_offer_made
                and self.state.creator_first_ask is None):
            self.state.creator_first_ask = a / max(self.msg.ask_video_count or 1, 1)
        return a

    @property
    def ask_per_video(self) -> Optional[float]:
        """Their ask normalized per video, so a 3-video counter is judged against
        per-video willingness instead of looking three times too high."""
        a = self.ask
        if a is None:
            return None
        return a / max(self.msg.ask_video_count or 1, 1)

    @property
    def ladder(self) -> PriceLadder:
        if self._ladder is None:
            self._ladder = _ladder(self.vm, self.econ, self.ctx)
        return self._ladder

    @property
    def willingness(self) -> float:
        """Per-video, matched against ask_per_video."""
        if self._willingness is None:
            self._willingness = _willingness_per_video(
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
    if topic == "budget":
        if not state.flat_offer_made:
            return _opening_flat(state, dctx.vm, dctx.econ, ctx,
                                 reason="asked about budget")
        # An offer is already on the table; a money question restates it
        # rather than promising a new number.
        return _hold_position(dctx)
    if not can_answer(msg.raw_text, brief):
        # A pure call request often reads as a question; a call answers it.
        if msg.wants_call and not state.call_proposed:
            return _propose_call(dctx)
        return _ask_human(dctx,
                          need=f"The creator asked something the brief does not "
                               f"cover: \"{msg.raw_text.strip()}\". Reply with "
                               f"the answer (or how to handle it) and I will "
                               f"draft the response.")
    state.questions_answered += 1
    state.phase = Phase.QA
    # Conversation-aware: do not answer in place forever. Once several questions
    # are answered and no number is on the table, the next step is a concrete
    # offer, so open it instead of fielding another question.
    if not state.flat_offer_made and state.questions_answered >= ctx.qa_offer_after:
        return _opening_flat(state, dctx.vm, dctx.econ, ctx,
                             reason="several questions answered, time for a number")
    answer = answer_question(msg.raw_text, brief)
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


def _ask_human(dctx: DecisionContext, need: str) -> Decision:
    """The agent is missing context only the team has (a question outside the
    brief, a reference to a call it was not on). Pause the thread and ask for
    exactly what is needed; nothing goes to the creator until the team replies.
    This is different from ESCALATE_HUMAN: the agent keeps the thread — it just
    needs an input."""
    dctx.state.notes.append(f"awaiting team input: {need}")
    return Decision(
        action=Action.ASK_HUMAN, deal=None, ev=None,
        human_prompt=need,
        rationale="Missing context only the team has. Pause and ask rather "
                  "than guess; the thread stays with the agent.",
        requires_approval=False)


def _ask_human_call_context(dctx: DecisionContext) -> Decision:
    return _ask_human(dctx,
                      need=f"The creator referenced a call/conversation I was "
                           f"not part of: \"{dctx.msg.raw_text.strip()[:300]}\". "
                           f"What was discussed or agreed that I should know "
                           f"before replying?")


def _propose_call(dctx: DecisionContext) -> Decision:
    """They want a call, and no counter is on the table to price. Get the call
    scheduled — the agent proposes our windows and confirms, and a real person
    runs the call. The team is flagged to own it from here."""
    state, ctx = dctx.state, dctx.ctx
    state.call_proposed = True
    windows = ctx.call_windows or "a couple of slots later this week"
    deal = (deal_from_dict(state.concession_history[-1].deal)
            if state.concession_history else None)
    ev = (_score_any(deal, dctx.vm, dctx.econ, catalog=dctx.catalog, seed=17)
          if deal is not None else None)
    return Decision(
        action=Action.PROPOSE_CALL, deal=deal, ev=ev,
        answer_text=windows,   # the availability the message should state
        human_prompt="Creator is ready for an intro call. A teammate needs to "
                     "send the invite and run the call (the agent cannot join); "
                     "afterwards, note anything agreed back into this thread so "
                     "the agent has the context.",
        rationale=f"They want a call and there is no new number to price. "
                  f"Propose {windows} and hand the call itself to a person — "
                  f"a live conversation moves this deal better than another "
                  f"email.",
        requires_approval=_requires_approval(
            Action.PROPOSE_CALL, deal.total_usd if deal else None, state, ctx))


def _hold_position(dctx: DecisionContext) -> Decision:
    """The creator followed up without a new number (a nudge, a 'still
    interested', an unclear note). Restate the offer on the table and keep the
    thread warm. Crucially, this costs no round and moves no money: a follow-up
    is not a reason to bid against ourselves."""
    state = dctx.state
    last = state.concession_history[-1]
    deal = deal_from_dict(last.deal)
    ev = _score_any(deal, dctx.vm, dctx.econ, catalog=dctx.catalog, seed=13)
    return Decision(
        action=Action.HOLD_FIRM, deal=deal, ev=ev,
        rationale=f"No new ask in their message. Hold ${deal.total_usd:,.0f} on "
                  f"the table and keep the thread warm; concede nothing without "
                  f"a counter.",
        requires_approval=_requires_approval(Action.HOLD_FIRM, deal.total_usd,
                                             state, dctx.ctx))


def _revise_flat(dctx: DecisionContext) -> Decision:
    """The opening flat was rejected. Before restructuring into a bundle, make one
    revised single-video offer at the top of what the numbers allow: meet the
    creator up to our ROI ceiling. Only if this is rejected too do we move to a
    bundle, so we have tried hardest on the simplest deal first."""
    state, vm, econ, ctx = dctx.state, dctx.vm, dctx.econ, dctx.ctx
    deal = flat_offer(_revised_flat_total(vm, econ, ctx, dctx.ask_per_video),
                      video_count=1)
    ev = _score(deal, vm, econ, seed=6)

    state.flat_revised = True
    state.phase = Phase.NEGOTIATING
    state.round_count += 1
    state.record_concession(deal_to_dict(deal), deal.total_usd, dctx.ask,
                            video_count=deal.video_count)

    reserve = bundle_from_flat(deal, target_videos=ctx.target_videos,
                               bulk_discount=ctx.bundle_bulk_discount)
    reserve_ev = _score(reserve, vm, econ, seed=8)
    ask_str = f"${dctx.ask:,.0f}" if dctx.ask is not None else "the flat rate"
    return Decision(
        action=Action.OPENING_FLAT, deal=deal, ev=ev,
        rationale=f"Flat rejected ({ask_str}). Hold the single video but revise up "
                  f"to ${deal.total_usd:,.0f} — our target rung (ROI "
                  f"{ev.roi_mean:.1f}x); the walk-away ceiling is never offered. "
                  f"If rejected again, restructure into a bundle.",
        reserve_deal=reserve, reserve_ev=reserve_ev,
        requires_approval=_requires_approval(Action.OPENING_FLAT, deal.total_usd,
                                             state, ctx))


def _accept_threshold_per_video(d: DecisionContext) -> float:
    """The most per video we will accept right now: willingness plus the
    thin-gap margin, hard-capped at the walk-away. The margin exists to close
    small gaps below the ceiling — it never crosses it (paying above the
    walk-away means an ordinary p10 underperformance loses money). The one
    exception is a composed multi-format bundle on the table: it prices off
    each format's own reach, so its blended rate is its own cap and the
    single-format ladder does not apply."""
    threshold = d.willingness * d.ctx.accept_margin
    state = d.state
    if (state.bundle_offered and state.concession_history and
            isinstance(deal_from_dict(state.concession_history[-1].deal), Bundle)):
        return threshold
    return min(threshold, d.ladder.walk_away)


# ---- registries: names the spec resolves against ---------------------------

GUARDS = {
    "human_intervened": lambda d: d.state.human_intervened(
        d.thread_last_sender, d.thread_last_ts),
    # A yes only locks terms when there are terms: with nothing on the table
    # yet, an eager first message falls through to the opening branch instead.
    "creator_accepted": lambda d: (d.msg.intent == Intent.ACCEPTED
                                   and bool(d.state.concession_history)),
    "creator_asked": lambda d: d.msg.intent == Intent.QUESTION,
    "contract_terms": lambda d: d.msg.contract_terms,
    # A yes to terms we have on record can close; any other reference to a
    # call the agent was not on (including a yes to unknown call-agreed terms)
    # needs the team's context first.
    "call_context_missing": lambda d: (
        d.msg.refers_to_call and not (d.msg.intent == Intent.ACCEPTED
                                      and d.state.concession_history)),
    "wants_call": lambda d: (
        d.msg.wants_call and d.ask is None and not d.state.call_proposed),
    "no_offer_yet": lambda d: not d.state.flat_offer_made,
    "no_new_ask": lambda d: (
        d.ask is None and d.msg.intent in (Intent.INTERESTED, Intent.UNCLEAR)
        and bool(d.state.concession_history)),
    "ask_acceptable": lambda d: (
        d.ask_per_video is not None and
        d.ask_per_video <= _accept_threshold_per_video(d)),
    "ask_extreme": lambda d: (
        d.ask_per_video is not None and
        d.ask_per_video > d.ladder.walk_away * d.ctx.extreme_ask_multiple),
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
    "ask_human_call_context": _ask_human_call_context,
    "propose_call": _propose_call,
    "hold_position": _hold_position,
    "revise_flat": _revise_flat,
    "accept_ask": lambda d: _accept_ask(
        d.msg, d.state, d.vm, d.econ, d.ctx, _accept_threshold_per_video(d)),
    "escalate_human_extreme": lambda d: _escalate_human(
        d.state, d.vm, d.econ, d.ctx, d.ask, extreme=True, catalog=d.catalog),
    "escalate_human_contract": lambda d: _escalate_human(
        d.state, d.vm, d.econ, d.ctx, d.ask, catalog=d.catalog,
        reason="They are negotiating contract terms (exclusivity, equity, kill "
               "fee, or similar), not just price. A person should own changes "
               "to the paper."),
    "escalate_human": lambda d: _escalate_human(
        d.state, d.vm, d.econ, d.ctx, d.ask, catalog=d.catalog),
    "escalate_bundle": lambda d: _escalate_bundle(
        d.state, d.vm, d.econ, d.ctx, d.ask, d.ask_per_video,
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
