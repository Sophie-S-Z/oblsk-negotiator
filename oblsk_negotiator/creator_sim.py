"""
A simulated creator for end-to-end testing. Rule-based and deterministic, so full
negotiations run without real people. A creator can open with questions, open with
interest and let the agent move first, or open by naming a number; every other
disposition (how hard they counter, whether volume moves them, when they walk) is
a constructor parameter, not a named persona.

A creator values pay *per video*: their floor is a per-video reservation, not a
flat total. Committing to several videos at once is worth a little to them
(guaranteed work, one negotiation), so they will accept a small bulk discount per
video, but only down to their floor and only up to a sane number of videos. This
is why a bundle that triples the work for the same money is correctly rejected.

For testing the agent against *real* threads instead of a simulation, see
replay.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .behavior_tree import CreatorMessage, Intent, Decision, Action
from .ev_engine import Bundle
from .rate_card import RateCard


@dataclass
class SimCreator:
    reservation_per_video: float       # private floor per video they will accept
    opening_ask: float = 0.0           # per-video number, if they open by naming one
    opens_with: str = "interest"       # question | interest | ask
    questions: list[str] = field(default_factory=list)
    counter_ratio: float = 0.6
    bulk_tolerance: float = 0.0         # bulk discount per video they'll grant for volume
    max_videos: int = 4                 # most videos they'll commit to at once
    rate_card: Optional[RateCard] = None  # their list prices, for judging a bundle
    max_bundle_discount: float = 0.20    # most off list they'll take on a package
    walks_if_lowballed: bool = False     # rejects outright after repeated low offers
    rng_seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.rng_seed)
        self._pending = list(self.questions)
        self._last_ask_pv = self.opening_ask if self.opening_ask else None
        self._last_bundle_ask = None
        self._rounds = 0

    def first_message(self) -> CreatorMessage:
        if self.opens_with == "question" and self._pending:
            return CreatorMessage(intent=Intent.QUESTION, raw_text=self._pending.pop(0))
        if self.opens_with == "ask" and self.opening_ask:
            return CreatorMessage(
                intent=Intent.NEGOTIATING, ask_total_usd=self.opening_ask,
                ask_video_count=1,
                raw_text=f"Thanks for reaching out. For a sponsorship I'd be around "
                         f"${self.opening_ask:,.0f}.")
        return CreatorMessage(intent=Intent.INTERESTED,
                              raw_text="Thanks for reaching out, I'd be interested. "
                                       "What did you have in mind?")

    def respond_to(self, decision: Decision) -> CreatorMessage:
        a = decision.action
        self._rounds += 1

        if a == Action.ANSWER_QUESTION:
            if self._pending:
                return CreatorMessage(intent=Intent.QUESTION, raw_text=self._pending.pop(0))
            return CreatorMessage(intent=Intent.INTERESTED,
                                  raw_text="Got it, thanks. So what were you "
                                           "thinking for this?")

        if a in (Action.OPENING_FLAT, Action.ESCALATE_BUNDLE, Action.ADD_SWEETENER):
            d = decision.deal
            if isinstance(d, Bundle):
                return self._respond_to_bundle(d, a)
            offered_per_video = d.effective_per_video
            offered_videos = d.video_count

            # A bundle grants a small bulk discount off the per-video floor, but
            # only for a sane number of videos. A sweetener nudges value up a touch.
            floor = self.reservation_per_video
            if offered_videos > 1:
                floor *= (1.0 - self.bulk_tolerance)
            perceived_per_video = offered_per_video
            if a == Action.ADD_SWEETENER:
                perceived_per_video *= 1.04

            too_many = offered_videos > self.max_videos
            if (not too_many and
                    perceived_per_video >= floor * self._rng.normal(1.0, 0.04)):
                return CreatorMessage(intent=Intent.ACCEPTED,
                                      raw_text="That works for me, let's do it.")

            if (self.walks_if_lowballed and
                    offered_per_video < 0.7 * self.reservation_per_video and
                    self._rounds >= 3):
                return CreatorMessage(intent=Intent.REJECTING,
                                      raw_text="That's still low for me, I'll pass.")

            anchor_pv = self._last_ask_pv if self._last_ask_pv \
                else self.reservation_per_video * 1.3
            new_pv = max(anchor_pv - self.counter_ratio * (anchor_pv - offered_per_video),
                         self.reservation_per_video)
            self._last_ask_pv = new_pv
            new_total = new_pv * offered_videos
            return CreatorMessage(intent=Intent.NEGOTIATING,
                                  ask_total_usd=round(new_total, -1),
                                  ask_video_count=offered_videos,
                                  raw_text=f"Could you do ${new_pv:,.0f} a video"
                                           f"{f' (${new_total:,.0f} for {offered_videos})' if offered_videos > 1 else ''}?")

        if a == Action.ACCEPT:
            return CreatorMessage(intent=Intent.ACCEPTED, raw_text="Great, thanks.")
        return CreatorMessage(intent=Intent.UNCLEAR, raw_text="Okay, thanks.")

    def _bundle_list_total(self, bundle: Bundle) -> float:
        """What this creator's own rate card would charge for the bundle's formats
        at full list (no discount). Falls back to a per-video proxy for any format
        not on their card."""
        total = 0.0
        for line in bundle.lines:
            price = self.rate_card.get(line.fmt) if self.rate_card else None
            if price is None:
                price = self.reservation_per_video
            total += float(price) * line.quantity
        return total

    def _respond_to_bundle(self, bundle: Bundle, action: Action) -> CreatorMessage:
        """Judge a multi-format package against list prices, not a per-video floor.
        Accept when the total we pay is at least their list less the bundle discount
        they will grant; otherwise counter toward their list total."""
        paid = bundle.total_usd
        list_total = self._bundle_list_total(bundle)
        floor = list_total * (1.0 - self.max_bundle_discount)
        perceived = paid * (1.04 if action == Action.ADD_SWEETENER else 1.0)
        if perceived >= floor * self._rng.normal(1.0, 0.03):
            return CreatorMessage(intent=Intent.ACCEPTED,
                                  raw_text="That works for me, let's do it.")
        anchor = self._last_bundle_ask if self._last_bundle_ask else list_total
        new_total = max(anchor - self.counter_ratio * (anchor - paid), floor)
        self._last_bundle_ask = new_total
        return CreatorMessage(
            intent=Intent.NEGOTIATING, ask_total_usd=round(new_total, -1),
            ask_video_count=bundle.video_count,
            raw_text=f"Could you get closer to ${new_total:,.0f} for the package?")
