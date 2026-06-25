"""
Simulated creators for end-to-end testing. Rule-based and deterministic, so full
negotiations run without real people. A creator can open with questions, open with
interest and let the agent move first, or open by naming a number.

A creator values pay *per video*: their floor is a per-video reservation, not a
flat total. Committing to several videos at once is worth a little to them
(guaranteed work, one negotiation), so they will accept a small bulk discount per
video, but only down to their floor and only up to a sane number of videos. This
is why a bundle that triples the work for the same money is correctly rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .behavior_tree import CreatorMessage, Intent, Decision, Action


class Personality(str, Enum):
    INQUISITIVE = "inquisitive"
    PRICE_SENSITIVE = "price_sensitive"
    VOLUME_FRIENDLY = "volume_friendly"
    AGGRESSIVE = "aggressive"
    EASYGOING = "easygoing"


@dataclass
class SimCreator:
    personality: Personality
    reservation_per_video: float       # private floor per video they will accept
    opening_ask: float = 0.0           # per-video number, if they open by naming one
    opens_with: str = "interest"       # question | interest | ask
    questions: list[str] = field(default_factory=list)
    counter_ratio: float = 0.6
    bulk_tolerance: float = 0.0         # bulk discount per video they'll grant for volume
    max_videos: int = 4                 # most videos they'll commit to at once
    rng_seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.rng_seed)
        self._pending = list(self.questions)
        self._last_ask_pv = self.opening_ask if self.opening_ask else None
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

            if (self.personality == Personality.AGGRESSIVE and
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


PERSONAS = {
    "inquisitive": lambda seed=0: SimCreator(
        Personality.INQUISITIVE, reservation_per_video=2450, opens_with="question",
        questions=["What would the deliverables be for this?",
                   "And what's the timeline looking like?"],
        counter_ratio=0.6, bulk_tolerance=0.06, max_videos=4, rng_seed=seed),
    "price_sensitive": lambda seed=0: SimCreator(
        Personality.PRICE_SENSITIVE, reservation_per_video=2450, opens_with="interest",
        counter_ratio=0.35, bulk_tolerance=0.0, max_videos=3, rng_seed=seed),
    "volume_friendly": lambda seed=0: SimCreator(
        Personality.VOLUME_FRIENDLY, reservation_per_video=2450, opens_with="interest",
        counter_ratio=0.6, bulk_tolerance=0.10, max_videos=5, rng_seed=seed),
    "aggressive": lambda seed=0: SimCreator(
        Personality.AGGRESSIVE, reservation_per_video=2700, opening_ask=3500,
        opens_with="ask", counter_ratio=0.45, bulk_tolerance=0.0, max_videos=3,
        rng_seed=seed),
    "easygoing": lambda seed=0: SimCreator(
        Personality.EASYGOING, reservation_per_video=1700, opens_with="interest",
        counter_ratio=0.7, bulk_tolerance=0.08, max_videos=5, rng_seed=seed),
}
