"""
Buyer-side deal pricing: the negotiation ladder.

Ported from the influencer-calculator pricing-engine upgrade (PR #1,
eyalcohen2524/influencer-calculator). That engine turns a delivery estimate and
funnel economics into three rungs a negotiator can hold in their head:

    anchor      where to OPEN: low but credible, leaves room to move up
    target      where to AIM: hits the ROI goal at expected delivery
    walk_away   the CEILING: the downside still clears the minimum ROI
                (or a CPM cap, whichever binds first)

plus the ROI at each rung under downside / expected / upside delivery, so the
risk is visible, not just a point estimate.

Here the delivery estimate comes from the Monte Carlo in ev_engine rather than a
bootstrap: downside = p10 revenue, upside = p90, and the expected value is a
winsorized mean (the tail capped at p-90) so one simulated viral hit cannot
inflate the number we plan against — the same whale-capping the calculator PR
introduced, applied to the simulated draws instead of the raw history.

Also ported: the authenticity heuristic (obvious bought-follower / engagement-pod
patterns become a risk discount that tightens the ceiling) and the tiered CPM
benchmark table (placeholder rates, overridable per campaign).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .ev_engine import CreatorEconomics, ViewModel


@dataclass(frozen=True)
class PricingPolicy:
    """Oblsk's side of the ladder, per campaign. Nothing here is creator data."""
    roi_target: float = 3.0        # revenue/cost multiple we aim for at expected delivery
    roi_min: float = 1.0           # multiple the p10 downside must still clear
    anchor_factor: float = 0.62    # opening offer as a fraction of target
    cpm_cap_usd: Optional[float] = None  # optional cap on $ per 1,000 expected views
    risk_discount: float = 1.0     # authenticity multiplier applied to the downside
    winsor_q: float = 0.90         # cap simulated revenue draws at this quantile
    min_offer_usd: float = 250.0   # never open below this, even for tiny creators


@dataclass(frozen=True)
class PriceLadder:
    """The three rungs plus the numbers behind them. All totals for the deal as
    priced (video_count videos); per-video figures are totals / video_count."""
    anchor: float
    target: float
    walk_away: float
    break_even: float              # fee where expected value == cost (ROI = 1)
    video_count: int
    views_down: float
    views_exp: float
    views_up: float
    value_down: float              # p10 revenue, risk-discounted
    value_exp: float               # winsorized-mean revenue
    value_up: float                # p90 revenue
    binding: str                   # which constraint set the ceiling

    @property
    def anchor_per_video(self) -> float:
        return self.anchor / self.video_count

    @property
    def target_per_video(self) -> float:
        return self.target / self.video_count

    @property
    def walk_away_per_video(self) -> float:
        return self.walk_away / self.video_count

    def roi_at(self, fee: float) -> dict:
        """ROI under downside / expected / upside delivery at a given fee."""
        if fee <= 0:
            return {"down": float("inf"), "exp": float("inf"), "up": float("inf")}
        return {"down": self.value_down / fee, "exp": self.value_exp / fee,
                "up": self.value_up / fee}

    @property
    def downside_limited(self) -> bool:
        """True when the fee that would hit the ROI goal exceeds what the p10
        downside supports, so the target is clamped to the walk-away. Common
        for very heavy-tailed creators: the expectation (tail-driven) is huge
        relative to a typical bad outcome."""
        return self.target >= self.walk_away - 1e-9

    def summary(self) -> str:
        r = self.roi_at(self.target)
        note = ("  [downside-limited: the ROI-goal fee exceeds what a p10 "
                "outcome supports, so target = walk-away]"
                if self.downside_limited else "")
        return (f"anchor ${self.anchor:,.0f}  target ${self.target:,.0f}  "
                f"walk-away ${self.walk_away:,.0f}  "
                f"(ROI at target: {r['down']:.1f}x down / {r['exp']:.1f}x exp / "
                f"{r['up']:.1f}x up; ceiling set by {self.binding}){note}")


def price_ladder(vm: ViewModel,
                 econ: CreatorEconomics,
                 policy: PricingPolicy,
                 *,
                 video_count: int = 1,
                 n_samples: int = 8000,
                 seed: int = 11) -> PriceLadder:
    """Build the ladder for `video_count` videos of one creator.

    Three independent ceilings, exactly as in the calculator PR:
      fee_from_target    expected value / roi_target   (hit the goal)
      fee_from_downside  p10 value * risk / roi_min    (survive the downside)
      fee_from_cpm       cpm_cap * expected views/1000 (policy cap, if set)
    walk_away = min(downside, cpm); target = min(fee_from_target, walk_away);
    anchor = anchor_factor * target, floored at min_offer and capped at target.
    """
    if video_count < 1:
        raise ValueError("video_count must be >= 1")
    rng = np.random.default_rng(seed)
    views = vm.sample(n_samples * video_count, rng).reshape(n_samples, video_count)
    views_total = views.sum(axis=1)
    revenue = views_total * econ.revenue_per_view

    views_exp = float(views_total.mean())
    cap = float(np.quantile(revenue, policy.winsor_q))
    value_exp = float(np.minimum(revenue, cap).mean())
    value_down = float(np.percentile(revenue, 10)) * policy.risk_discount
    value_up = float(np.percentile(revenue, 90))

    fee_from_target = value_exp / policy.roi_target if policy.roi_target > 0 else float("inf")
    fee_from_downside = value_down / policy.roi_min if policy.roi_min > 0 else float("inf")
    fee_from_cpm = (policy.cpm_cap_usd * views_exp / 1000.0
                    if policy.cpm_cap_usd else float("inf"))

    walk_away = max(0.0, min(fee_from_downside, fee_from_cpm))
    target = max(0.0, min(fee_from_target, walk_away))
    anchor = min(max(policy.anchor_factor * target, policy.min_offer_usd), target)
    binding = "downside-ROI" if fee_from_downside <= fee_from_cpm else "CPM-cap"

    return PriceLadder(
        anchor=anchor, target=target, walk_away=walk_away, break_even=value_exp,
        video_count=video_count,
        views_down=float(np.percentile(views_total, 10)), views_exp=views_exp,
        views_up=float(np.percentile(views_total, 90)),
        value_down=value_down, value_exp=value_exp, value_up=value_up,
        binding=binding)


# ---- authenticity heuristic (buyer-side risk) --------------------------------
# Cheap, transparent checks — no ML. The point is to catch obvious
# bought-follower / engagement-pod patterns and turn them into a risk discount
# on the downside, which tightens the walk-away ceiling automatically.

def authenticity(posts: list[dict], followers: int) -> dict:
    """Score 0..1 plus human-readable flags. Each post is a dict with any of
    views / likes / comments / shares. Feed the score into
    PricingPolicy.risk_discount (score of 0.8 -> discount downside by 0.8)."""
    flags: list[str] = []
    score = 1.0
    if not posts or not followers:
        return {"score": 0.85, "flags": ["Not enough data to verify authenticity."]}

    likes = sum(p.get("likes", 0) for p in posts)
    comments = sum(p.get("comments", 0) for p in posts)
    per_follower = [(p.get("likes", 0) + p.get("comments", 0)) / followers
                    for p in posts]
    erf = float(np.mean([x for x in per_follower if x > 0]) * 100) \
        if any(x > 0 for x in per_follower) else None

    # 1. Engagement-per-follower far below tier norm => possible fake followers.
    if erf is not None:
        floor = 0.6 if followers > 1e6 else 1.0 if followers > 1e5 else 1.5
        if erf < floor * 0.5:
            score -= 0.3
            flags.append(f"Very low engagement ({erf:.2f}%) for the follower "
                         f"count — possible inflated following.")
        elif erf < floor:
            score -= 0.12
            flags.append(f"Below-benchmark engagement ({erf:.2f}%).")

    # 2. Comment-to-like ratio near zero => like inflation / bots.
    if likes > 0 and comments / likes < 0.001:
        score -= 0.2
        flags.append("Almost no comments relative to likes — like inflation is likely.")

    # 3. Median views a tiny fraction of followers => stale or bought following.
    view_posts = [p["views"] for p in posts if p.get("views", 0) > 0]
    if len(view_posts) >= 3 and float(np.median(view_posts)) / followers < 0.03:
        score -= 0.15
        flags.append("Median views are a tiny fraction of followers — reach may "
                     "be suppressed or followers inflated.")

    score = float(np.clip(score, 0.2, 1.0))
    if not flags:
        flags.append("No obvious authenticity red flags.")
    return {"score": score, "flags": flags}


# ---- market CPM benchmark (editable placeholder table) -----------------------
# $ per 1,000 views a brand typically pays, by platform and follower tier.
# Placeholders from the calculator PR — calibrate against real closed deals
# before trusting absolute dollars; override per campaign via `overrides`.

TIERS = [("nano", 1e4), ("micro", 1e5), ("mid", 5e5), ("macro", 1e6),
         ("mega", float("inf"))]

CPM_TABLE = {
    "tiktok":    {"nano": (8, 12, 18), "micro": (7, 11, 16), "mid": (6, 9, 14),
                  "macro": (5, 8, 12), "mega": (4, 6, 10)},
    "instagram": {"nano": (12, 18, 28), "micro": (10, 16, 24), "mid": (9, 14, 20),
                  "macro": (7, 12, 18), "mega": (6, 10, 15)},
}


def tier_for(followers: float) -> str:
    for name, cap in TIERS:
        if followers < cap:
            return name
    return TIERS[-1][0]


def cpm_benchmark(platform: str, followers: float,
                  overrides: Optional[dict] = None) -> dict:
    """(low, mid, high) $ per 1,000 views for this platform and tier."""
    tier = tier_for(followers or 0)
    table = (overrides or {}).get(platform) or CPM_TABLE.get(platform) or {}
    low, mid, high = table.get(tier, (6, 10, 16))
    return {"tier": tier, "low": low, "mid": mid, "high": high}
