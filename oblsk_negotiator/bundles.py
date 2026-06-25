"""
Deal library. Builds the deals the negotiator weighs. Pure: returns deals, does
not score them.

The negotiator uses two in order:
    flat_offer       the opening move, one video at a fair rate
    bundle_from_flat the escalation, more videos at the same fair per-video rate
Plus small non-price sweeteners (faster pay, usage rights, whitelisting).
"""

from __future__ import annotations

from dataclasses import replace
from .ev_engine import Deal


def flat_offer(total_usd: float, video_count: int = 1) -> Deal:
    return Deal(video_count=video_count, flat_per_video=total_usd / video_count)


def same_total_repackage(total_usd: float, video_counts=(1, 2, 3, 4)) -> list[Deal]:
    """Hold the total roughly constant, vary the video count. Same headline
    number to the creator, more content and more draws for us.

    Note: this collapses per-video pay as the count rises, so a real creator will
    not accept the higher-count versions. It exists only to isolate the effect of
    extra draws on Oblsk's side; do not use it to construct an actual offer."""
    return [Deal(video_count=k, flat_per_video=total_usd / k) for k in video_counts]


def bundle_from_flat(flat: Deal, *, target_videos: int = 3,
                     bulk_discount: float = 0.10) -> Deal:
    """Turn a one-video flat offer into the bundle we escalate to.

    Keep the per-video rate close to the flat rate (a small bulk discount is
    standard for committing to several videos at once) and let the total scale
    with the video count. Each added video is another independent draw from the
    tail, so Oblsk's absolute net value scales with the count while the creator
    is still paid fairly per video. The creator sees a bigger headline number
    *and* a fair rate, so it is an offer they will actually accept."""
    per_video = flat.effective_per_video * (1.0 - bulk_discount)
    return Deal(video_count=target_videos, flat_per_video=per_video)


def add_non_price_sweetener(deal: Deal, lever: str) -> Deal:
    if lever == "fast_pay":
        return replace(deal, net_terms_days=15, deposit_upfront_pct=0.25)
    if lever == "usage_rights":
        return replace(deal, usage_rights_months=6)
    if lever == "whitelisting":
        return replace(deal, whitelisting=True)
    raise ValueError(f"unknown lever {lever!r}")


LEVER_PITCH = {
    "fast_pay": "we can move you to net-15 with 25% upfront",
    "usage_rights": "we'll keep usage rights to 6 months instead of a longer window",
    "whitelisting": "we can add whitelisting so the content keeps working for you",
}
