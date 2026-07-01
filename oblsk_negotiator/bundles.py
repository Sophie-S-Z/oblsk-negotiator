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
from typing import Optional
from .ev_engine import Deal, Line, Bundle, FormatCatalog
from .rate_card import RateCard, bundle_discount


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


# ---- multi-format bundle composition ----------------------------------------

def compose_bundle(
    rate_card: RateCard,
    catalog: FormatCatalog,
    *,
    roi_hurdle: float,
    primary_fmt: Optional[str] = None,
    allowed_formats: Optional[list] = None,
    max_formats: int = 3,
) -> Optional[Bundle]:
    """Build the most attractive bundle that still clears Oblsk's ROI hurdle.

    A creator's rate card mixes good-value formats (their list price sits below
    what the reach is worth to us) and rich ones (priced above it). We want a
    package the creator will sign, so we price every line at the creator's own
    list less the *market-standard* discount for that kind of bundle, and we only
    keep formats where doing so still clears the ROI hurdle. The result reads to
    the creator as a normal discounted combo, and to us as a margin-positive mix.

    Pure: scores candidates with the same conservative median-based value proxy
    the negotiator uses for ceilings, never a Monte Carlo. Returns None when no
    ROI-positive bundle exists at a discount a creator would accept, which is the
    signal to fall back to a reach-anchored single offer instead of overpaying.
    """
    allowed = allowed_formats or rate_card.formats()
    candidates = [f for f in allowed
                  if f in rate_card and f in catalog
                  and rate_card[f] and catalog[f].expected_value_per_unit > 0]
    if not candidates:
        return None

    value = {f: catalog[f].expected_value_per_unit for f in candidates}
    listed = {f: float(rate_card[f]) for f in candidates}
    # Efficiency = our value per dollar of their asking price. Higher is better.
    efficiency = {f: value[f] / listed[f] for f in candidates}

    anchor = primary_fmt if primary_fmt in candidates else \
        max(candidates, key=lambda f: efficiency[f])
    extras = sorted([f for f in candidates if f != anchor],
                    key=lambda f: efficiency[f], reverse=True)
    chosen = [anchor] + extras[:max_formats - 1]

    def roi_proxy(formats: list) -> float:
        disc = bundle_discount(formats) if len(formats) > 1 else 0.0
        total_value = sum(value[f] for f in formats)
        total_cost = sum(listed[f] * (1.0 - disc) for f in formats)
        return total_value / total_cost if total_cost > 0 else float("inf")

    # Drop the least efficient non-anchor format until the package clears ROI.
    while len(chosen) > 1 and roi_proxy(chosen) < roi_hurdle:
        worst = min((f for f in chosen if f != anchor), key=lambda f: efficiency[f])
        chosen.remove(worst)

    if roi_proxy(chosen) < roi_hurdle:
        return None  # even the anchor alone is too rich for its reach

    disc = bundle_discount(chosen) if len(chosen) > 1 else 0.0
    lines = tuple(
        Line(fmt=f, quantity=1, price_per_unit=round(listed[f] * (1.0 - disc), 2))
        for f in chosen)
    return Bundle(lines=lines)


def bundle_format_summary(bundle: Bundle) -> str:
    """Human phrase for the formats in a bundle, e.g. 'an IG Reel, an IG Story,
    and a TikTok'. Display only."""
    from .rate_card import FORMAT_LABELS
    names = [FORMAT_LABELS.get(l.fmt, l.fmt) for l in bundle.lines]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + (" and " + names[-1])
