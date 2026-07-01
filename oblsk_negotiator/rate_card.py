"""
The creator's asking side: a rate card is the list price the creator quoted for
each format they sell. It is paired with a FormatCatalog (the value side, in
ev_engine) at decision time; the negotiation closes the gap between what the
creator quoted and what the reach justifies at Oblsk's ROI hurdle.

Format names are canonical strings shared across the catalog, the rate card, and
the campaign brief. The bundle-discount figures below are not invented: they are
the discounts real creators already grant on their own packaged products. Across
the Lindseay, Latorre, and Chavez rate cards, an IG Story+Reel combo is priced
~10% under the two sold separately, and an IG/TikTok syndication ~15% under the
Reel and TikTok separately. The agent reuses those same figures so its bundles
look like the packages this market already prices, not a number it made up.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


# ---- canonical formats ------------------------------------------------------

IG_STORY = "ig_story"
IG_STORY_SET = "ig_story_set"
IG_REEL = "ig_reel"
IG_STATIC = "ig_static"
TIKTOK = "tiktok"
YT_SHORT = "yt_short"
YT_INTEGRATION = "yt_integration"
YT_DEDICATED = "yt_dedicated"

FORMAT_LABELS = {
    IG_STORY: "Instagram Story",
    IG_STORY_SET: "Instagram Story set",
    IG_REEL: "Instagram Reel",
    IG_STATIC: "Instagram in-feed post",
    TIKTOK: "TikTok video",
    YT_SHORT: "YouTube Short",
    YT_INTEGRATION: "YouTube integration",
    YT_DEDICATED: "YouTube dedicated video",
}

# Platform each format posts to. Used to detect cross-platform (syndication)
# bundles, which the market discounts harder than a single-platform combo.
PLATFORM_OF = {
    IG_STORY: "instagram", IG_STORY_SET: "instagram", IG_REEL: "instagram",
    IG_STATIC: "instagram", TIKTOK: "tiktok", YT_SHORT: "youtube",
    YT_INTEGRATION: "youtube", YT_DEDICATED: "youtube",
}

IG_FORMATS = {IG_STORY, IG_STORY_SET, IG_REEL, IG_STATIC}
_STORY_FORMATS = {IG_STORY, IG_STORY_SET}
_FEED_FORMATS = {IG_REEL, IG_STATIC}


# ---- market-standard bundle discounts (measured, not invented) --------------

COMBO_DISCOUNT = 0.10        # IG story + feed, one platform (measured ~10%)
SYNDICATION_DISCOUNT = 0.15  # same campaign across >1 platform (measured ~15%)
VOLUME_DISCOUNT = 0.10       # any other multi-post bundle on one platform


def platform_of(fmt: str) -> str:
    return PLATFORM_OF.get(fmt, "other")


def bundle_discount(formats) -> float:
    """The discount a creator will grant on the sum of list prices for this set
    of formats, picked to match how the market already prices these packages."""
    fs = set(formats)
    if len({platform_of(f) for f in fs}) > 1:
        return SYNDICATION_DISCOUNT
    if fs <= IG_FORMATS and (fs & _STORY_FORMATS) and (fs & _FEED_FORMATS):
        return COMBO_DISCOUNT
    return VOLUME_DISCOUNT


# ---- the rate card ----------------------------------------------------------

@dataclass(frozen=True)
class RateCard:
    """A creator's quoted list prices, one per format they sell. Prices are per
    single post of that format."""
    prices: dict          # fmt -> list price (USD)
    creator: str = ""

    def __contains__(self, fmt: str) -> bool:
        return fmt in self.prices

    def __getitem__(self, fmt: str) -> float:
        return self.prices[fmt]

    def get(self, fmt: str):
        return self.prices.get(fmt)

    def formats(self) -> list:
        return list(self.prices.keys())


# Keyword patterns for turning a pasted rate card into canonical formats. Order
# matters: more specific phrases (combo, syndication, story set) are checked
# before the plain single-format ones.
_PARSE_RULES = [
    (IG_STORY_SET, [r"story\s*set"]),
    (TIKTOK, [r"tiktok", r"tik tok"]),
    (YT_SHORT, [r"short"]),
    (YT_DEDICATED, [r"dedicated"]),
    (YT_INTEGRATION, [r"integration", r"\d+\s*-?\s*second youtube", r"youtube integration"]),
    (IG_REEL, [r"reel"]),
    (IG_STATIC, [r"static", r"in-?feed", r"feed post"]),
    (IG_STORY, [r"story", r"\big story\b"]),
]


def parse_rate_card(text: str, creator: str = "") -> RateCard:
    """Best-effort parse of a pasted rate card into canonical single-format
    prices. Combo and syndication lines are intentionally skipped: the agent
    composes those itself from the single-format prices and the measured
    discounts above, so it controls the package rather than taking the creator's.
    Lines it cannot confidently map are ignored."""
    prices: dict = {}
    for raw in text.splitlines():
        line = raw.strip().lower()
        if not line:
            continue
        money = re.search(r"\$\s*([\d,]+)", line)
        if not money:
            continue
        amount = float(money.group(1).replace(",", ""))
        if "combo" in line or "syndication" in line or "bundle" in line:
            continue  # composed by the agent, not taken from the card
        for fmt, pats in _PARSE_RULES:
            if any(re.search(p, line) for p in pats):
                prices.setdefault(fmt, amount)
                break
    return RateCard(prices=prices, creator=creator)
