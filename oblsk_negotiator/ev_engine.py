"""
The calculator. Prices any deal as a distribution of net value to Oblsk.

Creator views are heavy-tailed: most posts land near the median, a few go far
past it, and the mean sits well above the median because of those few. A model
built on the median alone misses this, which is why it can't see why more videos
from the same creator are worth more. So we fit a samplable view model and run a
Monte Carlo rather than returning a single number.

Two entry points, both pure:
    fit_view_model(post_views)                   -> ViewModel
    ev_distribution(view_model, deal, economics) -> EVResult

Flat per-video pricing only. Cost is fixed and known up front.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CreatorEconomics:
    """Revenue-side assumptions, from the client config rather than the agent."""
    conversion_rate: float      # fraction of views that convert
    ltv_usd: float              # value of one converted customer

    @property
    def revenue_per_view(self) -> float:
        return self.conversion_rate * self.ltv_usd


@dataclass(frozen=True)
class Deal:
    """A flat deal: pay flat_per_video per video, fixed, whatever the views do.

    The non-price levers cost the client close to nothing and carry real value to
    the creator, so they can close a small gap without moving the headline number.
    """
    video_count: int
    flat_per_video: float = 0.0

    usage_rights_months: int = 0
    whitelisting: bool = False
    net_terms_days: int = 30
    deposit_upfront_pct: float = 0.0

    @property
    def total_usd(self) -> float:
        return self.flat_per_video * self.video_count

    @property
    def effective_per_video(self) -> float:
        return self.flat_per_video


DealStructure = Deal  # alias kept for older call sites


# ---- multi-format deals -----------------------------------------------------
# A creator does not sell one homogeneous "video". They sell formats (an IG Reel,
# an IG Story, a TikTok, a YouTube integration, ...), each with its own rate card
# price *and* its own reach and conversion. A real deal is therefore a set of
# priced line items, not a single per-video number. The single-format Deal above
# is the one-line special case and is left untouched; the types below add the
# general case the negotiator escalates into when a rate card is on the table.

@dataclass(frozen=True)
class Line:
    """One format's contribution to a deal: `quantity` posts of `fmt` at
    `price_per_unit` each. Price is what *we* pay the creator, not their list
    price; the gap between the two is what the negotiation closes."""
    fmt: str
    quantity: int = 1
    price_per_unit: float = 0.0

    @property
    def subtotal(self) -> float:
        return self.price_per_unit * self.quantity


@dataclass(frozen=True)
class Bundle:
    """A multi-format deal: several formats, each its own line. Deal-level levers
    mirror Deal so the rest of the agent treats the two interchangeably.

    Exposes the same surface as Deal (total_usd, video_count, flat_per_video,
    effective_per_video) so existing consumers (prose, runner metrics, EV) read a
    Bundle without special-casing. video_count is the total unit count across
    lines; the per-video figures are blended averages, used only for display."""
    lines: tuple[Line, ...]
    usage_rights_months: int = 0
    whitelisting: bool = False
    net_terms_days: int = 30
    deposit_upfront_pct: float = 0.0

    @property
    def total_usd(self) -> float:
        return float(sum(l.subtotal for l in self.lines))

    @property
    def video_count(self) -> int:
        return int(sum(l.quantity for l in self.lines))

    @property
    def flat_per_video(self) -> float:
        n = self.video_count
        return self.total_usd / n if n else 0.0

    @property
    def effective_per_video(self) -> float:
        return self.flat_per_video

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = "bundle"
        return d


@dataclass(frozen=True)
class FormatSpec:
    """Our model of one format on one creator: how far a single post reaches
    (`reach`, a fitted view model) and what a reached viewer is worth (`econ`).
    This is the value side and is built from the creator's post history and the
    campaign config, entirely separate from the creator's quoted rate card."""
    name: str
    reach: "ViewModel"
    econ: CreatorEconomics

    @property
    def expected_value_per_unit(self) -> float:
        """Conservative (median-based) expected revenue for one post of this
        format, the same proxy the negotiator uses for ceilings."""
        return self.reach.median_views * self.econ.revenue_per_view


class FormatCatalog:
    """Name -> FormatSpec for one creator/campaign. The value side of the table;
    the creator's RateCard (in rate_card.py) is the asking side."""

    def __init__(self, specs: dict[str, FormatSpec]):
        self.specs = dict(specs)

    def __contains__(self, fmt: str) -> bool:
        return fmt in self.specs

    def __getitem__(self, fmt: str) -> FormatSpec:
        return self.specs[fmt]

    def get(self, fmt: str) -> Optional[FormatSpec]:
        return self.specs.get(fmt)

    def formats(self) -> list[str]:
        return list(self.specs.keys())


def deal_to_dict(deal) -> dict:
    """Serialize a Deal or Bundle with a kind tag so it round-trips."""
    if isinstance(deal, Bundle):
        return deal.to_dict()
    d = asdict(deal)
    d["kind"] = "deal"
    return d


def deal_from_dict(d: dict):
    """Inverse of deal_to_dict. Tolerates legacy dicts with no kind tag (Deal)."""
    d = dict(d)
    kind = d.pop("kind", "deal")
    if kind == "bundle":
        lines = tuple(Line(**l) for l in d.pop("lines"))
        return Bundle(lines=lines, **d)
    return Deal(**d)


DistFamily = Literal["lognormal", "pareto_lognorm", "empirical_prior"]


@dataclass
class ViewModel:
    """A fitted, samplable model of one creator's per-video views."""
    family: DistFamily
    params: dict
    n_posts_fit: int
    median_views: float
    notes: str = ""

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.family == "lognormal":
            mu, sigma = self.params["mu"], self.params["sigma"]
            return rng.lognormal(mean=mu, sigma=sigma, size=n)

        if self.family == "pareto_lognorm":
            mu, sigma = self.params["mu"], self.params["sigma"]
            alpha, x_min = self.params["alpha"], self.params["x_min"]
            tail_frac = self.params["tail_frac"]
            body = rng.lognormal(mean=mu, sigma=sigma, size=n)
            is_tail = rng.random(n) < tail_frac
            n_tail = int(is_tail.sum())
            if n_tail:
                body[is_tail] = (rng.pareto(alpha, size=n_tail) + 1.0) * x_min
            return body

        if self.family == "empirical_prior":
            prior = self.params["prior_samples"]
            scale = self.params.get("scale", 1.0)
            idx = rng.integers(0, len(prior), size=n)
            return prior[idx] * scale

        raise ValueError(f"unknown family {self.family!r}")

    def coefficient_of_variation(self, seed: int = 7, n: int = 5000) -> float:
        rng = np.random.default_rng(seed)
        s = self.sample(n, rng)
        return float(s.std() / s.mean()) if s.mean() > 0 else 0.0


def fit_view_model(
    post_views: list[float] | np.ndarray,
    *,
    min_posts: int = 6,
    prior_samples: Optional[np.ndarray] = None,
    tier_median: Optional[float] = None,
    timestamps: Optional[list[float]] = None,
    half_life_days: float = 90.0,
    sponsored_factor: float = 1.0,
) -> ViewModel:
    """Fit a samplable view model to a creator's recent per-video views.

    With enough history, fit a log-normal by MLE on log(views) and splice on a
    Pareto tail when the tail is heavier than log-normal allows (positive excess
    kurtosis in log space, alpha from a Hill estimate). With too little history,
    fall back to a tier prior scaled to whatever real posts exist.

    `timestamps` (epoch seconds, aligned with post_views) turns on recency
    weighting with the given half-life, so a rising creator is not dragged down
    by old posts. `sponsored_factor` is the organic->sponsored haircut: paid
    posts under-deliver organic history, so views are scaled by it before the
    fit (0.8 is the calculator's default; 1.0 leaves history untouched).
    """
    paired = [(x, timestamps[i] if timestamps is not None else None)
              for i, x in enumerate(post_views) if x is not None and x > 0]
    v = np.asarray([x for x, _ in paired], dtype=float) * float(sponsored_factor)
    n = len(v)

    weights = None
    if timestamps is not None and n > 0:
        ts = np.asarray([t if t else 0.0 for _, t in paired], dtype=float)
        known = ts[ts > 0]
        if len(known) >= max(3, n * 0.5):
            newest = float(known.max())
            age_days = np.where(ts > 0, np.maximum(0.0, (newest - ts) / 86400.0), 0.0)
            weights = np.power(0.5, age_days / float(half_life_days))

    if n < min_posts:
        if prior_samples is not None and len(prior_samples) > 0:
            scale = 1.0
            if n > 0 and tier_median is None:
                scale = float(np.median(v)) / float(np.median(prior_samples))
            elif tier_median is not None:
                scale = float(tier_median) / float(np.median(prior_samples))
            return ViewModel(
                family="empirical_prior",
                params={"prior_samples": np.asarray(prior_samples, float),
                        "scale": scale},
                n_posts_fit=n,
                median_views=float(np.median(prior_samples) * scale),
                notes=f"fallback prior ({n} posts, need {min_posts}); scaled x{scale:.2f}")
        if n == 0:
            raise ValueError("no usable post views and no prior provided")
        log_v = np.log(v)
        mu = float(log_v.mean())
        sigma = float(max(log_v.std(ddof=1) if n > 1 else 0.9, 0.9))
        return ViewModel(
            family="lognormal", params={"mu": mu, "sigma": sigma},
            n_posts_fit=n, median_views=float(np.exp(mu)),
            notes=f"thin data ({n} posts), inflated sigma, no prior")

    log_v = np.log(v)
    if weights is not None:
        w = weights / weights.sum()
        mu = float(np.sum(w * log_v))
        var = float(np.sum(w * (log_v - mu) ** 2)) * n / max(n - 1, 1)
        sigma = float(np.sqrt(var))
        wnote = f", recency-weighted ({half_life_days:.0f}d half-life)"
    else:
        mu = float(log_v.mean())
        sigma = float(log_v.std(ddof=1))
        wnote = ""
    if sponsored_factor != 1.0:
        wnote += f", sponsored haircut x{sponsored_factor:.2f}"
    log_kurt = float(stats.kurtosis(log_v, fisher=True, bias=False)) if n >= 4 else 0.0

    if log_kurt > 1.0 and n >= 10:
        x_min = float(np.quantile(v, 0.90))
        tail = v[v >= x_min]
        if len(tail) >= 3:
            alpha = float(np.clip(len(tail) / np.sum(np.log(tail / x_min)), 1.2, 4.0))
            tail_frac = float(len(tail) / n)
            return ViewModel(
                family="pareto_lognorm",
                params={"mu": mu, "sigma": sigma, "alpha": alpha,
                        "x_min": x_min, "tail_frac": tail_frac},
                n_posts_fit=n, median_views=float(np.exp(mu)),
                notes=f"lognormal body + Pareto tail (alpha={alpha:.2f}, "
                      f"x_min={x_min:.0f}, tail_frac={tail_frac:.2f}){wnote}")

    return ViewModel(
        family="lognormal", params={"mu": mu, "sigma": sigma},
        n_posts_fit=n, median_views=float(np.exp(mu)),
        notes=f"lognormal MLE (log-kurtosis={log_kurt:.2f}){wnote}")


@dataclass
class EVResult:
    """Distribution of net value to Oblsk for one deal."""
    revenue_p10: float
    revenue_p50: float
    revenue_p90: float
    revenue_mean: float
    cost_total: float
    net_p10: float
    net_p50: float
    net_p90: float
    net_mean: float
    roi_mean: float
    prob_net_positive: float
    deal: dict = field(default_factory=dict)
    view_model_notes: str = ""

    def summary(self) -> str:
        return (f"net mean ${self.net_mean:,.0f}  "
                f"[p10 ${self.net_p10:,.0f} .. p90 ${self.net_p90:,.0f}]  "
                f"cost ${self.cost_total:,.0f}  ROI {self.roi_mean:.1f}x  "
                f"P(net>0) {self.prob_net_positive:.0%}")


def _simulate_revenue(
    samplers: list[tuple["ViewModel", int, float]],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Total attributed revenue per Monte-Carlo draw. Each sampler is one format:
    (view model, unit count, revenue per view). For each draw we sample views for
    every unit of every format, turn them into revenue at that format's rate, and
    sum across formats. One shared rng keeps the whole simulation reproducible."""
    total = np.zeros(n_samples, dtype=float)
    for view_model, count, rev_per_view in samplers:
        if count <= 0:
            continue
        views = view_model.sample(n_samples * count, rng).reshape(n_samples, count)
        total += views.sum(axis=1) * rev_per_view
    return total


def _ev_from_revenue(revenue: np.ndarray, cost_total: float,
                     deal_dict: dict, notes: str) -> EVResult:
    """Build an EVResult from a revenue draw array and a fixed, known cost."""
    net = revenue - cost_total

    def pct(a, q):
        return float(np.percentile(a, q))

    roi = float(revenue.mean() / cost_total) if cost_total > 1e-9 else float("inf")
    return EVResult(
        revenue_p10=pct(revenue, 10), revenue_p50=pct(revenue, 50),
        revenue_p90=pct(revenue, 90), revenue_mean=float(revenue.mean()),
        cost_total=float(cost_total),
        net_p10=pct(net, 10), net_p50=pct(net, 50),
        net_p90=pct(net, 90), net_mean=float(net.mean()),
        roi_mean=roi, prob_net_positive=float((net > 0).mean()),
        deal=deal_dict, view_model_notes=notes)


def ev_distribution(
    view_model: ViewModel,
    deal: Deal,
    econ: CreatorEconomics,
    *,
    n_samples: int = 5000,
    seed: Optional[int] = None,
) -> EVResult:
    """Monte Carlo for a single-format flat deal. For each draw: sample views for
    every video, turn them into attributed revenue, subtract the fixed cost,
    record the net. Pure given seed.
    """
    if deal.video_count <= 0:
        raise ValueError("video_count must be >= 1")
    rng = np.random.default_rng(seed)
    revenue = _simulate_revenue(
        [(view_model, deal.video_count, econ.revenue_per_view)], n_samples, rng)
    return _ev_from_revenue(revenue, float(deal.total_usd),
                            asdict(deal), view_model.notes)


def ev_bundle(
    catalog: FormatCatalog,
    bundle: Bundle,
    *,
    n_samples: int = 5000,
    seed: Optional[int] = None,
) -> EVResult:
    """Monte Carlo for a multi-format deal. Each line draws from its own format's
    reach model and converts at its own rate; the cost is the fixed bundle total.
    Same heavy-tailed core as the single-format path, just summed over formats."""
    if not bundle.lines:
        raise ValueError("bundle has no lines")
    missing = [l.fmt for l in bundle.lines if l.fmt not in catalog]
    if missing:
        raise ValueError(f"no FormatSpec for {missing!r} in catalog")
    rng = np.random.default_rng(seed)
    samplers = [(catalog[l.fmt].reach, l.quantity, catalog[l.fmt].econ.revenue_per_view)
                for l in bundle.lines]
    revenue = _simulate_revenue(samplers, n_samples, rng)
    notes = "; ".join(f"{l.fmt}x{l.quantity}: {catalog[l.fmt].reach.notes}"
                      for l in bundle.lines)
    return _ev_from_revenue(revenue, float(bundle.total_usd),
                            bundle.to_dict(), notes)


def synthetic_tier_prior(tier_median: float, sigma: float = 0.8,
                         size: int = 2000, seed: int = 0) -> np.ndarray:
    """Stand-in tier prior for testing. In production this is built from real
    historical views across creators in the same platform and follower tier."""
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=np.log(tier_median), sigma=sigma, size=size)
