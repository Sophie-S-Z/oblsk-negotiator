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
) -> ViewModel:
    """Fit a samplable view model to a creator's recent per-video views.

    With enough history, fit a log-normal by MLE on log(views) and splice on a
    Pareto tail when the tail is heavier than log-normal allows (positive excess
    kurtosis in log space, alpha from a Hill estimate). With too little history,
    fall back to a tier prior scaled to whatever real posts exist.
    """
    v = np.asarray([x for x in post_views if x is not None and x > 0], dtype=float)
    n = len(v)

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
    mu = float(log_v.mean())
    sigma = float(log_v.std(ddof=1))
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
                      f"x_min={x_min:.0f}, tail_frac={tail_frac:.2f})")

    return ViewModel(
        family="lognormal", params={"mu": mu, "sigma": sigma},
        n_posts_fit=n, median_views=float(np.exp(mu)),
        notes=f"lognormal MLE (log-kurtosis={log_kurt:.2f})")


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


def ev_distribution(
    view_model: ViewModel,
    deal: Deal,
    econ: CreatorEconomics,
    *,
    n_samples: int = 5000,
    seed: Optional[int] = None,
) -> EVResult:
    """Monte Carlo. For each draw: sample views for every video, turn them into
    attributed revenue, subtract the fixed cost, record the net. Pure given seed.
    """
    rng = np.random.default_rng(seed)
    k = deal.video_count
    if k <= 0:
        raise ValueError("video_count must be >= 1")

    views = view_model.sample(n_samples * k, rng).reshape(n_samples, k)
    revenue = views.sum(axis=1) * econ.revenue_per_view
    cost_total = float(deal.total_usd)
    net = revenue - cost_total

    def pct(a, q):
        return float(np.percentile(a, q))

    roi = float(revenue.mean() / cost_total) if cost_total > 1e-9 else float("inf")
    return EVResult(
        revenue_p10=pct(revenue, 10), revenue_p50=pct(revenue, 50),
        revenue_p90=pct(revenue, 90), revenue_mean=float(revenue.mean()),
        cost_total=cost_total,
        net_p10=pct(net, 10), net_p50=pct(net, 50),
        net_p90=pct(net, 90), net_mean=float(net.mean()),
        roi_mean=roi, prob_net_positive=float((net > 0).mean()),
        deal=asdict(deal), view_model_notes=view_model.notes)


def synthetic_tier_prior(tier_median: float, sigma: float = 0.8,
                         size: int = 2000, seed: int = 0) -> np.ndarray:
    """Stand-in tier prior for testing. In production this is built from real
    historical views across creators in the same platform and follower tier."""
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=np.log(tier_median), sigma=sigma, size=size)
