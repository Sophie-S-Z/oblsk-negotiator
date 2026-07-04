# How the calculator prices a creator deal

This is the working explanation of the influencer calculator — the model behind
both the standalone tool ([eyalcohen2524/influencer-calculator](https://github.com/eyalcohen2524/influencer-calculator),
pricing-engine upgrade in PR #1, engine in `public/pricing.js`, full math in
`docs/PRICING_SPEC.md`) and the negotiation agent in this repo
(`oblsk_negotiator/pricing.py`, the Python port that prices every move the
agent makes). Same model, two frontends: the calculator is the interactive
what-if tool, the agent runs the ladder live in a thread.

The one-sentence version: **estimate what a creator's posts will honestly
deliver, translate delivery into dollars through our funnel, and derive three
prices — where to open, where to aim, and the ceiling we never cross.**

---

## 1. The delivery estimate (and why the old one was wrong)

The old model took the **median** of a creator's recent view counts and called
that the expected delivery. Two failure modes:

- **It ignored the tail.** Creator content is heavy-tailed: most posts land
  near the usual mark, a few run 10–30x past it. A median throws that upside
  away entirely — and the occasional viral hit is a real part of why a
  multi-video deal is worth more than one video.
- **The old Monte Carlo had the opposite bug too:** it sampled *without*
  replacement up to the pool size and reported a median-of-sums, so the
  per-video estimate *rose* as you booked more videos (23,400 → 42,410 views
  per video as the count went 1 → 10 — an artifact, not a fact). It was also
  unseeded, so the headline number jittered on every unrelated slider move.

The new delivery model, per platform:

1. **Recency weighting** — each post is weighted by `0.5^(age/90 days)`, so a
   rising creator is not dragged down by year-old posts (and a fading one is
   not propped up by them).
2. **Winsorization (whale-capping)** — views above the 90th percentile are
   capped *for the planning number*, so one viral outlier cannot inflate the
   "typical campaign" estimate. In the spec's worked example this is the whole
   story: raw mean 71,623 views vs. an honest central estimate of 20,699.
3. **Viral tail, reported separately** — posts above 3x the median are
   surfaced as `viralProb` and `viralMean` (e.g. "6.7% of posts run to
   ~725k"). The hits aren't deleted; they're *upside*, not baseline. You plan
   on the central number and you tell the client the lottery ticket comes free.
4. **Organic→sponsored haircut (0.8)** — sponsored posts under-deliver a
   creator's organic history; the central estimate is scaled accordingly.
5. **Seeded bootstrap band** — future posts are independent draws *with*
   replacement from the capped pool, giving a reproducible p10–p90 band per
   campaign (same inputs, same numbers, always — a hard requirement for
   anything used in a negotiation).
6. **Cross-post dedup** — when the same video runs on TikTok and Instagram,
   summing platform views double-counts the overlapping audience; a 25%
   overlap haircut against the smaller platform yields unique reach.

The negotiation agent's port keeps the same ideas but runs them on its Monte
Carlo: it fits a log-normal body with an explicit Pareto tail (so the upside
is *modeled*, not just reported), applies the recency weights and the
sponsored haircut in the fit, and winsorizes the simulated revenue draws when
computing the expected value the ladder prices from. Downside = p10 of the
simulation, upside = p90.

## 2. From views to dollars: the funnel

`Value(views) = views × (view→signup rate) × (signup→paying rate) × LTV × margin`

Applied to the downside / expected / upside delivery, this gives
`Value_down / Value_exp / Value_up` — the deal's value under pessimistic,
honest, and optimistic delivery. Every price below is derived from these three
numbers, which is what makes each price *defensible*: change the funnel
assumptions and every price moves with them, visibly.

## 3. The pricing ladder — the numbers you can defend to a client

Three independent ceilings:

| Ceiling | Formula | Meaning |
|---|---|---|
| `feeFromTargetRoi` | `Value_exp / ROI_target` | the fee that hits our ROI goal (default 3x) at expected delivery |
| `feeFromDownside` | `Value_down / ROI_min` | the fee at which even the p10 downside still clears the minimum ROI (default 1x = break even in the bad case) |
| `feeFromCpmCap` | `cap × views_exp / 1000` | an optional flat policy cap on $ per 1,000 expected views |

And from them, the ladder:

- **Anchor (where we open)** — ~62% of target, floored near market-low so the
  opening is credible rather than insulting. Justification to a creator: "this
  is what the expected delivery supports at our economics, with room to talk."
- **Target (fair fee at our goal)** — `min(feeFromTargetRoi, walkAway)`. The
  most we should *agree* to: at this fee, expected delivery returns our target
  multiple. This is the number to defend hardest.
- **Fair-market fee (context, not a commitment)** — the blended CPM benchmark
  mid × expected views. Tells you where the market would put the deal. When
  market sits above target (common), the read is: "the market rate is payable,
  but our ROI-optimal number is lower — negotiate below market." When market
  sits below our anchor, our open is generous and should close fast.
- **Break-even** — `Value_exp` itself: the fee at which expected ROI = 1.0.
  Useful in a client conversation as the absolute sanity line: *any* fee below
  this is expected-value positive; the question is only how positive.
- **Walk-away (never-exceed)** — `min(feeFromDownside, feeFromCpmCap)`.
  Above this, either a merely-unlucky campaign loses money or company CPM
  policy is broken. The engine reports which constraint is `binding`, so the
  negotiator knows which lever (ROI floor vs CPM policy) a stretch would
  actually bend.

Guaranteed by construction and by unit test: `anchor ≤ target ≤ walk-away`;
ROI at target ≥ the goal whenever the ROI constraint binds; downside ROI at
walk-away ≥ the floor whenever the downside binds.

**The client script, in one line per number:**
"We open at $X because that's what your expected reach supports at our
economics; we're happy at $Y, where the campaign pays 3-to-1; the market says
$M, which we can talk about; and we can't go past $Z, because past that an
ordinary underperformance — not a disaster, just a p10 post — puts the
campaign underwater."

Worked example from the spec (a micro creator, 5 videos, real engine output):
anchor **$996** → target **$1,366** (ROI 2.6x/3.0x/3.4x across
down/expected/up) → fair market **$2,104** → walk-away **$3,609**
(downside-ROI binding) → break-even **$4,097**.

## 4. The ROI surface — showing risk, not a point estimate

For any fee, the engine reports ROI under downside, expected, and upside
delivery. The negotiator (human or agent) sees a 3×3 grid — anchor / target /
walk-away × down / expected / up — instead of one number. That's the direct
answer to "what if the video flops?": at target, even the p10 outcome returns
2.6x in the worked example.

## 5. The authenticity heuristic (low-organic-engagement creators)

A brand pays for real, engaged reach; bought followers and engagement pods
inflate the topline. Three cheap, transparent checks — no ML, every flag
human-readable:

1. **Engagement per follower vs. tier norm** — (likes + comments) / followers
   against a floor that loosens with account size (1.5% under 100k followers,
   1.0% to 1M, 0.6% above). Far below floor: −0.30; merely below: −0.12.
2. **Comments-to-likes ratio** — near zero (< 0.001) suggests like inflation
   or bots: −0.20.
3. **Views-to-followers** — median views under 3% of followers suggests a
   stale or purchased following: −0.15.

The score (clamped to [0.2, 1.0]) becomes a **risk discount on the downside**:
`Value_down × score`. Because the walk-away ceiling is built from the
downside, a weak-authenticity creator automatically gets a *tighter* ceiling —
tested guarantee: lower authenticity ⇒ strictly lower walk-away. The flags
double as negotiation leverage ("engagement is 0.4% against a 1.5% norm for
this tier — we have to price for that risk").

## 6. What to calibrate before trusting absolute dollars

Placeholders, in priority order (spec §13 and §15 have the full parameter
table and data sources):

1. **Funnel numbers** (view→signup, signup→paying, LTV) — from attribution and
   product analytics. Everything scales linearly with these.
2. **CPM benchmark table** — currently rough US-market figures by platform and
   follower tier; replace with Oblsk's observed closed-deal rates.
3. **Sponsored factor (0.8)** — measure the organic→sponsored delivery ratio
   from past campaigns.
4. **Engagement floors** in the authenticity check — per-niche benchmarks
   (family/parenting content engages differently than gaming).

Relative comparisons (creator A vs. B, 1 video vs. 3) are robust before
calibration; absolute dollar outputs are only as good as the funnel inputs.

## 7. Where the agent uses each piece

| Calculator concept | In the negotiation agent |
|---|---|
| anchor | the opening flat offer |
| target | the one revised offer after pushback; the accept threshold early in a thread |
| walk-away | the accept ceiling late in a thread; asks above a multiple of it escalate to a person |
| binding constraint | reported in the ladder summary (`pricing.PriceLadder.summary()`) |
| authenticity → risk discount | `PricingPolicy.risk_discount` per creator |
| CPM cap | `cpm_cap_usd` in the campaign YAML |
| negotiation levers | the bundle escalation (volume is the buyer's strongest lever) and the fast-pay sweetener |

Backlog note: the calculator's usage-rights premium is not yet a first-class
input in either codebase — currently usage terms live in the campaign brief
text. A usage-rights toggle on the calculator (and a corresponding fee bump)
is a planned addition.
