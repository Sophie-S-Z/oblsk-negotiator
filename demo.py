"""Run Oblsk negotiations from the terminal (no Jupyter needed).

Talk to it yourself — you play the creator, the agent replies live (this is the
exact pipeline the Oblsk platform calls in production, just typed by hand):
    py demo.py --chat                       # you type messages, it responds
    py demo.py --chat --campaign examples/unest_campaign.yaml --median 150000

See how it prices a deal, in plain English:
    py demo.py --explain                    # show the calculation, then a run

Simulated negotiations (a rule-based creator plays the other side):
    py demo.py                              # one fixed negotiation (reproducible)
    py demo.py --random                     # a NEW random creator every run
    py demo.py --floor 2800 --opens ask --ask 3500   # a harder creator
    py demo.py --count 20 --quiet           # 20 varied runs, one line each + metrics
    py demo.py --autonomy                   # drop the human-approval gate
    py demo.py --reject opening_flat        # human rejects the 1st offer -> handoff

Load a campaign (brief, economics, negotiation stance from one YAML file):
    py demo.py --campaign examples/unest_campaign.yaml --random

Replay a real thread the team ran manually, or spar against a Claude manager:
    py demo.py --campaign examples/unest_campaign.yaml \\
               --replay examples/unest_thread.txt --median 150000
    py demo.py --spar "floor $2,600/video, opens at $4,500, concedes slowly"

Set ANTHROPIC_API_KEY to have the LLM draft messages and read real threads;
without it, deterministic templates and keyword rules are used instead.

How to plug this into the Oblsk platform (Gmail threads, the review UI): see
docs/INTEGRATION.md. The `--chat` loop here is the same interpret -> decide ->
draft the platform runs; there, Gmail delivers the messages instead of you.
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np

from oblsk_negotiator.ev_engine import CreatorEconomics, fit_view_model
from oblsk_negotiator.behavior_tree import CampaignContext
from oblsk_negotiator.campaign import Campaign, load_campaign
from oblsk_negotiator.pricing import price_ladder
from oblsk_negotiator.state import AutonomyLevel
from oblsk_negotiator.qa import CampaignBrief
from oblsk_negotiator.creator_sim import SimCreator
from oblsk_negotiator.replay import replay_thread
from oblsk_negotiator.runner import run_negotiation, batch_metrics, reject_action

NAMES = ["Devin", "Riley", "Jordan", "Sam", "Alex", "Maya",
         "Chris", "Taylor", "Noah", "Priya", "Quinn", "Harper"]

# Used when no --campaign file is given.
DEMO_CAMPAIGN = Campaign(
    name="demo campaign",
    brief=CampaignBrief(brand="Aurora Skincare", product="the daily SPF serum"),
    econ=CreatorEconomics(conversion_rate=0.0016, ltv_usd=80),
    ctx=CampaignContext())


def view_model(rng: np.random.Generator):
    """A creator's view history: a heavy body around a random median, plus a
    couple of viral hits. Drawing it fresh per run varies what a deal is worth."""
    median = float(rng.uniform(35000, 75000))
    sigma = float(rng.uniform(0.55, 0.78))
    body = rng.lognormal(np.log(median), sigma, int(rng.integers(18, 28)))
    hits = median * rng.uniform(6, 18, size=int(rng.integers(1, 4)))
    return fit_view_model(np.concatenate([body, hits]))


def make_creator(args, seed: int) -> SimCreator:
    """A creator from the CLI flags."""
    return SimCreator(
        reservation_per_video=args.floor,
        opening_ask=args.ask if args.opens == "ask" else 0.0,
        opens_with=args.opens,
        questions=["What would the deliverables be for this?",
                   "And what's the timeline looking like?"]
        if args.opens == "question" else [],
        counter_ratio=args.counter, bulk_tolerance=args.bulk,
        max_videos=args.max_videos, rng_seed=seed)


def random_creator(rng: np.random.Generator, vm, camp: Campaign):
    """A randomized creator so no two runs in a batch behave the same. Their
    private floor is drawn relative to what this creator's reach is worth under
    the campaign's economics, so the sweep stays meaningful whatever campaign
    is loaded (an absolute-dollar floor would make every deal in a cheap-CPM
    campaign escalate and every deal in a rich one accept instantly)."""
    name = str(rng.choice(NAMES))
    opens = str(rng.choice(["question", "interest", "interest", "ask"]))
    target = price_ladder(vm, camp.econ, camp.ctx.pricing_policy()).target
    floor = target * float(rng.uniform(0.6, 1.4))
    creator = SimCreator(
        reservation_per_video=floor,
        opening_ask=floor * float(rng.uniform(1.2, 1.6)) if opens == "ask" else 0.0,
        opens_with=opens,
        questions=["What would the deliverables be for this?",
                   "And what's the timeline looking like?"]
        if opens == "question" else [],
        counter_ratio=float(rng.uniform(0.3, 0.75)),
        bulk_tolerance=float(rng.uniform(0.0, 0.12)),
        max_videos=int(rng.integers(3, 6)),
        walks_if_lowballed=bool(rng.random() < 0.25),
        rng_seed=int(rng.integers(0, 100000)))
    return creator, name


def resolve_seed(args, fallback: int) -> int:
    """Use --seed when given; otherwise the mode's fallback (a fixed number for
    reproducible modes, a fresh random one for --random)."""
    return args.seed if args.seed is not None else fallback


def explain_ladder(vm, econ, ctx) -> str:
    """Plain-English derivation of the three prices, so the numbers are never a
    black box: reach -> dollars -> the ladder."""
    lad = price_ladder(vm, econ, ctx.pricing_policy())
    rpv = econ.revenue_per_view
    out = [
        "How these prices are calculated (plain English):",
        f"  1. REACH  — we simulate this creator's next video 8,000 times from",
        f"     their view history. One video lands around {lad.views_exp:,.0f} views",
        f"     (a bad day {lad.views_down:,.0f}, a good day {lad.views_up:,.0f}).",
        f"  2. DOLLARS — each view is worth ${rpv:.4f} to us "
        f"({econ.conversion_rate:.3%} sign up x ${econ.ltv_usd:,.0f} each).",
        f"     So one video is worth about ${lad.value_exp:,.0f} to us "
        f"(${lad.value_down:,.0f} on a bad day).",
        f"  3. PRICES — from that value and our {ctx.roi_target:.0f}x return goal:",
        f"       anchor    ${lad.anchor:>8,.0f}  where we open (credible, leaves room)",
        f"       target    ${lad.target:>8,.0f}  fair fee that still returns {ctx.roi_target:.0f}x",
        f"       walk-away ${lad.walk_away:>8,.0f}  ceiling — above it a bad outcome loses money",
    ]
    if lad.downside_limited:
        out += [
            "     (target = walk-away here: this creator's views are so tail-heavy",
            "      that the fee hitting our ROI goal would lose money on a bad",
            "      outcome, so the ceiling caps it. Not a bug — the safe read.)"]
    return "\n".join(out)


def summary_lines(o) -> list[str]:
    out = [f"status              {o.status}",
           f"rounds              {o.rounds}",
           f"final total         {f'${o.final_total:,.0f}' if o.final_total else '-'}",
           f"videos              {o.final_video_count if o.final_video_count else '-'}",
           f"human took over     {o.human_rejected}",
           f"approvals requested {o.approvals_requested}"]
    if o.status == "accepted" and o.final_ev_roi is not None:
        out.append(f"final ROI           {o.final_ev_roi:.1f}x")
        out.append(f"final net (mean)    ${o.final_ev_net_mean:,.0f}")
    return out


def run_one(args, camp: Campaign) -> None:
    seed = resolve_seed(args, 2)
    vm = view_model(np.random.default_rng(seed))
    autonomy = AutonomyLevel.AUTONOMOUS if args.autonomy else AutonomyLevel.HUMAN_APPROVAL
    kwargs = {}
    if args.reject:
        kwargs["approver"] = reject_action(args.reject)

    if args.explain:
        print(explain_ladder(vm, camp.econ, camp.ctx) + "\n")
    print(f"view model: {vm.notes}")
    print(f"ladder:     {price_ladder(vm, camp.econ, camp.ctx.pricing_policy()).summary()}")
    # The floor is the SIMULATED creator's private minimum — a test-harness
    # parameter for the fake counterpart, hidden from the agent. Our side's
    # ceiling is the ladder's walk-away above.
    print(f"sim creator's hidden floor ${args.floor:,.0f}/video  |  "
          f"autonomy: {autonomy.value}  |  max rounds: {args.rounds}\n")

    o = run_negotiation(
        vm, camp.econ, camp.ctx, make_creator(args, seed),
        brief=camp.brief, creator_name=args.name, autonomy=autonomy,
        max_rounds=args.rounds, verbose=True, **kwargs)

    print("\n--- result ---")
    print("\n".join(summary_lines(o)))


def run_random(args, camp: Campaign) -> None:
    """One fresh, fully-detailed negotiation with a random creator — different
    every run (pass --seed to reproduce one). Mimics a real inbound: a creator
    we've not priced yet, negotiated end to end with the reasoning shown."""
    seed = resolve_seed(args, random.randrange(1_000_000))
    rng = np.random.default_rng(seed)
    vm = view_model(rng)
    creator, name = random_creator(rng, vm, camp)
    autonomy = AutonomyLevel.AUTONOMOUS if args.autonomy else AutonomyLevel.HUMAN_APPROVAL

    header = f"random creator: {name}   (seed {seed} — pass --seed {seed} to replay)"
    print(header)
    print("=" * len(header) + "\n")
    print(explain_ladder(vm, camp.econ, camp.ctx) + "\n")
    print(f"view model: {vm.notes}")
    print(f"ladder:     {price_ladder(vm, camp.econ, camp.ctx.pricing_policy()).summary()}")
    print(f"sim creator's hidden floor ${creator.reservation_per_video:,.0f}/video  |  "
          f"autonomy: {autonomy.value}  |  max rounds: {args.rounds}\n")

    o = run_negotiation(vm, camp.econ, camp.ctx, creator, brief=camp.brief,
                        creator_name=name, autonomy=autonomy,
                        max_rounds=args.rounds, verbose=True)
    print("\n--- result ---")
    print("\n".join(summary_lines(o)))


def run_chat(args, camp: Campaign) -> None:
    """Interactive: you are the creator, the agent replies. This is the same
    interpret -> decide -> draft pipeline the Oblsk platform runs on real Gmail
    threads (docs/INTEGRATION.md); here you type the messages instead."""
    from oblsk_negotiator.behavior_tree import Action, decide
    from oblsk_negotiator.prose import write_message
    from oblsk_negotiator.replay import interpret_message
    from oblsk_negotiator.state import NegotiationState

    seed = resolve_seed(args, 2)
    vm = fit_view_model(np.random.default_rng(seed).lognormal(np.log(args.median), 0.6, 24))
    print(explain_ladder(vm, camp.econ, camp.ctx))
    print(f"\nladder: {price_ladder(vm, camp.econ, camp.ctx.pricing_policy()).summary()}")
    print("\nYou are the CREATOR. Type a message and the agent replies. It reads "
          "your\nmessage, prices the move off the ladder above, and drafts a "
          "reply.\nType 'quit' to stop.\n")

    state = NegotiationState("chat", camp.name, "chat", max_rounds=args.rounds)
    history: list[str] = []
    terminal = {Action.ACCEPT, Action.SOFT_CLOSE, Action.ESCALATE_HUMAN}
    while True:
        try:
            text = input("CREATOR (you) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            break

        msg = interpret_message(text)
        decision = decide(msg, state, vm, camp.econ, camp.ctx, brief=camp.brief)
        reply = write_message(decision, creator_name=args.name,
                              thread_history="\n".join(history[-6:]))

        ask = f", ask ${msg.ask_total_usd:,.0f}" if msg.ask_total_usd else ""
        print(f"  read as: {msg.intent.value}{ask}")
        money = f" | ${decision.deal.total_usd:,.0f}" if decision.deal else ""
        roi = f" | ROI {decision.ev.roi_mean:.1f}x" if decision.ev else ""
        gate = ("would send after your OK" if decision.requires_approval
                else "auto-send")
        print(f"  AGENT [{decision.action.value}{money}{roi} | {gate}]:")
        print("    " + reply.replace("\n", "\n    "))
        print(f"  why: {decision.rationale}")
        if decision.human_prompt:
            print(f"  NEEDS A HUMAN: {decision.human_prompt}")
        print()

        history.append(f"creator: {text}")
        history.append(f"us: {reply}")
        if decision.action in terminal:
            print(f"--- thread {state.status.value} ---")
            break


def run_many(args, camp: Campaign) -> None:
    rng = np.random.default_rng(resolve_seed(args, 2))
    autonomy = AutonomyLevel.AUTONOMOUS if args.autonomy else AutonomyLevel.HUMAN_APPROVAL
    outcomes = []

    for i in range(1, args.count + 1):
        vm = view_model(rng)
        creator, name = random_creator(rng, vm, camp)
        header = (f"negotiation {i}/{args.count}  |  {name}  |  "
                  f"sim hidden floor ${creator.reservation_per_video:,.0f}/video")
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
        if not args.quiet:
            print(f"view model: {vm.notes}\n")

        o = run_negotiation(
            vm, camp.econ, camp.ctx, creator, brief=camp.brief, creator_name=name,
            autonomy=autonomy, max_rounds=args.rounds, thread_id=f"t{i}",
            verbose=not args.quiet)
        outcomes.append(o)

        if args.quiet:
            deal = f"${o.final_total:,.0f}" if o.final_total else "-"
            roi = f"{o.final_ev_roi:.1f}x" if o.final_ev_roi is not None else "-"
            print(f"  -> {o.status:10s} {o.rounds} rounds  final {deal:>8s}  ROI {roi}")
        else:
            print("\n--- result ---")
            print("\n".join(summary_lines(o)))

    print("\n" + "#" * 40)
    print(f"aggregate over {len(outcomes)} negotiations\n")
    print(batch_metrics(outcomes).report())


def run_spar(args, camp: Campaign) -> None:
    from oblsk_negotiator.spar import spar
    rng = np.random.default_rng(resolve_seed(args, 2))
    views = rng.lognormal(np.log(args.median), 0.6, 24)
    vm = fit_view_model(views)
    print(f"sparring vs Claude-played manager  |  profile: {args.spar}")
    print(f"ladder: {price_ladder(vm, camp.econ, camp.ctx.pricing_policy()).summary()}\n")
    try:
        result = spar(vm, camp.econ, camp.ctx, brief=camp.brief,
                      counterpart_profile=args.spar, creator_name=args.name,
                      max_rounds=args.rounds)
    except RuntimeError as e:
        print(f"cannot spar: {e}")
        return
    print(result.report())


def run_replay(args, camp: Campaign) -> None:
    with open(args.replay, encoding="utf-8") as f:
        thread = f.read()
    # The replay prices off the creator's typical reach; pass --median with
    # their real number so the agent's ladder matches the actual deal size.
    rng = np.random.default_rng(resolve_seed(args, 2))
    views = rng.lognormal(np.log(args.median), 0.6, 24)
    vm = fit_view_model(views)

    print(f"replaying {args.replay}")
    print(f"view model: {vm.notes}")
    print(f"ladder:     {price_ladder(vm, camp.econ, camp.ctx.pricing_policy()).summary()}\n")
    report = replay_thread(thread, vm, camp.econ, camp.ctx, brief=camp.brief,
                           us_aliases=camp.us_aliases,
                           creator_name=args.name, max_rounds=args.rounds)
    print(report.report())


def main() -> None:
    # Windows consoles often default to a legacy code page; the agent writes
    # real punctuation, so print as UTF-8 instead of mojibake.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Run Oblsk creator negotiations in the terminal.")
    p.add_argument("--campaign", metavar="YAML", default=None,
                   help="campaign config file (brief, economics, negotiation knobs); "
                        "see examples/unest_campaign.yaml")
    p.add_argument("--replay", metavar="FILE", default=None,
                   help="shadow the agent over a real chat/email thread (see examples/)")
    p.add_argument("--spar", metavar="PROFILE", default=None,
                   help="negotiate live against a Claude-played manager with this "
                        "private playbook, e.g. \"floor $2,600/video, opens at "
                        "$4,500, concedes slowly\" (needs ANTHROPIC_API_KEY)")
    p.add_argument("--median", type=float, default=50000,
                   help="creator's typical per-video views, for --replay pricing (default: 50000)")
    p.add_argument("--chat", action="store_true",
                   help="interactive: you type as the creator, the agent replies "
                        "live (the same pipeline the platform runs on Gmail)")
    p.add_argument("--random", action="store_true",
                   help="one fresh random creator, full step-by-step negotiation "
                        "(different every run; pass --seed to reproduce)")
    p.add_argument("--explain", action="store_true",
                   help="print the plain-English calculation behind the prices")
    p.add_argument("--count", type=int, default=1,
                   help="run this many simulated negotiations, each varied (default: 1)")
    p.add_argument("--name", default="Devin", help="creator display name")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed (fixed by default for reproducibility; "
                        "omit with --random for a new scenario each run)")
    p.add_argument("--rounds", type=int, default=3,
                   help="max concession rounds (default: 3)")
    p.add_argument("--floor", type=float, default=2450,
                   help="sim creator's private per-video floor (default: 2450)")
    p.add_argument("--opens", choices=["question", "interest", "ask"], default="interest",
                   help="how the sim creator opens the thread (default: interest)")
    p.add_argument("--ask", type=float, default=3500,
                   help="sim creator's opening per-video ask when --opens ask (default: 3500)")
    p.add_argument("--counter", type=float, default=0.6,
                   help="how hard the sim creator counters, 0..1 (default: 0.6)")
    p.add_argument("--bulk", type=float, default=0.06,
                   help="bulk discount the sim creator grants for volume (default: 0.06)")
    p.add_argument("--max-videos", type=int, default=4,
                   help="most videos the sim creator will commit to (default: 4)")
    p.add_argument("--autonomy", action="store_true",
                   help="run without the human-approval gate")
    p.add_argument("--reject", metavar="ACTION", default=None,
                   help="single run: human rejects one action once, e.g. opening_flat")
    p.add_argument("--quiet", action="store_true",
                   help="with --count, print one summary line per run instead of transcripts")
    args = p.parse_args()

    camp = load_campaign(args.campaign) if args.campaign else DEMO_CAMPAIGN
    if args.campaign:
        print(f"campaign: {camp.name}\n")
    if args.chat:
        run_chat(args, camp)
    elif args.spar:
        run_spar(args, camp)
    elif args.replay:
        run_replay(args, camp)
    elif args.random:
        run_random(args, camp)
    elif args.count > 1:
        run_many(args, camp)
    else:
        run_one(args, camp)


if __name__ == "__main__":
    main()
