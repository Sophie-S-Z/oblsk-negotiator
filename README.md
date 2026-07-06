# Oblsk Negotiation Agent

An agent that prices every creator deal and negotiates like a real person, with someone signing off on each reply until it earns its way to sending on its own.

There are two parts: the quantitative side, which does the math, and the communication side, which runs the conversation. When the negotiator considers a move, it bases its decision off what the calculator determines each option is worth — the LLM writes the words and reads the creator's messages, but it never invents a price or picks the move.

## The idea

A creator's next video will not get their average number of views. Most posts land near their usual mark, and once in a while one runs far past it. Those rare hits are incredibly valuable, but they're hidden by an average. So the calculator fits a heavy-tailed model of the creator's views (recency-weighted, with an organic-to-sponsored haircut) and draws thousands of outcomes from it, then reports the range: a soft downside, a typical case, an upside, the return on spend, and the odds the deal clears zero.

From that distribution it builds the **pricing ladder** every negotiation runs on:

- **anchor** — where to open: low but credible, leaves room to move up
- **target** — where to aim: the fee that hits the campaign's ROI goal at expected delivery
- **walk-away** — the ceiling: the p10 downside still clears the minimum ROI (or an optional CPM cap, whichever binds first)

The negotiator opens at the anchor, revises once toward the target when the creator pushes back, and judges every ask against the ladder — accept at or under it, escalate to a person when an ask is a multiple of the walk-away. Past the target, a lower price only hands value to the creator, whereas each added video is another independent shot at the tail — so the next move is a bundle: a fair per-video rate, a small bulk discount, and a total that grows with the count. When the creator has a rate card, the bundle is composed from their own list prices at the market-standard combo discount.

The conversation itself is a behavior tree: one pass per message, top to bottom, first branch that fits wins. A follow-up with no counter gets the standing offer restated, never an unforced concession; a second straight rejection with no counter gets a graceful close instead of another pitch; a message negotiating contract terms (exclusivity compensation, equity structure, kill fees) goes straight to a person — the agent never negotiates paper. The branch order lives in `tree_spec.yaml`, and every number the negotiator uses lives in the campaign config, so a campaign tunes its stance (ROI targets, anchor aggressiveness, bundle size, approval ceilings) in a YAML file, not code. Every outbound message waits for a human approval until autonomy is switched on.

## Using it from the Oblsk platform

The platform (rockcorp/oblsk) already polls Gmail threads, classifies messages, and runs the review UI — what it asks a negotiator for is the next reply. This agent serves exactly that as HTTP:

```bash
py -m oblsk_negotiator.service --campaign examples/unest_campaign.yaml --port 8788
```

`POST /suggest` takes the platform's `NegotiatorInput` shape (the thread's messages, `medianViewsPerVideo`, `recentPostViews`) and returns its `NegotiatorSuggestion` shape (drafted message, strategy, debug numbers) plus an `agent` block with the precise action, the ladder, `requiresApproval`, and any `humanPrompt`. The service is stateless — state is derived from the thread itself, so offers a *person* made in the thread count as the agent's positions and it picks up mid-flight without re-opening at the anchor. Wiring details, the Convex `fetch` snippet, and the response contract: [docs/INTEGRATION.md](docs/INTEGRATION.md).

## Humans stay in the loop

Beyond per-message approvals, the agent knows what it cannot do and asks:

- **`ask_human`** — a question the brief doesn't cover ("is UNest FDIC insured?") or a reference to a call it wasn't on pauses the thread with a specific request to the team, instead of bluffing. The team answers; the agent keeps the thread.
- **`propose_call`** — when the creator wants a call and there's no counter to price, the agent proposes the campaign's `call_windows` and flags a teammate to send the invite and run the call (the agent can't join calls). The handoff note asks whoever runs the call to write what was agreed back into the thread, so the agent isn't blind afterwards.
- **`escalate_human`** — extreme asks, exhausted rounds, and contract-term negotiation hand the whole thread to a person.

## Campaign config

One YAML file per campaign holds the brief the agent answers questions from, the economics that price every deal, the full negotiation stance, and how to recognize our side of an email thread. `examples/unest_campaign.yaml` is a complete example built from a real campaign's program agreement and content guide:

```bash
py demo.py --campaign examples/unest_campaign.yaml --count 10 --quiet
```

## Where the LLM fits

Set `ANTHROPIC_API_KEY` (optionally `OBLSK_LLM_MODEL`) and the agent uses Claude to:

- **draft every outbound message** from the decision + the thread so far (`prose.write_message`) — a draft that drops the calculator's numbers is discarded;
- **answer open-ended questions** from the campaign brief without ever quoting a rate (`qa.answer_question`);
- **read real creator messages** into intent + ask for replay and live threads (`replay.interpret_message`, structured output).

Every LLM call has a deterministic offline fallback (templates, keyword rules), so tests, CI, and the demo run with no network — set `OBLSK_NO_LLM=1` to force it.

## Testing against real threads

`replay.py` shadow-runs the agent over a chat or email thread your team handled manually. Paste the thread — a raw Gmail thread works as-is, sender headers, signatures, quoted replies, confidentiality footers and all — and at each creator message the report shows what the agent would have done (how it read the message, the move, the drafted reply, and why) next to what your team actually sent. Nothing is ever sent.

```bash
py demo.py --campaign examples/unest_campaign.yaml \
           --replay examples/unest_thread.txt --median 150000
```

`examples/unest_thread.txt` is a real agency-run negotiation; on it the agent opens within $100 of the team's real opener, accepts the same $2,000/video the team agreed, holds firm on the "just following up" nudges, proposes the intro call when the manager confirms availability (flagging a teammate to run it), and escalates the contract-revision email to a human. `--median` is the creator's typical per-video views, so the ladder prices the same deal your team was pricing.

### Sparring: Claude plays the other side

To probe conversation quality beyond history, have Claude role-play the creator's manager against the agent and judge the transcript (needs `ANTHROPIC_API_KEY`):

```bash
py demo.py --campaign examples/unest_campaign.yaml \
           --spar "the creator's floor is $2,600/video; open by asking $4,500; cite her reach; concede slowly; suggest a call once"
```

The manager's private playbook is whatever you type; the report shows every exchange with the agent's move and rationale, so a human can grade the back-and-forth.

## Layout

```
oblsk_negotiator/      the package
  ev_engine.py         heavy-tailed view model & Monte Carlo; every number starts here
  pricing.py           the ladder (anchor/target/walk-away), authenticity risk, CPM benchmarks
  rate_card.py         creator formats and their rates; market-standard bundle discounts
  bundles.py           the deals the agent can offer, incl. multi-format composition
  behavior_tree.py     the decision logic (one move per message) + CampaignContext config
  tree_spec.yaml       the branch order the tree walks, as config
  qa.py                answers questions from the campaign brief (LLM-first)
  prose.py             turns a decision into one human message (LLM-first)
  llm.py               the one wrapper around the Anthropic SDK, with offline fallback
  service.py           HTTP /suggest service the Oblsk platform calls (stateless)
  campaign.py          one YAML per campaign: brief, economics, stance, aliases
  replay.py            shadow-run the agent over real chat/email threads
  spar.py              Claude plays the creator's manager; agent negotiates back
  state.py             one DB row per thread: phase, rounds, approvals, autonomy
  events.py            the same thread as an append-only event log
  creator_sim.py       a parameterized simulated creator for closed-loop testing
  runner.py            the loop, the approval gate, the metrics
tests/                 72 checks, all offline
examples/              campaign config + real and sample threads for replay
docs/                  USAGE.md (how to run it) + CALCULATOR.md (the pricing model) + INTEGRATION.md
figures/               architecture and behavior-tree diagrams
```

The pricing ladder, recency weighting, sponsored haircut, authenticity heuristic, and CPM benchmark table are ported from the influencer-calculator pricing-engine upgrade (PR #1 on eyalcohen2524/influencer-calculator), adapted to run on the Monte Carlo instead of a bootstrap. The CPM table is a placeholder — calibrate against closed deals before trusting absolute dollars.

## Running it

New here? [docs/USAGE.md](docs/USAGE.md) is the full plain-language manual: install, every way to run it, every feature explained, start to finish.

```bash
pip install -r requirements.txt
pip install anthropic                    # optional: enables the LLM path
python -m pytest                         # 72 checks, offline
```

**Talk to it yourself** — you play the creator, the agent replies live. This is the exact read → price → draft the platform runs on real Gmail threads; here you type the messages:

```bash
py demo.py --chat                        # type a message, get a real reply
py demo.py --chat --campaign examples/unest_campaign.yaml --median 150000
```

**See the calculation**, in plain English (reach → dollars → the three prices):

```bash
py demo.py --explain                     # print the derivation, then a run
```

**Watch full negotiations** — a fresh random creator each run, or a sweep:

```bash
py demo.py --random                      # a new creator + full transcript every run
py demo.py --floor 2800 --opens ask      # a specific harder creator
py demo.py --count 20 --quiet            # 20 varied runs + aggregate metrics
py demo.py --campaign examples/unest_campaign.yaml --replay examples/unest_thread.txt
```

The notebook in `notebooks/` walks the same ground interactively. To run it in Google Colab against this private repo, see [docs/COLAB_SETUP.md](docs/COLAB_SETUP.md) (add a `GITHUB_TOKEN` secret — two minutes).
