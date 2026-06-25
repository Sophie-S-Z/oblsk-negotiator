# Oblsk Negotiation Agent

An agent that prices every creator deal and negotiates like a real person, with someone signing off on each reply until it earns its way to sending on its own.

There are two parts to this agent: the quantitative side, which does the math, and the communication side, which runs the conversation. When the negotiator considers a move, it bases its decision off of what the calculator determines each option is worth.

## The idea

A creator's next video will not get their average number of views. Most posts land near their usual mark, and once in a while one runs far past it. Those rare hits are incredibly valuable, but they're hidden by an average. So the calculator fits a heavy-tailed model of the creator's views and draws thousands of outcomes from it, then reports the range: a soft downside, a typical case, an upside, the return on spend, and the odds the deal clears zero.

That model is why the agent's fallback move is a bundle rather than a lower price. Each added video is another independent shot at the tail, so the agent keeps the per-video rate fair — a small bulk discount for committing to several at once — and lets the total grow with the count. Oblsk's absolute net value multiplies while ROI stays high, and the creator is still paid fairly per video, so it's a deal they'll actually take.

The negotiator is a behavior tree. One pass per message, top to bottom, first branch that fits wins. It answers questions first, opens with one clean flat number, reshapes to a small bundle if the creator pushes back, closes a thin gap, and hands off to a person when the ask is unreasonable or the rounds run out. As of right now, every outbound message waits for a human approval, but autonomy can be enabled with just one setting.

## Layout

```
oblsk_negotiator/      the package
  ev_engine.py         the calculator: heavy-tailed view model & Monte Carlo
  bundles.py           the deals the agent can offer
  behavior_tree.py     the decision logic (one move per message)
  qa.py                answers questions from the campaign brief
  prose.py             turns a decision into one human message
  state.py             one DB row per thread: phase, rounds, approvals, autonomy
  creator_sim.py       stand-in creators for end-to-end testing
  runner.py            the loop, the approval gate, the metrics
tests/                 21 checks
notebooks/             a Colab walkthrough
figures/               the architecture and behavior-tree diagrams
```

## File breakdown

| File | Role | What it does |
|------|------|--------------|
| `ev_engine.py` | Calculator | Fits the heavy-tailed view model and runs the Monte Carlo. Every number starts here. |
| `bundles.py` | Calculator | Builds the flat offer, the bundle, and the non-price sweeteners. |
| `behavior_tree.py` | Negotiator | Picks one move per message and reads the calculator to do it. No pricing math itself. |
| `qa.py` | Negotiator | Answers a creator's questions from the brief before any price is on the table. |
| `prose.py` | Negotiator | Turns a decision into one short, human-sounding message. One offer, never a menu. |
| `state.py` | Memory | One database row per thread. Saves and reloads cleanly so a thread can pause and resume. |
| `creator_sim.py` | Testing | Stand-in creators with different temperaments. |
| `runner.py` | Glue | Runs the loop, enforces the approval gate, tallies win rate, rounds, discount, and ROI. |

The demo writes its messages from templates so it runs anywhere with no network.
In production, the same decisions and numbers go to a language model that writes the wording. That model never invents a price or decides the move.

## Running it

```bash
pip install -r requirements.txt
python -m pytest                 # 21 checks
```

The notebook in `notebooks/` runs the whole thing in Colab: the pricing thesis, a
full negotiation with approvals, a creator the agent hands off, and a sweep
across creator types.
