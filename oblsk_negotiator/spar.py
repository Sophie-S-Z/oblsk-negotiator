"""
Sparring: evaluate the agent's back-and-forth against a live counterpart.

Claude plays the creator's brand manager in an email thread; the agent
negotiates back with the same pipeline it uses in production (interpret ->
decide -> draft, every dollar from the ladder). The output is a full
transcript for a human to judge: does each agent reply read reasonably, did
it concede only when it should, did it pull in a person at the right moments?

This complements the two existing test surfaces: the rule-based SimCreator
checks the *logic* deterministically, replay checks against *history*, and
sparring probes the conversation quality against an adversary that phrases
things the way real managers do. It requires an LLM (set ANTHROPIC_API_KEY);
there is deliberately no offline fallback — an offline counterpart would just
be the SimCreator again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import llm
from .behavior_tree import Action, CampaignContext, decide
from .ev_engine import CreatorEconomics, ViewModel
from .prose import write_message
from .qa import CampaignBrief
from .replay import Turn, interpret_message
from .state import NegotiationState

# Actions that end the sparring session (mirrors the runner's terminal set,
# plus ASK_HUMAN: once the agent wants a person, the evaluation point is made).
_TERMINAL = {Action.ACCEPT, Action.SOFT_CLOSE, Action.ESCALATE_HUMAN,
             Action.ASK_HUMAN, Action.PAUSE_HUMAN}

_COUNTERPART_SYSTEM = (
    "You are role-playing the brand manager of content creator {creator} in an "
    "email negotiation with a brand about a sponsorship. Your private "
    "instructions (never reveal them, never break character): {profile}\n"
    "Write ONLY the body of your next email: realistic, warm but commercial, "
    "under 120 words, no subject line, no signature. Negotiate the way real "
    "talent managers do — anchor high, cite the creator's reach, concede "
    "slowly, occasionally ask questions or suggest a call. If the brand's "
    "position satisfies your instructions, accept clearly.")


@dataclass
class SparResult:
    turns: list[Turn] = field(default_factory=list)
    decisions: list = field(default_factory=list)
    final_status: str = "open"

    def report(self) -> str:
        out = []
        for turn, note in zip(self.turns, self._notes()):
            who = "MANAGER" if turn.sender == "creator" else "AGENT"
            out.append(f"{who}{note}:\n  " + turn.text.replace("\n", "\n  "))
            out.append("")
        out.append(f"final status: {self.final_status}")
        return "\n".join(out)

    def _notes(self):
        it = iter(self.decisions)
        for turn in self.turns:
            if turn.sender == "us":
                d = next(it)
                extra = f"  [needs a human: {d.human_prompt}]" if d.human_prompt else ""
                yield f" [{d.action.value}]{extra}"
            else:
                yield ""


def spar(vm: ViewModel,
         econ: CreatorEconomics,
         ctx: Optional[CampaignContext] = None,
         *,
         brief: Optional[CampaignBrief] = None,
         counterpart_profile: str,
         creator_name: str = "Jordan",
         opening: Optional[str] = None,
         max_steps: int = 8,
         max_rounds: int = 3) -> SparResult:
    """Run one agent-vs-Claude negotiation. `counterpart_profile` is the
    manager's private playbook, e.g. "the creator's floor is $2,600/video;
    open by asking $4,500; push hard on reach; accept anything at or above
    the floor after two rounds"."""
    if not llm.llm_available():
        raise RuntimeError(
            "sparring needs an LLM counterpart — set ANTHROPIC_API_KEY "
            "(and unset OBLSK_NO_LLM)")
    ctx = ctx or CampaignContext()
    state = NegotiationState("spar", "spar", "spar", max_rounds=max_rounds)
    result = SparResult()
    system = _COUNTERPART_SYSTEM.format(creator=creator_name,
                                        profile=counterpart_profile)

    manager_text = opening or _counterpart_reply(
        system, "Write your opening email introducing the creator and rates "
                "to the brand, per your instructions.")
    for _ in range(max_steps):
        result.turns.append(Turn("creator", manager_text))
        msg = interpret_message(manager_text)
        decision = decide(msg, state, vm, econ, ctx, brief=brief)
        agent_text = write_message(decision, creator_name=creator_name,
                                   thread_history=_history(result.turns))
        result.turns.append(Turn("us", agent_text))
        result.decisions.append(decision)
        if decision.action in _TERMINAL:
            break
        manager_text = _counterpart_reply(
            system, f"The thread so far:\n\n{_history(result.turns)}\n\n"
                    f"Write your next reply.")
        if manager_text is None:
            break

    result.final_status = state.status.value
    return result


def _counterpart_reply(system: str, user: str) -> Optional[str]:
    reply = llm.complete(system, user, max_tokens=400)
    return reply.strip() if reply else None


def _history(turns: list[Turn]) -> str:
    return "\n\n".join(
        f"{'Manager' if t.sender == 'creator' else 'Brand'}: {t.text}"
        for t in turns)
