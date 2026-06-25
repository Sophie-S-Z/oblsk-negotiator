"""
The words the creator sees. Turns a Decision into one short, human message.

The tree decides the move and the numbers; this layer only renders them. It
leads with one offer, never a menu, and never changes a number. render_template
is the no-network version used in the demo and as a fallback. build_llm_prompt is
the prompt for the production model, which writes the same decision in the same
voice.
"""

from __future__ import annotations

import random
from .behavior_tree import Decision, Action
from .ev_engine import Deal


_GREETINGS = ["Hey {name},", "Hi {name},", "{name},", "Hey {name}!"]
_SOFTENERS = ["Really liked your recent stuff.", "Been enjoying your content lately.",
              "Your last few posts have been great."]
_CLOSERS = ["Let me know what you think.", "Happy to adjust, what feels right on your end?",
            "Let me know if that works for you."]


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _deal_phrase(d: Deal) -> str:
    if d.video_count == 1:
        return f"{_money(d.flat_per_video)} for the video"
    return (f"{_money(d.total_usd)} total for {d.video_count} videos "
            f"({_money(d.flat_per_video)} each)")


def render_template(decision: Decision, creator_name: str = "there",
                    seed: int | None = None) -> str:
    rng = random.Random(seed)
    g = rng.choice(_GREETINGS).format(name=creator_name)
    soft = rng.choice(_SOFTENERS)
    close = rng.choice(_CLOSERS)
    a = decision.action

    if a == Action.ANSWER_QUESTION:
        return f"{g}\n\n{decision.answer_text}\n\nAnything else you want to know?"

    if a == Action.OPENING_FLAT:
        return (f"{g}\n\n{soft} We'd love to have you on this campaign. We were "
                f"thinking {_deal_phrase(decision.deal)}.\n\n{close}")

    if a == Action.ESCALATE_BUNDLE:
        return (f"{g}\n\nTotally hear you on the rate. Here is something that might "
                f"work better for both of us: {_deal_phrase(decision.deal)}. A few "
                f"videos tends to perform a lot better than a one-off, and it is a "
                f"stronger package for you.\n\n{close}")

    if a == Action.ADD_SWEETENER:
        from .bundles import LEVER_PITCH
        pitch = LEVER_PITCH.get(decision.non_price_lever or "", "")
        extra = f" On top of that, {pitch}." if pitch else ""
        return (f"{g}\n\nI want to make this easy to say yes to. We can keep it at "
                f"{_deal_phrase(decision.deal)}.{extra}\n\n{close}")

    if a == Action.ACCEPT:
        deal = _deal_phrase(decision.deal) if decision.deal else "the terms we discussed"
        return (f"{g}\n\nPerfect, {deal} works great on our end. I'll get the "
                f"agreement over to you and we'll get rolling. Excited to work "
                f"together.")

    if a == Action.SOFT_CLOSE:
        return (f"{g}\n\nNo problem at all, and no pressure. If the timing is ever "
                f"better down the line, I'd love to revisit. Wishing you the best.")

    if a == Action.ESCALATE_HUMAN:
        return (f"{g}\n\nThanks for working through this with me. Let me check a "
                f"couple of things on our side and come right back to you.")

    if a == Action.PAUSE_HUMAN:
        return "[agent paused: a teammate stepped into the thread]"

    return f"{g}\n\n{close}"


def build_llm_prompt(decision: Decision, creator_name: str,
                     thread_history: str = "") -> dict:
    d = decision.deal
    if decision.action == Action.ANSWER_QUESTION:
        deal_brief = "no offer yet, answer the question"
    elif d is not None:
        deal_brief = _deal_phrase(d)
    else:
        deal_brief = "no offer, soft close"

    system = (
        "You are an experienced influencer-marketing account manager writing a "
        "short message to a creator. Warm, casual, a little imperfect, like a real "
        "person. Never reveal or imply you are an AI. Never sound templated. Soften "
        "the offer. Use exactly the deal terms provided; do not invent numbers. "
        "Lead with one offer; never present a menu. Under 110 words.")
    parts = [f"Creator name: {creator_name}",
             f"Move to communicate: {decision.action.value}",
             f"Deal terms to state exactly: {deal_brief}"]
    if decision.action == Action.ANSWER_QUESTION and decision.answer_text:
        parts.append(f"Answer to convey: {decision.answer_text}")
    if decision.non_price_lever:
        parts.append(f"Sweetener to mention: {decision.non_price_lever}")
    if decision.action == Action.ESCALATE_BUNDLE:
        parts.append("Frame the bundle as a bigger opportunity, not a fallback.")
    if thread_history:
        parts.append(f"Prior thread (for tone):\n{thread_history}")
    parts.append("Write only the message body. No subject line.")
    return {"system": system, "user": "\n".join(parts)}
