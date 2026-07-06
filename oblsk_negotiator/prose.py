"""
The words the creator sees. Turns a Decision into one short, human message.

The tree decides the move and the numbers; this layer only renders them. It
leads with one offer, never a menu, and never changes a number.

write_message is the entry point: it has the LLM draft the message from the
decision (build_llm_prompt) and falls back to render_template — the
deterministic, no-network version — when no LLM is configured or the draft
drops a required number.
"""

from __future__ import annotations

import random
from . import llm
from .behavior_tree import Decision, Action
from .ev_engine import Deal, Bundle
from .rate_card import FORMAT_LABELS


_GREETINGS = ["Hey {name},", "Hi {name},", "{name},", "Thanks for the note, {name}.",
              "Good to hear from you, {name}.", "Appreciate you getting back to me.",
              "Thanks for the reply.", "Great to connect, {name}."]
_SOFTENERS = ["Really liked your recent stuff.", "Been enjoying your content lately.",
              "Your last few posts have been great."]
_CLOSERS = ["Let me know what you think.", "Happy to adjust, what feels right on your end?",
            "Let me know if that works for you."]


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _bundle_phrase(b: Bundle) -> str:
    """Describe a multi-format bundle by its formats, not a video count: e.g.
    '$48,600 total for an Instagram Reel, an Instagram Story, and a TikTok video'."""
    parts = []
    for line in b.lines:
        label = FORMAT_LABELS.get(line.fmt, line.fmt)
        if line.quantity > 1:
            parts.append(f"{line.quantity}x {label}")
        else:
            parts.append(f"{_article(label)} {label}")
    if len(parts) == 1:
        items = parts[0]
    else:
        items = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return f"{_money(b.total_usd)} total for {items}"


def _deal_phrase(d) -> str:
    if isinstance(d, Bundle):
        return _bundle_phrase(d)
    if d.video_count == 1:
        return f"{_money(d.flat_per_video)} for the video"
    per = d.flat_per_video
    approx = "" if abs(per * d.video_count - d.total_usd) < 0.5 else "about "
    return (f"{_money(d.total_usd)} total for {d.video_count} videos "
            f"({approx}{_money(per)} each)")


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
        if isinstance(decision.deal, Bundle):
            pitch = ("Packaging a few formats together gets you in front of the "
                     "same audience in more than one place, and it is a stronger "
                     "package for you")
        else:
            pitch = ("A few videos tends to perform a lot better than a one-off, "
                     "and it is a stronger package for you")
        return (f"{g}\n\nTotally hear you on the rate. Here is something that might "
                f"work better for both of us: {_deal_phrase(decision.deal)}. {pitch}."
                f"\n\n{close}")

    if a == Action.ASK_HUMAN:
        # Internal: nothing goes to the creator until the team replies.
        return f"[agent asks the team: {decision.human_prompt}]"

    if a == Action.PROPOSE_CALL:
        deal_line = (f" In the meantime, where we stand: "
                     f"{_deal_phrase(decision.deal)}." if decision.deal else "")
        return (f"{g}\n\nWould love to get a call on the calendar so we can dig "
                f"into the campaign properly. We could do "
                f"{decision.answer_text}; let me know what works and we'll "
                f"send the invite.{deal_line}\n\n{close}")

    if a == Action.HOLD_FIRM:
        return (f"{g}\n\nThanks for following up — this is still very much on "
                f"our radar. Where we stand: {_deal_phrase(decision.deal)}. If "
                f"that works on your end, we can get things moving right away."
                f"\n\n{close}")

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


def write_message(decision: Decision, creator_name: str = "there",
                  thread_history: str = "", seed: int | None = None) -> str:
    """The outbound message for a decision: LLM-drafted when available, template
    otherwise. A draft that does not state the deal's total is discarded — the
    numbers come from the calculator, and a message that loses them is worse
    than a template that keeps them."""
    if decision.action in (Action.ASK_HUMAN, Action.PAUSE_HUMAN):
        # Internal moves: nothing is drafted for the creator.
        return render_template(decision, creator_name=creator_name, seed=seed)
    prompt = build_llm_prompt(decision, creator_name, thread_history)
    draft = llm.complete(prompt["system"], prompt["user"], max_tokens=400)
    if draft and _numbers_intact(decision, draft):
        return draft.strip()
    return render_template(decision, creator_name=creator_name, seed=seed)


def _numbers_intact(decision: Decision, draft: str) -> bool:
    """True when the draft states the deal total (and per-video rate for a
    multi-video flat deal). Guards against a reworded offer losing its price."""
    d = decision.deal
    if d is None or decision.action == Action.PROPOSE_CALL:
        return True   # a call proposal need not restate the standing offer
    required = [d.total_usd]
    if not isinstance(d, Bundle) and d.video_count > 1:
        required.append(d.flat_per_video)
    return all(_money(x) in draft for x in required)


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
    if decision.action == Action.HOLD_FIRM:
        parts.append("They followed up without a counter. Warmly reaffirm the "
                     "standing offer; do not offer more money or new terms.")
    if decision.action == Action.PROPOSE_CALL:
        parts.append(f"Propose an intro call at: {decision.answer_text}. A "
                     f"teammate will send the invite and run the call. Confirm "
                     f"any availability they already gave; do not discuss new "
                     f"prices.")
    if thread_history:
        parts.append(f"Prior thread (for tone):\n{thread_history}")
    parts.append("Write only the message body. No subject line.")
    return {"system": system, "user": "\n".join(parts)}
