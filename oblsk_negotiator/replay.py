"""
Replay: test the agent against a real chat or email thread.

Paste a thread where the deal flow was executed manually. The replay steps
through it creator-message by creator-message, and at each point asks the agent
what it *would* have done — interpreting the raw text into intent and ask,
running the same behavior tree the live agent runs, and drafting the message it
would have sent. The report puts the agent's move next to what your team
actually replied, so you can judge the agent against deals you already closed
(or lost) before letting it near a live thread. Nothing is ever sent.

Two transcript formats are understood, auto-detected by parse_thread:

Prefix format — one prefix per turn, continuation lines attach above. Creator
lines start with `Creator:` / `Them:` / their name via `creator_aliases`; our
side is `Us:` / `Me:` / `Agent:`:

    Creator: Thanks for reaching out! What would the deliverables be?
    Us: One reel plus a story frame, two weeks out. Rate-wise we were at $1,800.
    Creator: I'm usually around $3k for a reel, could you do $2,500?

Email format — a Gmail-style thread pasted as-is: `Name <email>` header lines,
date lines, `to me` lines, signatures, confidentiality footers and all. Pass
`us_aliases` (names, emails, or domains — e.g. ["unest.co", "Eyal Cohen"]) so
the parser knows which messages are ours; messages addressed "to me" are
treated as the other side's when no alias matches.

Message interpretation is LLM-first (structured output) with a keyword/regex
fallback, so replays run offline too — just less tolerant of unusual phrasing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from . import llm
from .behavior_tree import (CampaignContext, CreatorMessage, Decision, Intent,
                            decide)
from .ev_engine import CreatorEconomics, FormatCatalog, ViewModel
from .prose import write_message
from .qa import CampaignBrief
from .rate_card import RateCard
from .state import NegotiationState


# ---- transcript parsing -------------------------------------------------------

_CREATOR_PREFIXES = ("creator", "them", "influencer", "talent")
_US_PREFIXES = ("us", "me", "agent", "oblsk", "brand", "we")


@dataclass(frozen=True)
class Turn:
    sender: str   # "creator" | "us"
    text: str


def parse_transcript(text: str, creator_aliases: Optional[list[str]] = None,
                     us_aliases: Optional[list[str]] = None) -> list[Turn]:
    """Split a pasted thread into turns. A line like `Name: message` starts a
    new turn when Name matches a known prefix or alias (case-insensitive);
    anything else continues the current turn."""
    creator_names = {p.lower() for p in _CREATOR_PREFIXES}
    creator_names |= {a.lower() for a in (creator_aliases or [])}
    us_names = {p.lower() for p in _US_PREFIXES}
    us_names |= {a.lower() for a in (us_aliases or [])}

    turns: list[Turn] = []
    current_sender: Optional[str] = None
    current_lines: list[str] = []

    def flush():
        nonlocal current_lines
        if current_sender and any(l.strip() for l in current_lines):
            turns.append(Turn(current_sender, "\n".join(current_lines).strip()))
        current_lines = []

    for raw in text.splitlines():
        m = re.match(r"^\s*([A-Za-z][\w .'-]{0,40}?)\s*:\s*(.*)$", raw)
        name = m.group(1).strip().lower() if m else None
        if name in creator_names or name in us_names:
            flush()
            current_sender = "creator" if name in creator_names else "us"
            current_lines = [m.group(2)]
        elif current_sender is not None:
            current_lines.append(raw)
    flush()
    if not turns:
        raise ValueError(
            "no turns found — lines must start with 'Creator:'/'Them:' or "
            "'Us:'/'Me:' (or pass creator_aliases/us_aliases for other names)")
    return turns


# ---- email-format parsing -------------------------------------------------

# A Gmail-style date line: "Jun 18, 2026, 12:30 PM" optionally with a relative
# suffix like "(11 days ago)".
_DATE_LINE_RE = re.compile(
    r"^\s*[A-Z][a-z]{2,8} \d{1,2}, \d{4},? .*?\d{1,2}:\d{2}\s*(AM|PM|am|pm)")
_HEADER_RE = re.compile(r"^\s*(.{1,60}?)\s*(?:<([^>]+)>)?\s*$")
_TO_LINE_RE = re.compile(r"^\s*to\s+\S+", re.I)
# Boilerplate that ends the useful part of an email body.
_BODY_STOP_RE = re.compile(r"^\s*(CONFIDENTIALITY NOTICE|\d+\s*Attachments?$"
                           r"|.*Scanned by Gmail)", re.I)


def _matches_alias(name: str, email: str, aliases: set[str]) -> bool:
    name_l, email_l = name.lower(), (email or "").lower()
    domain = email_l.split("@")[-1] if "@" in email_l else ""
    return any(a == name_l or a == email_l or (domain and a == domain)
               for a in aliases)


def parse_email_thread(text: str,
                       us_aliases: Optional[list[str]] = None) -> list[Turn]:
    """Parse a pasted Gmail-style thread into turns. A message starts at a
    `Name <email>` (or bare name) line followed within two lines by a date
    line; a `to ...` line after the date is consumed. Bodies are cut at
    confidentiality footers and attachment markers. Senders matching
    `us_aliases` (name, email, or domain) are ours; otherwise a message
    addressed "to me" is the other side's (the mailbox owner is us)."""
    aliases = {a.lower() for a in (us_aliases or [])}
    lines = text.splitlines()
    messages: list[tuple[str, Optional[str], Optional[str], list[str]]] = []
    current: Optional[list] = None   # [name, email, to_line, body_lines]
    skip_body = False

    def date_at(j: int, fillers_allowed: int) -> int:
        """Index of the date line directly after a header: the very next
        non-empty line, or one line later when the header carries an email
        (Gmail inserts an 'Attachments' line there). Anything looser lets a
        signature line ('Founder, Scale+') steal the header slot."""
        seen = 0
        while j < len(lines) and seen <= fillers_allowed:
            if lines[j].strip():
                if _DATE_LINE_RE.match(lines[j]):
                    return j
                seen += 1
            j += 1
        return -1

    i = 0
    while i < len(lines):
        line = lines[i]
        header = _HEADER_RE.match(line) if line.strip() else None
        d = -1
        if header and (header.group(2) or len(line.strip()) <= 40):
            d = date_at(i + 1, 1 if header.group(2) else 0)
        if header and d != -1:
            if current:
                messages.append(tuple(current))
            name = header.group(1).strip().rstrip(":")
            current = [name, header.group(2), None, []]
            skip_body = False
            i = d + 1
            if i < len(lines) and _TO_LINE_RE.match(lines[i]):
                current[2] = lines[i].strip()
                i += 1
            continue
        if current is not None and not skip_body:
            if _BODY_STOP_RE.match(line):
                skip_body = True   # boilerplate: skip to the next message
            else:
                current[3].append(line)
        i += 1
    if current:
        messages.append(tuple(current))
    if not messages:
        raise ValueError("no email messages found in thread")

    turns = []
    for name, email, to_line, body in messages:
        text_body = "\n".join(body).strip()
        if not text_body:
            continue
        if _matches_alias(name, email, aliases):
            sender = "us"
        elif to_line and to_line.lower().startswith("to me"):
            sender = "creator"    # addressed to the mailbox owner: not ours
        else:
            sender = "creator" if aliases else "us"
        turns.append(Turn(sender, text_body))
    return turns


def parse_thread(text: str, *, us_aliases: Optional[list[str]] = None,
                 creator_aliases: Optional[list[str]] = None) -> list[Turn]:
    """Auto-detect the transcript format: an email thread when Gmail-style
    date lines are present, the Creator:/Us: prefix format otherwise."""
    if any(_DATE_LINE_RE.match(l) for l in text.splitlines()):
        return parse_email_thread(text, us_aliases=us_aliases)
    return parse_transcript(text, creator_aliases=creator_aliases,
                            us_aliases=us_aliases)


# ---- message interpretation ---------------------------------------------------

# Strong accepts close the deal even inside a longer message; weak ones only
# count when the message asks no question ("that works with my schedule — what
# about budget?" is a question, not a yes).
_STRONG_ACCEPT = ["let's do it", "lets do it", "deal!", "it's a deal",
                  "sign me up", "send the contract", "send over the contract",
                  "send the agreement"]
_IM_IN_RE = re.compile(r"\bi'?m in\b[.!\s]")   # "I'm in" but not "I'm interested"
_ACCEPT_PHRASES = ["that works", "works for me", "sounds good", "agreed",
                   "happy with that", "we're good", "perfect, let"]
_REJECT_PHRASES = ["not interested", "i'll pass", "ill pass", "pass on this",
                   "no thanks", "not a fit", "not for me", "going to pass",
                   "won't work for me", "wont work for me", "too low for me, i'll"]

# Contract/legal-terms negotiation: the agent never negotiates paper.
_CONTRACT_RE = re.compile(
    r"\bsection\s+\d|\bclause\b|\bprovision\b|\bkill fee\b|\bindemnif"
    r"|\bexclusivity (provision|clause|terms?)\b|\brevisions? to the (contract|agreement|terms)"
    r"|\b(contract|agreement) (revisions?|changes?|redlines?)\b|\bredline", re.I)

# They propose, request, or confirm availability for a call.
_WANTS_CALL_RE = re.compile(
    r"\b(hop|jump|get) on a (quick |short |intro )?call\b|\bintro call\b"
    r"|\bschedule a call\b|\bquick call\b|\bcalendly\b|\bcal\.com\b"
    r"|\bgrab a time\b|\bavailable (on|for) (a call|mon|tues?|wed|thur?s?|fri)"
    r"|\bi'?ll be available\b|\bmeet(ing)? (with )?(you and )?the team\b", re.I)

# They reference a call/conversation the agent has no record of.
_REFERS_CALL_RE = re.compile(
    r"\b(as|per) (we )?discussed\b|\bon (our|the) call\b|\bduring (our|the) call\b"
    r"|\bwhen we spoke\b|\bper our (conversation|chat)\b|\bfrom (our|the) call\b"
    r"|\bgreat (connecting|speaking|talking) with you\b", re.I)

_MONEY_RE = re.compile(r"\$\s*([\d][\d,]*(?:\.\d{1,2})?)\s*([kK])?"
                       r"|(?<![\w$.])([\d]+(?:\.\d)?)\s*[kK]\b")
# A dollar amount explicitly marked as the package total ("$7,000 total",
# "$7,500 for the 3 videos", "$8,000 for the package"). Our own bundle messages
# carry both a total and a per-video rate, so the total marker must win.
_TOTAL_MONEY_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d{1,2})?)\s*([kK])?\s*"
    r"(?:total\b|all[- ]?in\b|for\s+(?:the\s+|all\s+)?"
    r"(?:\d{1,2}\s+|one\s+|two\s+|three\s+|four\s+|five\s+|six\s+)?"
    r"(?:videos|reels|posts|tiktoks|deliverables|package|bundle|deal|"
    r"set|series|everything|both)\b)", re.I)
# A dollar amount explicitly marked as a per-video rate ("$2,500 a video",
# "$2,333 each", "$1,450 for the video").
_PV_MONEY_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d{1,2})?)\s*([kK])?\s*"
    r"(?:each\b|(?:a|per)\s+(?:video|reel|post|piece|tiktok)\b"
    r"|/\s*(?:video|reel|post)\b"
    r"|for\s+(?:the|a|one)\s+(?:cross-?posted\s+)?(?:video|reel|post|tiktok)\b)",
    re.I)
_VIDEOS_RE = re.compile(r"(?:for|across|of|all)?\s*(\d{1,2}|one|two|three|four|"
                        r"five|six)\s*(?:cross-?posted\s+)?"
                        r"(?:videos?|reels?|posts?|tiktoks?|deliverables?)", re.I)
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}

_INTERPRET_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string",
                   "enum": ["question", "interested", "negotiating",
                            "accepted", "rejecting", "unclear"]},
        "ask_total_usd": {"type": ["number", "null"]},
        "ask_video_count": {"type": ["integer", "null"]},
        "contract_terms": {"type": "boolean"},
        "wants_call": {"type": "boolean"},
        "refers_to_call": {"type": "boolean"},
    },
    "required": ["intent", "ask_total_usd", "ask_video_count", "contract_terms",
                 "wants_call", "refers_to_call"],
    "additionalProperties": False,
}

_INTERPRET_SYSTEM = (
    "You read one message from a content creator in a brand-sponsorship "
    "negotiation and classify it for a deal agent. intent: 'question' (asking "
    "about deliverables/timeline/usage/etc.), 'interested' (engaged, no number "
    "named), 'negotiating' (names or implies a price they want), 'accepted' "
    "(agrees to the deal on the table), 'rejecting' (declines outright), or "
    "'unclear'. ask_total_usd: the TOTAL dollar amount they are asking for, if "
    "any ('$3k' is 3000; a per-video rate times a stated video count is the "
    "total). ask_video_count: how many videos/posts that ask covers, if stated. "
    "Use null when not present. contract_terms: true when the message negotiates "
    "contract or legal terms (exclusivity provisions, equity structure, kill "
    "fees, usage-rights clauses, indemnities, requested contract revisions) "
    "rather than only a price. wants_call: true when they propose, request, or "
    "confirm availability for a call or meeting. refers_to_call: true when they "
    "reference a past call or conversation (e.g. 'as discussed on our call'). "
    "Judge only the creator's position, never ours.")


def _amount(m) -> float:
    v = float(m.group(1).replace(",", ""))
    return v * 1000.0 if m.group(2) else v


def _ask_amount(text: str,
                videos: Optional[int]) -> tuple[Optional[float], Optional[int]]:
    """The message's ask as (total_usd, video_count or None when unstated).

    An amount marked as a total ("$7,000 total", "$7,500 for the 3 videos",
    "$8,000 for the package") wins — a bundle message states both a total and a
    per-video rate, and the total is the deal. Otherwise the last amount is the
    ask (counters usually end with it: 'you offered $1,800 but could you do
    $2,500?'); when it carries a per-video marker ("$2,500 a video") it is
    scaled by the stated video count."""
    total_m = None
    for m in _TOTAL_MONEY_RE.finditer(text):
        total_m = m
    if total_m:
        return _amount(total_m), videos
    last = None
    for m in _MONEY_RE.finditer(text):
        last = m
    if last is None:
        return None, videos
    value = _amount(last) if last.group(1) is not None else float(last.group(3)) * 1000
    pv = None
    for m in _PV_MONEY_RE.finditer(text):
        pv = m
    if pv is not None and pv.start() == last.start():   # the ask is a rate
        return value * (videos or 1), videos or 1
    return value, videos


def heuristic_interpret(text: str) -> CreatorMessage:
    """Keyword/regex fallback for reading a creator message. Deliberately
    simple: the LLM path handles nuance; this keeps replays running offline."""
    lower = text.lower()
    contract = bool(_CONTRACT_RE.search(text))
    flags = {"wants_call": bool(_WANTS_CALL_RE.search(text)),
             "refers_to_call": bool(_REFERS_CALL_RE.search(text))}
    videos_m = _VIDEOS_RE.search(text)
    videos = None
    if videos_m:
        raw = videos_m.group(1).lower()
        videos = _WORD_NUMBERS.get(raw) or int(raw)
    total, videos = _ask_amount(text, videos)

    # Order matters: an explicit closer wins; then a named number means they
    # are negotiating (a message quoting rates is not a yes, even if it says
    # "a solution that works for both sides"); soft agreement phrases only
    # count when nothing is being asked or quoted.
    if any(p in lower for p in _STRONG_ACCEPT) or _IM_IN_RE.search(lower + " "):
        return CreatorMessage(Intent.ACCEPTED, raw_text=text, **flags)
    if any(p in lower for p in _REJECT_PHRASES):
        return CreatorMessage(Intent.REJECTING, raw_text=text, **flags)
    if total is not None:
        return CreatorMessage(Intent.NEGOTIATING, ask_total_usd=total,
                              ask_video_count=videos, raw_text=text,
                              contract_terms=contract, **flags)
    if contract:
        return CreatorMessage(Intent.NEGOTIATING, raw_text=text,
                              contract_terms=True, **flags)
    if any(p in lower for p in _ACCEPT_PHRASES) and "?" not in text:
        return CreatorMessage(Intent.ACCEPTED, raw_text=text, **flags)
    if "?" in text:
        return CreatorMessage(Intent.QUESTION, raw_text=text, **flags)
    if any(w in lower for w in ("interested", "keen", "love to", "tell me more",
                                "what did you have in mind")):
        return CreatorMessage(Intent.INTERESTED, raw_text=text, **flags)
    return CreatorMessage(Intent.UNCLEAR, raw_text=text, **flags)


def interpret_message(text: str) -> CreatorMessage:
    """Read a raw creator message into intent + ask: LLM with structured output
    when available, keyword fallback otherwise."""
    result = llm.complete_json(_INTERPRET_SYSTEM,
                               f"Creator message:\n{text}", _INTERPRET_SCHEMA,
                               max_tokens=300)
    if result:
        try:
            ask = result.get("ask_total_usd")
            videos = result.get("ask_video_count")
            return CreatorMessage(
                Intent(result["intent"]),
                ask_total_usd=float(ask) if ask is not None else None,
                ask_video_count=int(videos) if videos else None,
                raw_text=text,
                contract_terms=bool(result.get("contract_terms")),
                wants_call=bool(result.get("wants_call")),
                refers_to_call=bool(result.get("refers_to_call")))
        except (ValueError, KeyError):
            pass
    return heuristic_interpret(text)


# ---- the replay ----------------------------------------------------------------

@dataclass
class ReplayStep:
    creator_text: str
    interpreted: CreatorMessage
    decision: Decision
    agent_message: str
    human_reply: Optional[str]     # what your team actually sent next, if anything


@dataclass
class ReplayReport:
    steps: list[ReplayStep] = field(default_factory=list)
    final_status: str = "open"

    def report(self) -> str:
        out = []
        for i, s in enumerate(self.steps, 1):
            ask = (f"  ask ${s.interpreted.ask_total_usd:,.0f}"
                   f"/{s.interpreted.ask_video_count or 1} video(s)"
                   if s.interpreted.ask_total_usd else "")
            deal = (f"  ->  ${s.decision.deal.total_usd:,.0f}"
                    if s.decision.deal else "")
            out.append(f"--- step {i} " + "-" * 50)
            out.append(f"CREATOR: {s.creator_text}")
            out.append(f"  read as: {s.interpreted.intent.value}{ask}")
            out.append(f"AGENT WOULD [{s.decision.action.value}{deal}]:")
            out.append("  " + s.agent_message.replace("\n", "\n  "))
            out.append(f"  why: {s.decision.rationale}")
            if s.decision.human_prompt:
                out.append(f"  NEEDS A HUMAN: {s.decision.human_prompt}")
            if s.human_reply is not None:
                out.append("TEAM ACTUALLY SENT:")
                out.append("  " + s.human_reply.replace("\n", "\n  "))
            out.append("")
        out.append(f"final status if the agent had run this thread: {self.final_status}")
        return "\n".join(out)


def replay_thread(transcript: str | list[Turn],
                  vm: ViewModel,
                  econ: CreatorEconomics,
                  ctx: Optional[CampaignContext] = None,
                  *,
                  brief: Optional[CampaignBrief] = None,
                  catalog: Optional[FormatCatalog] = None,
                  rate_card: Optional[RateCard] = None,
                  creator_aliases: Optional[list[str]] = None,
                  us_aliases: Optional[list[str]] = None,
                  creator_name: str = "there",
                  max_rounds: int = 3) -> ReplayReport:
    """Shadow-run the agent over a real thread. For each creator message the
    agent interprets, decides, and drafts — then the report pairs that with the
    reply your team actually sent. Nothing is sent anywhere."""
    turns = (parse_thread(transcript, creator_aliases=creator_aliases,
                          us_aliases=us_aliases)
             if isinstance(transcript, str) else list(transcript))
    ctx = ctx or CampaignContext()
    state = NegotiationState("replay", "replay", "replay", max_rounds=max_rounds)
    report = ReplayReport()

    for i, turn in enumerate(turns):
        if turn.sender != "creator":
            continue
        msg = interpret_message(turn.text)
        decision = decide(msg, state, vm, econ, ctx, brief=brief,
                          catalog=catalog, rate_card=rate_card)
        draft = write_message(decision, creator_name=creator_name,
                              thread_history=_turns_as_text(turns[:i]),
                              seed=i)
        human_reply = next((t.text for t in turns[i + 1:i + 2] if t.sender == "us"),
                           None)
        report.steps.append(ReplayStep(
            creator_text=turn.text, interpreted=msg, decision=decision,
            agent_message=draft, human_reply=human_reply))

    report.final_status = state.status.value
    return report


def _turns_as_text(turns: list[Turn]) -> str:
    return "\n".join(f"{t.sender}: {t.text}" for t in turns[-6:])
