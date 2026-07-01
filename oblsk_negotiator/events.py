"""
An append-only record of a negotiation.

Today the live state is a single mutable NegotiationState row. This log is the
same history written the other way round: an ordered, append-only list of events
that is never edited in place. Events are appended by the runner as things happen
(a message arrives, a decision is made, an approval lands, a message is sent),
and `fold_events` replays them back into the negotiation facts the row tracks.
Because the row is reconstructable from the log, the log can become the source of
truth and the row a cache; for now the two are kept side by side and a test
asserts they agree.

The fold derives the structural facts (which offers were made, the status, the
final total) purely from the event stream, so the log alone is enough to know
what happened. The one thing it does not re-derive is the QA counter, which the
decision events carry directly.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import time

# Event kinds.
MESSAGE_RECEIVED = "message_received"   # a creator message arrived
DECISION = "decision"                   # the tree chose a move
APPROVAL = "approval"                   # a proposal was approved (or auto-approved)
SENT = "sent"                           # an approved message went out
HUMAN_REJECTED = "human_rejected"       # a reviewer rejected a proposal; thread handed off

# Actions that advance a negotiation round. A flat open, a revised flat, a bundle
# and a sweetener each cost a round; so does accepting the creator's own ask (that
# handler carries a deal, which is how we tell it from accepting on confirmation).
_ROUND_ACTIONS = frozenset({"opening_flat", "escalate_bundle", "add_sweetener"})


@dataclass(frozen=True)
class Event:
    seq: int
    ts: float
    kind: str
    data: dict


class EventLog:
    """Append-only list of events. There is no update or delete; the only writes
    are appends, each stamped with a monotonic sequence number."""

    def __init__(self):
        self._events: list[Event] = []

    def append(self, kind: str, **data) -> Event:
        ev = Event(seq=len(self._events), ts=time.time(), kind=kind, data=data)
        self._events.append(ev)
        return ev

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def to_json(self) -> str:
        return json.dumps([asdict(e) for e in self._events])

    @classmethod
    def from_json(cls, s: str) -> "EventLog":
        log = cls()
        for d in json.loads(s):
            log._events.append(Event(**d))
        return log


def fold_events(events) -> dict:
    """Replay the log into the negotiation facts it implies: the same fields the
    NegotiationState row tracks. A live row and a fold of its log should agree on
    every field returned here."""
    st = {
        "round_count": 0,
        "questions_answered": 0,
        "flat_offer_made": False,
        "flat_revised": False,
        "bundle_offered": False,
        "status": "open",
        "final_total": None,
    }
    last_offer_total = None
    for e in events:
        if e.kind != DECISION:
            if e.kind == HUMAN_REJECTED:
                st["status"] = "escalated"
            continue
        action = e.data.get("action")
        total = e.data.get("total")
        has_deal = total is not None
        if "qa_count" in e.data:
            st["questions_answered"] = e.data["qa_count"]
        if action in _ROUND_ACTIONS or (action == "accept" and has_deal):
            st["round_count"] += 1
        if action == "opening_flat":
            # The first flat opens; a second flat is the revised offer.
            if st["flat_offer_made"]:
                st["flat_revised"] = True
            st["flat_offer_made"] = True
            last_offer_total = total
        elif action == "escalate_bundle":
            st["bundle_offered"] = True
            last_offer_total = total
        elif action == "add_sweetener":
            last_offer_total = total
        elif action == "accept":
            st["status"] = "accepted"
            if has_deal:
                last_offer_total = total
        elif action == "escalate_human":
            st["status"] = "escalated"
        elif action == "soft_close":
            st["status"] = "rejected"
    # The final total is whatever was last on the table, accepted or not, which is
    # how the runner reports it too.
    st["final_total"] = last_offer_total
    return st
