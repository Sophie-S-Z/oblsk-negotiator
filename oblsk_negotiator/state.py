"""
Conversation state. One DB row per thread, extending negotiationConversation.

Two ideas matter:
  autonomy_level  in human_approval the agent proposes and waits; in autonomous
                  the same proposals send on their own. Flipping it is how the
                  human comes out of the loop later, with no rewrite.
  phase           where the conversation is: questions, opening flat, negotiating,
                  closing.

State serializes and deserializes cleanly so the agent can pause and resume, and
it detects when a human steps into the thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json


class Status(str, Enum):
    OPEN = "open"
    ESCALATED = "escalated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"


class Phase(str, Enum):
    QA = "qa"
    OPENING = "opening"
    NEGOTIATING = "negotiating"
    CLOSING = "closing"


class AutonomyLevel(str, Enum):
    HUMAN_APPROVAL = "human_approval"
    AUTONOMOUS = "autonomous"


@dataclass
class ConcessionRecord:
    round: int
    offered_total: float
    offered_video_count: int
    asked_for: Optional[float]
    deal: dict


@dataclass
class ApprovalRecord:
    round: int
    action: str
    proposed_total: Optional[float]
    outcome: str                     # approved | rejected | edited | auto
    note: str = ""


@dataclass
class NegotiationState:
    creator_id: str
    campaign_id: str
    thread_id: str

    autonomy_level: AutonomyLevel = AutonomyLevel.HUMAN_APPROVAL
    # Frozen calculator inputs for this thread (view model + economics). The
    # stateless service builds it on the first turn and echoes it to the platform
    # as `evSnapshot`; resent on later turns it pins pricing so the ladder does
    # not drift as the creator's post history grows. See service._ev_snapshot.
    ev_snapshot: dict = field(default_factory=dict)

    phase: Phase = Phase.QA
    round_count: int = 0
    max_rounds: int = 3
    questions_answered: int = 0
    flat_offer_made: bool = False
    flat_revised: bool = False
    bundle_offered: bool = False
    call_proposed: bool = False

    concession_history: list[ConcessionRecord] = field(default_factory=list)
    approval_history: list[ApprovalRecord] = field(default_factory=list)

    last_human_touch_at: Optional[float] = None
    last_agent_message_at: Optional[float] = None
    last_thread_sender: str = "creator"

    status: Status = Status.OPEN
    final_total: Optional[float] = None
    final_deal: Optional[dict] = None
    escalation_reason: Optional[str] = None

    creator_first_ask: Optional[float] = None   # per-video, normalized at capture
    consecutive_rejections: int = 0             # creator rejections in a row
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        d["autonomy_level"] = self.autonomy_level.value
        d["phase"] = self.phase.value
        d["status"] = self.status.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "NegotiationState":
        d = json.loads(s)
        d["autonomy_level"] = AutonomyLevel(d["autonomy_level"])
        d["phase"] = Phase(d["phase"])
        d["status"] = Status(d["status"])
        d["concession_history"] = [ConcessionRecord(**c) for c in d["concession_history"]]
        d["approval_history"] = [ApprovalRecord(**a) for a in d["approval_history"]]
        return cls(**d)

    def human_intervened(self, thread_last_sender: str, thread_last_ts: float) -> bool:
        """True when a person sent the most recent message after the agent did."""
        if thread_last_sender == "human":
            if (self.last_agent_message_at is None or
                    thread_last_ts > self.last_agent_message_at):
                return True
        return False

    def record_concession(self, deal_dict: dict, total: float,
                          creator_ask: Optional[float],
                          video_count: Optional[int] = None):
        vc = video_count if video_count is not None \
            else deal_dict.get("video_count", 1)
        self.concession_history.append(ConcessionRecord(
            round=self.round_count, offered_total=total,
            offered_video_count=vc, asked_for=creator_ask, deal=deal_dict))

    def record_approval(self, action: str, proposed_total: Optional[float],
                        outcome: str, note: str = ""):
        self.approval_history.append(ApprovalRecord(
            round=self.round_count, action=action,
            proposed_total=proposed_total, outcome=outcome, note=note))
