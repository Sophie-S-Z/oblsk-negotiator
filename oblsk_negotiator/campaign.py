"""
Campaign configuration. One YAML file per campaign holds everything the agent
needs to negotiate on a brand's behalf: the facts it answers questions from
(the brief), the economics that price every deal, the negotiation stance
(every CampaignContext knob), and how to recognize our side of an email
thread. Swapping campaigns is swapping files — no code changes.

See examples/unest_campaign.yaml for a complete, annotated example built from
a real campaign.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from typing import Optional

import yaml

from .behavior_tree import CampaignContext
from .ev_engine import CreatorEconomics
from .qa import CampaignBrief


@dataclass
class Campaign:
    name: str
    brief: CampaignBrief
    econ: CreatorEconomics
    ctx: CampaignContext
    # Names/emails/domains that identify our side of a thread (for replay).
    us_aliases: list[str] = field(default_factory=list)


def _build(cls, data: dict, section: str):
    """Construct a dataclass from a config section, failing loudly on unknown
    keys so a typo in the YAML surfaces at load, not as a silently-default
    negotiation stance."""
    allowed = {f.name for f in dc_fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)!r} in campaign "
                         f"section '{section}' (allowed: {sorted(allowed)})")
    return cls(**data)


def load_campaign(path: str) -> Campaign:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    known = {"name", "brief", "economics", "negotiation", "us_aliases"}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown top-level key(s) {sorted(unknown)!r} in "
                         f"{path} (allowed: {sorted(known)})")
    if "economics" not in data:
        raise ValueError(f"{path} must define an 'economics' section "
                         f"(conversion_rate, ltv_usd)")
    return Campaign(
        name=data.get("name", path),
        brief=_build(CampaignBrief, data.get("brief", {}), "brief"),
        econ=_build(CreatorEconomics, data["economics"], "economics"),
        ctx=_build(CampaignContext, data.get("negotiation", {}), "negotiation"),
        us_aliases=list(data.get("us_aliases", [])))
