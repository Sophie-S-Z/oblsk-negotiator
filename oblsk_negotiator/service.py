"""
The integration surface for the Oblsk platform (rockcorp/oblsk).

The platform already owns the negotiation pipeline: Gmail-backed
`negotiationConversation` threads, message polling and classification, the
approval UI, and sending. What it asks a negotiator for is one thing — given
the thread so far, suggest the next reply. Its Convex node action calls
`generateNegotiatorSuggestion(input) -> NegotiatorSuggestion`; this service is
that call over HTTP, powered by this agent instead:

    POST /suggest
        body:     the platform's NegotiatorInput shape (messages with
                  brand/creator roles, medianViewsPerVideo, recentPostViews,
                  creatorHandle, ...)
        response: the platform's NegotiatorSuggestion shape (message,
                  strategy{kind, title, instructions, packageTerms, debug},
                  model) plus an `agent` block carrying the full decision
                  (action, deal, ladder, requiresApproval, humanPrompt).

Stateless by design: the platform's thread is the source of truth, so every
request carries the whole thread and negotiation state is derived from it —
offers our side already made (whoever made them, human or agent) are read out
of the brand messages, the creator's first ask out of theirs. No second store,
nothing to migrate, humans and the agent can freely interleave on a thread.

Run it:
    py -m oblsk_negotiator.service --campaign examples/unest_campaign.yaml --port 8788

Set OBLSK_SERVICE_TOKEN to require `Authorization: Bearer <token>` on every
request. See docs/INTEGRATION.md for the Convex-side wiring.
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np

from . import llm
from .behavior_tree import Action, Decision, decide, resolve_reply_language
from .bundles import flat_offer
from .campaign import Campaign, load_campaign
from .ev_engine import CreatorEconomics, ViewModel, deal_to_dict, fit_view_model
from .pricing import price_ladder
from .prose import write_message
from .replay import _WANTS_CALL_RE, Turn, heuristic_interpret, interpret_message
from .state import NegotiationState

# Nearest fit into the platform's closed NegotiatorStrategyKind union (it only
# drives UI display; the precise action rides in the `agent` block).
_PLATFORM_KIND = {
    Action.OPENING_FLAT: "initial_offer",
    Action.HOLD_FIRM: "initial_offer",
    Action.ANSWER_QUESTION: "initial_offer",
    Action.ESCALATE_BUNDLE: "package_counter",
    Action.ADD_SWEETENER: "package_counter",
    Action.ACCEPT: "accept",
    Action.PROPOSE_CALL: "missing_context",
    Action.ASK_HUMAN: "missing_context",
    Action.ESCALATE_HUMAN: "escalate",
    Action.PAUSE_HUMAN: "escalate",
    Action.SOFT_CLOSE: "escalate",
}

_TITLE = {
    Action.OPENING_FLAT: "Offer one flat number",
    Action.HOLD_FIRM: "Hold the standing offer",
    Action.ANSWER_QUESTION: "Answer their question",
    Action.ESCALATE_BUNDLE: "Counter with a package",
    Action.ADD_SWEETENER: "Hold the number, add a sweetener",
    Action.ACCEPT: "Accept their ask",
    Action.PROPOSE_CALL: "Get a call scheduled (a person runs it)",
    Action.ASK_HUMAN: "Needs input from the team",
    Action.ESCALATE_HUMAN: "Hand the thread to a person",
    Action.PAUSE_HUMAN: "Paused: a teammate is in the thread",
    Action.SOFT_CLOSE: "Close gently, keep the door open",
}


def _view_model_from(payload: dict) -> ViewModel:
    views = [v for v in (payload.get("recentPostViews") or []) if v and v > 0]
    if len(views) >= 6:
        return fit_view_model(views)
    median = payload.get("medianViewsPerVideo")
    if median and median > 0:
        # Thin history: a synthetic lognormal around the platform's median,
        # deterministic so the same creator always prices the same.
        synthetic = np.random.default_rng(0).lognormal(np.log(median), 0.6, 24)
        return fit_view_model(synthetic)
    raise ValueError("need recentPostViews (>=6 posts) or medianViewsPerVideo")


def _turns_from(payload: dict) -> list[Turn]:
    msgs = sorted(payload.get("messages") or [],
                  key=lambda m: m.get("receivedAt") or 0)
    turns = [Turn("us" if m.get("role") == "brand" else "creator",
                  (m.get("content") or "").strip())
             for m in msgs if (m.get("content") or "").strip()]
    if not turns:
        raise ValueError("messages is empty")
    if turns[-1].sender != "creator":
        raise ValueError("last message is ours; nothing to reply to")
    return turns


def _state_from_turns(turns: list[Turn]) -> NegotiationState:
    """Derive negotiation state from the thread itself. Brand-side dollar
    offers become recorded concessions whoever sent them (the agent or a
    person), so the agent picks a thread up mid-flight without re-opening at
    the anchor; the creator's first ask is captured per-video the same way the
    live tree captures it."""
    state = NegotiationState("platform", "platform", "thread")
    last_brand_vc = 1
    for turn in turns:
        # History is reconstructed with the deterministic heuristic, not the LLM.
        # The service is stateless, so state is rebuilt on every poll; running the
        # LLM over the whole thread each time would be both slow (one call per
        # past turn) and non-reproducible (the same thread could reconstruct to a
        # different offer twice). The live incoming message still gets the LLM
        # (see suggest); the authoritative fix for messy human-phrased offers is a
        # structured per-thread ledger, which is owned platform-side.
        m = heuristic_interpret(turn.text)
        if turn.sender == "us":
            if m.ask_total_usd:
                vc = max(m.ask_video_count or 1, 1)
                deal = flat_offer(m.ask_total_usd, vc)
                if vc > 1:
                    state.bundle_offered = True
                elif state.flat_offer_made:
                    state.flat_revised = True
                state.flat_offer_made = True
                state.round_count += 1
                state.record_concession(deal_to_dict(deal), deal.total_usd,
                                        None, video_count=vc)
                last_brand_vc = vc
            if _WANTS_CALL_RE.search(turn.text):
                state.call_proposed = True
        else:
            if m.intent.value == "question":
                state.questions_answered += 1
            if (m.ask_total_usd and state.flat_offer_made
                    and state.creator_first_ask is None):
                # No count stated means the ask covers the standing offer.
                state.creator_first_ask = (
                    m.ask_total_usd / max(m.ask_video_count or last_brand_vc, 1))
            state.consecutive_rejections = (
                state.consecutive_rejections + 1
                if m.intent.value == "rejecting" else 0)
    return state


def _variance(views: list[float], vm: ViewModel) -> dict:
    n = len(views)
    if n < 6:
        label = "insufficient_data"
    else:
        cov = vm.coefficient_of_variation()
        label = "stable" if cov < 0.4 else "moderate" if cov < 0.9 else "volatile"
    return {
        "postCount": n,
        "coefficientOfVariation": round(vm.coefficient_of_variation(), 3) if n else None,
        "label": label,
        "minViews": min(views) if views else None,
        "maxViews": max(views) if views else None,
        "averageViews": round(float(np.mean(views))) if views else None,
    }


def _suggestion_from(decision: Decision, message: str, payload: dict,
                     msg_ask, ladder, vm: ViewModel, camp: Campaign,
                     ev_snapshot: dict | None = None) -> dict:
    d = decision.deal
    package = None
    if d is not None and d.video_count > 1:
        package = {
            "videoCount": d.video_count,
            "totalUsd": round(d.total_usd, 2),
            "effectivePerVideoUsd": round(d.total_usd / d.video_count, 2),
            "requestedVideoCount": msg_ask.ask_video_count or d.video_count,
        }
    creator_ask = None
    if msg_ask.ask_total_usd:
        creator_ask = {
            "amountUsd": msg_ask.ask_total_usd,
            "unit": "per_video" if (msg_ask.ask_video_count or 1) == 1 else "total",
            "suggestedVideoCount": msg_ask.ask_video_count or 1,
            "source": "message",
            "rawText": msg_ask.raw_text[:300],
        }
    instructions = decision.rationale
    if decision.human_prompt:
        instructions += f" NEEDS A HUMAN: {decision.human_prompt}"
    views = [v for v in (payload.get("recentPostViews") or []) if v and v > 0]
    return {
        "message": message,
        "strategy": {
            "kind": _PLATFORM_KIND[decision.action],
            "title": _TITLE[decision.action],
            "instructions": instructions,
            "packageTerms": package,
            "cpmTerms": None,
            "debug": {
                "creatorAsk": creator_ask,
                "variance": _variance(views, vm),
                "targetPerVideoUsd": round(ladder.target, 2),
                "maxPerVideoUsd": round(ladder.walk_away, 2),
                "targetCpmUsd": None,
                "maxCpmUsd": None,
                "initialCounterPerVideoUsd": round(ladder.anchor, 2),
                "maxAdditionalVideos": camp.ctx.target_videos - 1,
                "reasons": [decision.rationale],
            },
        },
        "model": llm.model_name() if llm.llm_available() else None,
        "agent": {
            "action": decision.action.value,
            "requiresApproval": decision.requires_approval,
            "humanPrompt": decision.human_prompt,
            "deal": deal_to_dict(d) if d is not None else None,
            "ladder": {
                "anchorUsd": round(ladder.anchor, 2),
                "targetUsd": round(ladder.target, 2),
                "walkAwayUsd": round(ladder.walk_away, 2),
                "binding": ladder.binding,
                "downsideLimited": ladder.downside_limited,
            },
            # Frozen calculator inputs for this thread. Store and resend this in
            # the next request's `evSnapshot` to keep pricing stable as the
            # creator's post history grows.
            "evSnapshot": ev_snapshot,
        },
    }


def _ev_snapshot(vm: ViewModel, econ: CreatorEconomics) -> dict:
    """Freeze the calculator's per-thread inputs: the fitted reach model and the
    revenue economics. Echoed back to the platform so later turns price off the
    same numbers instead of re-fitting from a post history that keeps growing."""
    return {"viewModel": vm.to_dict(),
            "econ": {"conversionRate": econ.conversion_rate,
                     "ltvUsd": econ.ltv_usd}}


def _from_ev_snapshot(snapshot: dict) -> tuple[ViewModel, CreatorEconomics]:
    # A malformed snapshot is a bad client payload (-> 400), not a server fault
    # (-> 500), so surface it as ValueError rather than letting a KeyError/TypeError
    # bubble up as an unhandled 500.
    try:
        e = snapshot["econ"]
        return (ViewModel.from_dict(snapshot["viewModel"]),
                CreatorEconomics(conversion_rate=float(e["conversionRate"]),
                                 ltv_usd=float(e["ltvUsd"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed evSnapshot: {exc}") from exc


def suggest(payload: dict, camp: Campaign) -> dict:
    """One suggestion for one thread. Raises ValueError on a bad payload."""
    turns = _turns_from(payload)
    # Freeze the calculator inputs for the life of the thread. If the platform
    # sent back the snapshot from an earlier turn, price off that; otherwise
    # build it now from the payload and echo it so the next turn can pin it.
    incoming = payload.get("evSnapshot")
    if incoming:
        vm, econ = _from_ev_snapshot(incoming)
    else:
        vm, econ = _view_model_from(payload), camp.econ
    snapshot = incoming or _ev_snapshot(vm, econ)
    state = _state_from_turns(turns[:-1])
    state.ev_snapshot = snapshot
    msg = interpret_message(turns[-1].text)
    decision = decide(msg, state, vm, econ, camp.ctx, brief=camp.brief)
    history = "\n".join(f"{t.sender}: {t.text}" for t in turns[:-1][-6:])
    message = write_message(decision,
                            creator_name=payload.get("creatorHandle") or "there",
                            thread_history=history,
                            language=resolve_reply_language(camp.brief, msg),
                            voice_template=camp.brief.voice_template)
    ladder = price_ladder(vm, econ, camp.ctx.pricing_policy())
    return _suggestion_from(decision, message, payload, msg, ladder, vm, camp,
                            ev_snapshot=snapshot)


# ---- HTTP wrapper (stdlib only, no new dependencies) --------------------------

class _Handler(BaseHTTPRequestHandler):
    campaign: Campaign = None   # set by serve()

    def _send(self, code: int, body: dict):
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        token = os.environ.get("OBLSK_SERVICE_TOKEN")
        if not token:
            return True
        return self.headers.get("Authorization") == f"Bearer {token}"

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "campaign": self.campaign.name,
                             "llm": llm.llm_available()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorized():
            self._send(401, {"error": "missing or wrong bearer token"})
            return
        if self.path != "/suggest":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, suggest(payload, self.campaign))
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):   # one clean line per request
        print(f"{self.address_string()} {fmt % args}")


def serve(campaign_path: str, host: str = "127.0.0.1", port: int = 8788):
    camp = load_campaign(campaign_path)
    _Handler.campaign = camp
    server = ThreadingHTTPServer((host, port), _Handler)
    guard = "bearer token required" if os.environ.get("OBLSK_SERVICE_TOKEN") \
        else "no auth (set OBLSK_SERVICE_TOKEN to require one)"
    print(f"negotiator service on http://{host}:{port}  |  campaign: {camp.name}")
    print(f"llm: {'enabled' if llm.llm_available() else 'offline fallback'}  |  {guard}")
    server.serve_forever()


def main():
    p = argparse.ArgumentParser(description="Negotiation agent HTTP service "
                                            "for the Oblsk platform.")
    p.add_argument("--campaign", required=True,
                   help="campaign YAML (see examples/unest_campaign.yaml)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8788)
    args = p.parse_args()
    serve(args.campaign, args.host, args.port)


if __name__ == "__main__":
    main()
