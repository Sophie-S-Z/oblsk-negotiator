"""Generate docs/AGENT_GUIDE.pdf, the user guide for the negotiation agent.

Run from the repo root after feature changes so the guide stays current:
    py docs/make_guide.py

Layout is deliberately plain: one accent color, core fonts, no decoration.
Uses latin-1-safe characters only (core PDF fonts).
"""

from __future__ import annotations

import os
from fpdf import FPDF

INK = (34, 38, 42)          # near-black body text
ACCENT = (16, 101, 88)      # deep teal for headings
MUTED = (110, 116, 122)     # captions, labels
RULE = (216, 220, 223)
CODE_BG = (244, 246, 247)

OUT = os.path.join(os.path.dirname(__file__), "AGENT_GUIDE.pdf")


class Guide(FPDF):
    def __init__(self):
        super().__init__(format="letter")
        self.set_margins(22, 20, 22)
        self.set_auto_page_break(True, margin=20)

    def footer(self):
        self.set_y(-14)
        self.set_font("helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, f"Oblsk Negotiation Agent - User Guide - page {self.page_no()}",
                  align="C")

    # -- building blocks -----------------------------------------------------
    def h1(self, text):
        self.set_font("helvetica", "B", 20)
        self.set_text_color(*ACCENT)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def h2(self, text):
        self.ln(3)
        self.set_font("helvetica", "B", 12.5)
        self.set_text_color(*ACCENT)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*RULE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2.5)

    def p(self, text, size=10):
        self.set_font("helvetica", "", size)
        self.set_text_color(*INK)
        self.multi_cell(0, 5, text)
        self.ln(1.2)

    def bullet(self, head, text):
        self.set_font("helvetica", "B", 10)
        self.set_text_color(*INK)
        self.set_x(self.l_margin + 2)
        self.multi_cell(0, 5, f"{head}  -  {text}" if text else head,
                        markdown=False)
        self.ln(0.8)

    def item(self, head, text):
        self.set_x(self.l_margin + 2)
        self.set_font("helvetica", "B", 10)
        self.set_text_color(*INK)
        head_w = self.get_string_width(head) + 2
        self.cell(head_w, 5, head)
        self.set_font("helvetica", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(1.0)

    def code(self, *lines):
        self.set_fill_color(*CODE_BG)
        self.set_font("courier", "", 9)
        self.set_text_color(*INK)
        pad = 1.6
        self.ln(0.5)
        for line in lines:
            self.set_x(self.l_margin)
            self.cell(0, 5 + pad, "  " + line, fill=True,
                      new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def caption(self, text):
        self.set_font("helvetica", "", 8.5)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 4.2, text)
        self.ln(1.5)


pdf = Guide()
pdf.add_page()

# ---------------------------------------------------------------------------
pdf.h1("Oblsk Negotiation Agent")
pdf.set_font("helvetica", "", 10.5)
pdf.set_text_color(*MUTED)
pdf.multi_cell(0, 5, "User guide - how to run it, what it does, and where "
                     "people stay in the loop.")
pdf.ln(3)

pdf.h2("What it is")
pdf.p("The agent negotiates creator sponsorship deals over email or DM. It has "
      "two halves. The calculator simulates what a creator's posts will "
      "deliver and turns that into three prices. The negotiator runs the "
      "conversation one message at a time, reading every dollar off those "
      "prices. An LLM (Claude) writes the outgoing messages and reads the "
      "incoming ones; it never invents a price and never picks the move. "
      "Every outbound message waits for a human approval until you switch "
      "autonomy on.")

pdf.h2("The three prices (the ladder)")
pdf.item("Anchor.", "Where the agent opens. Low but credible - a calculated "
         "number, not a lowball.")
pdf.item("Target.", "Where it aims. The fee that hits the campaign's ROI goal "
         "(default 3x) at expected delivery. The agent revises to here once "
         "if the creator pushes back, and no further.")
pdf.item("Walk-away.", "The ceiling. Above this, an ordinary underperformance "
         "(a bottom-10% outcome) loses money. The agent never offers it and "
         "escalates asks far above it.")
pdf.p("If the flat offer stalls, the agent restructures instead of overpaying: "
      "more videos at a fair per-video rate, so the total grows while the "
      "rate holds. Full math and the client-facing justification for every "
      "number: docs/CALCULATOR.md.")

pdf.h2("Setup")
pdf.code("pip install -r requirements.txt",
         "pip install anthropic          # optional, enables the LLM",
         "set ANTHROPIC_API_KEY=...      # optional; omit to run offline",
         "python -m pytest               # 67 checks, all offline")
pdf.p("Without a key everything still runs: messages come from fixed "
      "templates and incoming text is read by keyword rules. With a key, "
      "Claude drafts each message from the live thread and reads real-world "
      "phrasing properly. Use the key for real work; offline mode is for "
      "tests and demos.")

pdf.h2("One file per campaign")
pdf.p("A campaign is a YAML file (copy examples/unest_campaign.yaml). It is "
      "the only thing you edit per client:")
pdf.item("brief:", "the facts the agent answers questions from - "
         "deliverables, timeline, usage rights, exclusivity, payment. "
         "Add keyword-keyed extra_facts (e.g. equity) to widen what it can "
         "answer without a human.")
pdf.item("economics:", "conversion rate and LTV. Every price scales with "
         "these - calibrate them against real attribution before trusting "
         "absolute dollars.")
pdf.item("negotiation:", "the stance - ROI target, opening aggressiveness, "
         "bundle size, the approval ceiling, call windows.")
pdf.item("us_aliases:", "names, emails, or domains that identify your side "
         "of an email thread, so replay knows who is who.")

pdf.add_page()
pdf.h2("Ways to run it")
pdf.item("Talk to it yourself.", "You type as the creator, the agent replies "
         "live. This is the exact read-price-draft the platform runs on real "
         "Gmail threads - here you type the messages. The fastest way to see "
         "it is a real agent, not a canned demo.")
pdf.code("py demo.py --chat",
         "py demo.py --chat --campaign examples/unest_campaign.yaml --median 150000")
pdf.item("See the calculation.", "Print the plain-English derivation of the "
         "three prices - reach, then dollars, then the ladder - so no number "
         "is a black box.")
pdf.code("py demo.py --explain")
pdf.item("Simulate.", "A rule-based creator plays the other side; you watch "
         "the full negotiation with approvals and the reasoning for each move. "
         "--random gives a fresh creator every run.")
pdf.code("py demo.py --random",
         "py demo.py --floor 2800 --opens ask --ask 3500",
         "py demo.py --campaign examples/unest_campaign.yaml --count 20 --quiet")
pdf.item("Replay a real thread.", "Paste a Gmail thread into a text file "
         "as-is (headers, signatures, quoted replies - the parser handles "
         "them). The agent shadows the deal: at each creator message you see "
         "how it read the message, what it would have done and why, next to "
         "what your team actually sent. Nothing is ever sent.")
pdf.code("py demo.py --campaign examples/unest_campaign.yaml \\",
         "           --replay examples/unest_thread.txt --median 150000")
pdf.caption("--median is the creator's typical per-video views, so the agent "
            "prices the same deal your team was pricing.")
pdf.item("Spar against Claude.", "Claude role-plays a talent manager with a "
         "private playbook you write; the agent negotiates back. Use the "
         "transcript to judge the back-and-forth before trusting it live. "
         "Needs an API key.")
pdf.code('py demo.py --spar "floor $2,600/video, opens at $4,500, '
         'concedes slowly"')
pdf.item("Notebook.", "notebooks/Oblsk_Negotiation_Agent_Demo.ipynb walks "
         "the pricing model, a live negotiation, and the replay "
         "interactively.")

pdf.h2("Reading a decision")
pdf.p("Every move the agent proposes shows the same four things: the action "
      "(open, revise, bundle, hold, accept, ...), the deal (total, videos, "
      "per-video rate), the expected ROI with its downside and upside, and a "
      "one-sentence rationale. If a line starting with NEEDS A HUMAN "
      "appears, the agent wants something from the team - read it, it says "
      "exactly what.")

pdf.h2("Where people stay in the loop")
pdf.item("Approval gate.", "In the default mode a person approves every "
         "outbound message. In autonomous mode, anything above the campaign's "
         "dollar ceiling still requires sign-off.")
pdf.item("ask_human.", "A question the brief cannot answer, or a reference "
         "to a call the agent was not on, pauses the thread with a specific "
         "request. Answer it; the agent keeps the thread.")
pdf.item("propose_call.", "When the creator wants a call, the agent proposes "
         "your call windows and flags a teammate to send the invite and run "
         "it. After the call, write what was agreed back into the thread - "
         "the agent cannot hear calls.")
pdf.item("escalate_human.", "Extreme asks, exhausted rounds, and any "
         "contract-terms negotiation (exclusivity, equity, kill fees) hand "
         "the whole thread to a person. The agent never negotiates paper.")
pdf.item("hold_firm.", "Not a handoff, but worth knowing: a follow-up with "
         "no counter gets the standing offer restated. The agent never bids "
         "against itself.")

pdf.h2("Plugging into the Oblsk platform")
pdf.p("The platform already polls Gmail threads, classifies messages, and "
      "runs the review UI. The agent plugs in as an HTTP service that "
      "suggests the next reply:")
pdf.code("py -m oblsk_negotiator.service --campaign examples/unest_campaign.yaml "
         "--port 8788")
pdf.p("POST /suggest takes the thread and the creator's view numbers; the "
      "response has the drafted message, the move, the ladder, whether it "
      "needs approval, and any task for a human. The service is stateless: "
      "state is derived from the thread itself, so offers a person made count "
      "as the agent's positions and it picks threads up mid-flight. Wiring "
      "details and the exact request/response shapes: docs/INTEGRATION.md.")

pdf.h2("Walk-away vs. a creator's floor")
pdf.p("Walk-away is OUR ceiling, computed from the numbers: above it, an "
      "ordinary bottom-10% outcome loses money. A creator's floor is THEIR "
      "private minimum - the agent never sees it. In demos, the 'sim "
      "creator's hidden floor' line is a parameter of the fake counterpart, "
      "shown so you can judge the run; real threads have no such input. When "
      "target and walk-away print as the same number, the creator's views are "
      "so volatile that the fee hitting the ROI goal would lose money on a "
      "bad outcome - the ceiling clamps the target, and the printout says so.")

pdf.h2("Environment variables")
pdf.item("ANTHROPIC_API_KEY", "- enables the LLM (drafting, reading, sparring).")
pdf.item("OBLSK_LLM_MODEL", "- model override; defaults to claude-opus-4-8.")
pdf.item("OBLSK_NO_LLM=1", "- force offline mode even if a key is set.")

pdf.h2("If something looks wrong")
pdf.p("Prices too low or too high across the board: fix economics in the "
      "campaign file - every number is derived from conversion x LTV. A "
      "message read incorrectly in replay: you are probably offline; the "
      "keyword reader is approximate, the LLM reader is not. A thread "
      "stopped: look for the NEEDS A HUMAN line; the agent is waiting on "
      "you, not stuck.")

pdf.output(OUT)
print(f"wrote {OUT} ({pdf.page_no()} pages)")
