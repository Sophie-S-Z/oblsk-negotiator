"""
Q&A mode. Answers a creator's questions before any rate is on the table:
deliverables, timeline, usage, payment, and anything off-topic. A budget
question is the signal to move to an offer.

answer_question is the entry point: LLM-drafted from the brief when available
(build_qa_llm_prompt), falling back to answer_from_brief — the deterministic
keyword version — offline. Neither path ever quotes a rate; money questions
route to the offer instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import llm


@dataclass
class CampaignBrief:
    """The facts the Q&A layer answers from. Populated from the campaign config in
    production; defaults here keep the demo runnable."""
    brand: str = "the brand"
    product: str = "the product"
    platform: str = "Instagram"
    content_type: str = "a Reel"
    deliverables: str = "one in-feed Reel, 30 to 60 seconds, plus a story frame"
    timeline: str = "about two weeks from when we lock the brief"
    usage_rights: str = "6 months of organic usage, no paid whitelisting unless we agree to it"
    exclusivity: str = "no category exclusivity, you can work with others outside this brand"
    revisions: str = "one round of revisions, we keep notes light"
    payment_terms: str = "net-30 after the content goes live, faster pay available"
    creative_freedom: str = "you have creative control, we share a few talking points and let you run with it"
    content_style: str = "your usual style — we don't over-direct the edit or the voice, whatever feels native to your feed"
    primary_format: Optional[str] = None      # the format the campaign opens on
    allowed_formats: Optional[list] = None     # formats the agent may bundle in
    extra_facts: dict = field(default_factory=dict)


_TOPIC_KEYWORDS = {
    "deliverables": ["deliverable", "how many", "what do i", "what would i",
                     "post", "reel", "video", "story", "content", "make"],
    "timeline": ["when", "deadline", "due", "timeline", "how long", "turnaround"],
    "usage_rights": ["usage", "rights", "own", "ownership", "repost", "reuse",
                     "whitelist", "ads", "boost"],
    "exclusivity": ["exclusive", "exclusivity", "compete", "competitor", "other brand"],
    "revisions": ["revision", "edit", "change", "approve", "approval", "feedback"],
    "payment_terms": ["payment", "invoice", "net-", "net ", "deposit", "upfront",
                      "paid", "when do i get"],
    "creative_freedom": ["creative", "script", "say", "freedom", "control",
                         "talking point", "guideline"],
    # Note: multiword "editing style" (not bare "edit", which is a revisions
    # keyword) and no bare "voice" (that collides with the voice_template
    # concept) — these ask what the content should feel like, not who edits it.
    "content_style": ["style", "vibe", "tone", "aesthetic", "editing style",
                      "look and feel", "how should it look", "content style",
                      "style of content"],
    "product": ["product", "brand", "who is", "what is", "about the", "company"],
    "budget": ["budget", "rate", "how much", "pay me", "fee", "price", "cost",
               "does this pay", "have in mind", "what were you thinking",
               "what are you thinking"],
}


def classify_question_topic(question: str) -> str:
    """Best-matching topic, or 'general'. Budget wins on ties so money questions
    always route to the offer."""
    q = question.lower()
    if any(kw in q for kw in _TOPIC_KEYWORDS["budget"]):
        return "budget"
    best, best_hits = "general", 0
    for topic, kws in _TOPIC_KEYWORDS.items():
        if topic == "budget":
            continue
        hits = sum(1 for kw in kws if kw in q)
        if hits > best_hits:
            best, best_hits = topic, hits
    return best


def signals_ready_for_offer(question: str) -> bool:
    return classify_question_topic(question) == "budget"


def can_answer(question: str, brief: CampaignBrief) -> bool:
    """Whether the brief covers this question. Anything it does not cover
    (ingredient details, medical claims, things said on a call) is a question
    for the team, not something to bluff through — the tree routes those to
    ASK_HUMAN instead of answering."""
    topic = classify_question_topic(question)
    if topic != "general":
        return True
    return _extra_fact_for(question, brief) is not None


def _extra_fact_for(question: str, brief: CampaignBrief):
    """An extra_facts entry whose key appears in the question (keys are keywords
    like 'equity', not full sentences)."""
    q = question.strip().lower()
    for key, fact in brief.extra_facts.items():
        if key.strip().lower() in q:
            return fact
    return None


def forward_nudge(questions_answered: int, offer_on_table: bool) -> str:
    """A short line that moves the conversation toward the next step, sized to how
    far along the thread is. Appended to a Q&A answer so the agent does not just
    answer in place: early on it offers to put a proposal together; deeper in it
    pushes for real numbers; once an offer is out it points back to that offer.
    Empty string when no nudge fits, so the caller can append unconditionally."""
    if offer_on_table:
        return "Let me know if you want me to adjust anything on the offer."
    if questions_answered <= 1:
        return "Whenever you are ready, I can put together a proposal that fits."
    return ("If the fit feels right on your end, I am happy to send over an offer "
            "so you can see real numbers.")


def answer_from_brief(question: str, brief: CampaignBrief) -> str:
    topic = classify_question_topic(question)
    if topic == "budget":
        return ("Good question. Let me put together a number that fits what we are "
                "asking for and send it right over.")
    answers = {
        "deliverables": f"For this one we are looking at {brief.deliverables}.",
        "timeline": f"Timeline is {brief.timeline}.",
        "usage_rights": f"On usage, {brief.usage_rights}.",
        "exclusivity": f"On exclusivity, {brief.exclusivity}.",
        "revisions": f"For revisions, {brief.revisions}.",
        "payment_terms": f"On payment, {brief.payment_terms}.",
        "creative_freedom": f"On the creative, {brief.creative_freedom}.",
        "content_style": f"On the style, {brief.content_style}.",
        "product": (f"It is for {brief.brand}, promoting {brief.product}. Happy to "
                    f"share more from the brief if useful."),
    }
    if topic in answers:
        return answers[topic]
    fact = _extra_fact_for(question, brief)
    if fact is not None:
        return fact
    return ("Happy to help with that. I can walk you through the deliverables, "
            "timeline, usage, or anything else on your mind, just let me know.")


def answer_question(question: str, brief: CampaignBrief,
                    thread_history: str = "") -> str:
    """Answer a creator question: LLM from the brief when available, keyword
    fallback otherwise. Budget questions never get an answer here — the caller
    routes them to an offer — but guard anyway so no path quotes a rate."""
    if classify_question_topic(question) == "budget":
        return answer_from_brief(question, brief)
    prompt = build_qa_llm_prompt(question, brief, thread_history)
    draft = llm.complete(prompt["system"], prompt["user"], max_tokens=300)
    if draft and "$" not in draft:
        return draft.strip()
    return answer_from_brief(question, brief)


def build_qa_llm_prompt(question: str, brief: CampaignBrief,
                        thread_history: str = "") -> dict:
    facts = "\n".join([
        f"Brand: {brief.brand}", f"Product: {brief.product}",
        f"Platform: {brief.platform}", f"Deliverables: {brief.deliverables}",
        f"Timeline: {brief.timeline}", f"Usage rights: {brief.usage_rights}",
        f"Exclusivity: {brief.exclusivity}", f"Revisions: {brief.revisions}",
        f"Payment: {brief.payment_terms}", f"Creative: {brief.creative_freedom}",
        f"Content style: {brief.content_style}"])
    system = (
        "You are an experienced influencer-marketing account manager answering a "
        "creator's question over email or DM. Warm, brief, real. Never reveal or "
        "imply you are an AI. Use only the campaign facts provided; if you do not "
        "know something, say you will check. Never quote a rate or budget; if asked "
        "about money, say you will send a number shortly. Under 90 words.")
    user = [f"Campaign facts:\n{facts}", f"\nCreator asked: {question}"]
    if thread_history:
        user.append(f"\nPrior thread (for tone):\n{thread_history}")
    user.append("\nWrite only the reply.")
    return {"system": system, "user": "\n".join(user)}
