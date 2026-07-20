"""Reply language + voice-template threading.

The two LLM prompt builders are the single choke point for outbound wording, so
these assert the language instruction and voice baseline actually reach the
prompt — and that the English default is left untouched (so golden parity, which
runs offline in English, is unaffected).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from oblsk_negotiator.behavior_tree import (Action, CreatorMessage, Decision,
                                            Intent, resolve_reply_language)
from oblsk_negotiator.qa import CampaignBrief, build_qa_llm_prompt
from oblsk_negotiator.prose import build_llm_prompt


def _answer_decision():
    return Decision(action=Action.ANSWER_QUESTION, deal=None, ev=None,
                    rationale="", answer_text="Here are the deliverables.")


def test_qa_prompt_adds_language_only_when_non_english():
    brief = CampaignBrief()
    es = build_qa_llm_prompt("what's the timeline?", brief, language="Spanish")
    assert "Spanish" in es["system"]
    en = build_qa_llm_prompt("what's the timeline?", brief, language="English")
    assert "Write your reply in" not in en["system"]


def test_prose_prompt_adds_language_only_when_non_english():
    es = build_llm_prompt(_answer_decision(), "Ana", language="Spanish")
    assert "Spanish" in es["system"]
    en = build_llm_prompt(_answer_decision(), "Ana", language="English")
    assert "Write your message in" not in en["system"]


def test_prose_prompt_injects_voice_template():
    tmpl = "We run every creator through our calculator; our budget lands us at X."
    out = build_llm_prompt(_answer_decision(), "Ana", voice_template=tmpl)
    assert tmpl in out["user"]
    assert "baseline voice" in out["user"].lower()
    # No template -> no baseline instruction.
    assert "baseline voice" not in build_llm_prompt(
        _answer_decision(), "Ana")["user"].lower()


def test_resolve_reply_language_precedence():
    msg = CreatorMessage(Intent.QUESTION, raw_text="hola", language="Spanish")
    # Campaign override wins.
    assert resolve_reply_language(CampaignBrief(reply_language="French"), msg) == "French"
    # Else detected message language.
    assert resolve_reply_language(CampaignBrief(), msg) == "Spanish"
    # Else English.
    assert resolve_reply_language(CampaignBrief(),
                                  CreatorMessage(Intent.QUESTION, raw_text="hi")) == "English"
    assert resolve_reply_language(None, CreatorMessage(Intent.UNCLEAR)) == "English"
