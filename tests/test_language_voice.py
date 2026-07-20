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
from oblsk_negotiator.ev_engine import Deal
from oblsk_negotiator.qa import CampaignBrief, build_qa_llm_prompt, _MONEY_LEAK_RE
from oblsk_negotiator.prose import build_llm_prompt, write_message, _numbers_intact

os.environ["OBLSK_NO_LLM"] = "1"   # force the deterministic template fallback


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


def test_template_fallback_flags_non_english_for_translation():
    # Offline (no LLM) -> the English template. A non-English thread must not
    # ship English silently; it gets a reviewer note. English threads don't.
    d = Decision(action=Action.OPENING_FLAT,
                 deal=Deal(video_count=1, flat_per_video=2000.0),
                 ev=None, rationale="")
    es = write_message(d, creator_name="Ana", language="Spanish")
    assert "translate to Spanish" in es
    en = write_message(d, creator_name="Ana", language="English")
    assert "translate to" not in en


def test_numbers_intact_is_locale_tolerant():
    # A multi-video deal ($6,000 total, $2,000 each). A localized draft ("$6.000",
    # "2 000") should still pass the number check.
    d = Decision(action=Action.OPENING_FLAT,
                 deal=Deal(video_count=3, flat_per_video=2000.0),
                 ev=None, rationale="")
    assert _numbers_intact(d, "Serían $6.000 en total, unos 2 000 por vídeo.")
    # A draft that dropped the numbers entirely still fails.
    assert not _numbers_intact(d, "Suena genial, hablemos pronto.")


def test_money_leak_guard_catches_non_dollar_currencies():
    for leak in ("Podemos pagar €1.500", "Serían 2000 dólares", "R$3.000", "about 2k"):
        assert _MONEY_LEAK_RE.search(leak), leak
    # A clean, money-free answer passes.
    assert not _MONEY_LEAK_RE.search("Trabajamos con tu estilo habitual, sin guion.")
