"""The call's greeting, said once.

Production run 4603 (2026-11-12, the copy of Mirror Labs on ``-harkbell.15``):
the caller said "Hello?" while the greeting was still playing, and the next
reply was the greeting again, word for word -- "Hi, this is Emma at Mirror
Labs. Thanks for calling. How can I help you?". The prompt has said not to
restart the greeting since config 145, in three forms; MiniMax does it anyway
often enough that a caller on a bad line can hear it four times (T-670). A
prose rule is not a guarantee, so this is one.

The greeting is the call's first spoken reply that no caller's words came
before: the one the model opens the call with, after any notice the start
pathway played. A later reply that opens by repeating the greeting's first
sentence is re-greeting, and from there every greeting sentence it repeats
is dropped -- except a question: "How can I help you?" stays, so the caller
still gets an answer. A re-greeting with nothing else in it says only the
greeting's question, and nothing when the greeting asked none.

It is that narrow on purpose. Nothing but the greeting's own sentences is
ever dropped (a caller may ask for an email address or a read-back twice and
must hear it twice), and a greeting sentence repeated on its own -- "Thanks
for calling." in a goodbye -- is not a re-greeting and is said.
"""

import re

from loguru import logger

from api.services.pipecat.cut_off import CUT_OFF_MARKER
from api.services.pipecat.phrase_turn_start import normalize_transcript

_SENTENCE = re.compile(r".+?(?:[.!?…]+[\"'”’)\]]*(?=\s|$)|$)", re.DOTALL)
_DASHES = re.compile(r"[—–]")
_CUT_OFF_SUFFIX = re.compile(rf"\s*—\s*{re.escape(CUT_OFF_MARKER)}\s*$")


def sentence_key(sentence: str) -> str:
    """A sentence as compared: case, punctuation, dashes and spacing ignored."""
    return normalize_transcript(_DASHES.sub(" ", sentence))


def is_question(sentence: str) -> bool:
    return sentence.strip().rstrip("\"'”’)]").endswith("?")


def split_sentences(text: str) -> list[str]:
    return [match.group().strip() for match in _SENTENCE.finditer(text) if match.group().strip()]


def greeting_from_context(context) -> str | None:
    """The call's greeting as the context holds it: the first assistant line
    before any caller's words, without a cut-off marker. None when the caller
    spoke first or nothing was said yet."""
    for message in getattr(context, "messages", None) or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            return None
        content = message.get("content")
        if role == "assistant" and isinstance(content, str) and content.strip():
            return _CUT_OFF_SUFFIX.sub("", content).strip() or None
    return None


def caller_has_spoken(context) -> bool:
    return any(
        isinstance(message, dict) and message.get("role") == "user"
        for message in getattr(context, "messages", None) or []
    )


class GreetingRepeatGuard:
    """One later reply, judged sentence by sentence against the greeting."""

    def __init__(self, greeting: str):
        sentences = split_sentences(greeting)
        self._opening = (
            sentence_key(sentences[0]) if sentences and not is_question(sentences[0]) else ""
        )
        self._keys = {sentence_key(s) for s in sentences if not is_question(s)} - {""}
        self._questions = [s for s in sentences if is_question(s)]
        self.regreeting = False
        self.dropped: list[str] = []

    def keep(self, sentence: str) -> bool:
        """Whether a sentence of the reply may be said."""
        key = sentence_key(sentence)
        if not key or not self._opening or is_question(sentence):
            return True
        if key == self._opening:
            self.regreeting = True
        if self.regreeting and key in self._keys:
            self.dropped.append(sentence.strip())
            return False
        return True

    def settle(self, *, said_anything: bool) -> str:
        """The reply has ended: what to say when the re-greeting left nothing."""
        if self.dropped:
            logger.warning(
                f"Dropped a repeat of the call's greeting: {' '.join(self.dropped)!r}"
            )
        if self.regreeting and not said_anything:
            return " ".join(self._questions)
        return ""
