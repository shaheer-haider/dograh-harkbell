"""The model's own reasoning, written as its reply, kept out of what a caller hears.

On staging round 181c (2026-11-09, run 3831) Harkbell's replica of Mirror Labs
answered "tomorrow at seven thirty" with:

    callback_asked is false, so I won't ask about a number. Let me give Priya
    the email route. Wait — the caller is Priya, not Dev. Let me re-read. The
    caller said "I'm Priya with Vivid Seats" — no, they said "I'm Dev with…

MiniMax runs with its thinking switched off, and this came back as ordinary
content, not as a thinking block, so nothing stood between it and the speech
service: on a phone call the caller hears the paragraph (Harkbell T-662).

A spoken reply to a caller never names one of the agent's parameters or tools
(``callback_asked``, ``route_call``), never calls the person on the line "the
caller" or "the user", and never tells itself to re-read. Those are the
signals, and any one of them makes a sentence reasoning. Everything after it
in the same paragraph is reasoning too: the model talks itself round to an
answer and that answer is reasoning as well ("— no, they said …"). A blank
line ends it, and what follows -- a model that reasons and then answers puts
the answer in a paragraph of its own -- is judged afresh.

Self-talk openers ("I should", "I need to", "Let me …", "Wait") are said to
callers too ("Let me read that back", "Wait, did you say seven or eleven?"),
so on their own they decide nothing. A sentence that opens with one is held
back until the next sentence settles it: dropped with it when that one is
reasoning, said when it is not or when the reply ends.

The speech service aggregates whole sentences before it speaks (Eleven v4 on
the dialogue socket accepts nothing else), so judging a sentence once it has
ended holds nothing back that would otherwise have been said sooner.
"""

import re

from loguru import logger

from api.services.pipecat.cut_off import CUT_OFF_MARKER
from api.services.pipecat.greeting_repeats import GreetingRepeatGuard
from api.services.pipecat.reasks import ReAskGuard

# A parameter or tool name: lower_snake_case. Not part of an email address or a
# web address, and not an address being read out ("first_last at gmail").
_IDENTIFIER = re.compile(
    r"(?<![\w@./\-])[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?![\w@/\-]|\.\w|\s+at\b)"
)
# The person on the line in the third person. "The caller ID" and "the user
# account" are things, and may be said.
_THIRD_PERSON = re.compile(
    r"\bthe\s+(?:caller|user)(?:'s|’s)?\b"
    r"(?!\s*(?:id|ids|name|names|account|accounts|profile|portal|login|number)\b)",
    re.IGNORECASE,
)
_SELF_CHECK = re.compile(
    r"\blet\s+me\s+(?:re-?read|re-?check|reconsider|re-?think|re-?evaluate)\b",
    re.IGNORECASE,
)
_SELF_TALK_OPENER = re.compile(
    r"^\W*(?:(?:okay|ok|so|hmm|right|now|actually)\b[\s,—–-]*)?"
    r"(?:i\s+should\b|i\s+need\s+to\b|i(?:'ll|’ll|\s+will)\s+call\b|let\s+me\b"
    r"|wait\b|hmm\b|actually\b)",
    re.IGNORECASE,
)
# A sentence ends at terminal punctuation followed by a space, before a line
# break, or at a line break. A blank line ends a paragraph.
#
# A match may only start at the first mark of a run, and takes the run without
# giving any of it back: written as ``[.!?…]+`` it tried every position inside
# a run and backtracked over the rest from each, so a run of 20,000 marks with
# no space after it -- a degenerate reply -- held one search for over four
# seconds, on the event loop, and every chunk of the reply searched it again
# (production, ``-harkbell.21``, run 4817: see MAX_HELD_CHARS).
_SENTENCE_END = re.compile(
    r"(?<![.!?…])[.!?…]++[\"'”’)\]]*+(?:[ \t]+|(?=\n))|\n"
)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
# The most text held while waiting for a sentence to end. Every chunk searches
# what is held, so a reply that never ends a sentence -- a degenerate MiniMax
# stream of marks or identifiers -- made each chunk dearer than the last, said
# nothing at all, and never tripped the stall guard because chunks kept
# coming. Production run 4817 (``-harkbell.21``, 2026-11-13): the reply to the
# first silence prompt said nothing for 64 s, and the engine process stopped
# answering anything soon after -- ECS killed it five minutes later. Past this
# many characters the held text is let through as a sentence of its own,
# which no spoken sentence comes near.
MAX_HELD_CHARS = 600


def is_reasoning(sentence: str) -> bool:
    """Whether a sentence is the model reasoning rather than speaking to a caller."""
    # The marker a cut-off line carries in the context names "the caller"; a
    # model that copies it into a reply is not reasoning, and the speech
    # filter removes the marker itself.
    sentence = sentence.replace(CUT_OFF_MARKER, "")
    return bool(
        _IDENTIFIER.search(sentence)
        or _THIRD_PERSON.search(sentence)
        or _SELF_CHECK.search(sentence)
    )


def opens_with_self_talk(sentence: str) -> bool:
    """Whether a sentence opens the way self-talk does, which callers also hear."""
    return bool(_SELF_TALK_OPENER.search(sentence))


class SpokenReasoningFilter:
    """One response's text, sentence by sentence, without the reasoning in it."""

    def __init__(
        self,
        repeats: "GreetingRepeatGuard | None" = None,
        reasks: "ReAskGuard | None" = None,
    ):
        self._buffer = ""
        self._held: list[str] = []
        self._reasoning = False
        self._dropped: list[str] = []
        # The call's greeting, when this reply must not say it again
        # (greeting_repeats.py).
        self._repeats = repeats
        # The run of re-asks this reply may be the third of (reasks.py).
        self._reasks = reasks
        # Everything this reply has handed on to be spoken.
        self.said = ""

    def feed(self, text: str) -> str:
        """Take the next piece of the stream; return what may be said now."""
        self._buffer += text
        said = []
        while True:
            paragraph = _PARAGRAPH_BREAK.search(self._buffer)
            sentence = _SENTENCE_END.search(self._buffer)
            if paragraph and (not sentence or paragraph.start() <= sentence.start()):
                said.append(self._sentence(self._buffer[: paragraph.start()]))
                said.append(self._paragraph_break(self._buffer[paragraph.start() : paragraph.end()]))
                self._buffer = self._buffer[paragraph.end() :]
                continue
            if sentence is None:
                if len(self._buffer) > MAX_HELD_CHARS:
                    cut = self._buffer.rfind(" ", 0, MAX_HELD_CHARS) + 1 or MAX_HELD_CHARS
                    said.append(self._sentence(self._buffer[:cut]))
                    self._buffer = self._buffer[cut:]
                    continue
                break
            if sentence.group() == "\n" and not self._buffer[sentence.end() :].strip():
                # A line break with nothing after it yet may be the first half
                # of a paragraph break.
                break
            said.append(self._sentence(self._buffer[: sentence.end()]))
            self._buffer = self._buffer[sentence.end() :]
        return "".join(said)

    def finish(self) -> str:
        """The response has ended: settle what is held and what is left."""
        said = self._sentence(self._buffer)
        self._buffer = ""
        said += self._release(self._held)
        self._held = []
        if self._repeats is not None:
            instead = self._repeats.settle(said_anything=bool(self.said.strip()))
            if instead:
                said += instead
                self.said += instead
        if self._dropped:
            logger.warning(
                "MiniMax spoke its own reasoning; dropped from the reply: "
                f"{''.join(self._dropped)[:300]!r}"
            )
            self._dropped = []
        return said

    def _sentence(self, sentence: str) -> str:
        if not sentence:
            return ""
        if self._reasoning or (sentence.strip() and is_reasoning(sentence)):
            self._reasoning = True
            self._dropped.extend(self._held)
            self._dropped.append(sentence)
            self._held = []
            return ""
        if sentence.strip() and opens_with_self_talk(sentence):
            self._held.append(sentence)
            return ""
        released = self._release([*self._held, sentence])
        self._held = []
        return released

    def _paragraph_break(self, gap: str) -> str:
        if self._reasoning:
            self._reasoning = False
            return ""
        released = self._release([*self._held, gap])
        self._held = []
        return released

    def _release(self, pieces: list[str]) -> str:
        """Hand sentences on to be spoken, less any repeat of the greeting."""
        kept = "".join(
            piece
            for piece in pieces
            if (self._repeats is None or not piece.strip() or self._repeats.keep(piece))
            and (self._reasks is None or self._reasks.pass_on(piece))
        )
        self.said += kept
        return kept
