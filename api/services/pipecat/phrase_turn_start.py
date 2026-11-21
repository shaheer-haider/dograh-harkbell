"""Phrase-aware turn start: back-channels never interrupt, break-in words always do.

``MinWordsUserTurnStartStrategy`` decides by count alone, and a count cannot
tell "uh-huh" from "hold on": at one word both interrupt, at three neither
does. This strategy keeps the count as the fallback and adds two phrase lists,
applied only while the bot is speaking — when it is not, one word opens a
turn exactly as before.

- A transcript made only of ignore phrases ("yeah", "mm-hmm", "yeah yeah",
  "okay, I see") is a listener's acknowledgement. It never starts a turn and
  is dropped the way a sub-threshold transcript is today.
- A transcript carrying a break-in phrase ("stop", "hold on", "actually")
  starts the turn at once, whatever the count.
- A transcript that carries a number ("44957", "3", "seven three nine")
  starts the turn at once: a caller reading an order number is never a
  back-channel, and a transcriber writes the digits as one token, which a
  word count reads as one word.
- Anything else is judged by the word count, kept across the short finals a
  transcriber cuts one sentence into ("Hi," then "StubHub support.").

Both sides are normalised the same way — lower-cased, hyphens to spaces,
punctuation stripped — so "Mm-hmm." and "mm hmm" are the same phrase.
"""

import re
import time

from loguru import logger

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy

_SEPARATORS = re.compile(r"[-_/]+")
_NOT_WORD = re.compile(r"[^\w\s]+")


_DIGIT_WORDS = frozenset(
    "zero oh one two three four five six seven eight nine".split()
)
# How long a short final still counts towards the next one's word count.
_CARRY_SECONDS = 3.0
# How long after a final carrying a number a lone digit word is still part of it.
_NUMBER_CONTEXT_SECONDS = 8.0


def carries_a_number(tokens: list[str]) -> bool:
    """Whether a transcript is a number being read out.

    Harkbell. On the Mirror Labs copy (2026-11-19, pipeline runs 6469, 6470,
    6480, 6481) an order number said in two breaths, or slowly while typing,
    lost every digit that landed on the agent's speech: the transcriber
    writes "four four nine five seven" as "44957." -- one token, under a
    three-word threshold -- and each slow digit as its own final ("3.", "9.").
    The agent read back and filed "6587" for 658744957 and "73916" for
    73920516. A token with a digit in it is a number; so is a transcript made
    of nothing but digit words. "one" on its own stays a word ("one second" is
    a break-in phrase of its own).
    """
    if any(any(ch.isdigit() for ch in token) for token in tokens):
        return True
    return len(tokens) >= 2 and all(token in _DIGIT_WORDS for token in tokens)


def normalize_transcript(text: str) -> str:
    """Lower-case, turn hyphens into spaces, strip punctuation, collapse spaces."""
    lowered = _SEPARATORS.sub(" ", text.lower())
    return " ".join(_NOT_WORD.sub("", lowered).split())


def normalize_phrases(phrases: list[str] | None) -> list[str]:
    """Normalise a configured phrase list, dropping blanks and duplicates."""
    seen: list[str] = []
    for phrase in phrases or []:
        normalized = normalize_transcript(phrase) if isinstance(phrase, str) else ""
        if normalized and normalized not in seen:
            seen.append(normalized)
    return seen


class PhraseList:
    """A normalised phrase list that can be matched against transcript tokens."""

    def __init__(self, phrases: list[str] | None):
        self.phrases = normalize_phrases(phrases)
        self._tokens = [tuple(phrase.split()) for phrase in self.phrases]

    def __bool__(self) -> bool:
        return bool(self._tokens)

    def covers(self, tokens: list[str]) -> bool:
        """Whether the whole token sequence is a run of listed phrases."""
        if not tokens or not self._tokens:
            return False
        reachable = [True] + [False] * len(tokens)
        for end in range(1, len(tokens) + 1):
            for phrase in self._tokens:
                start = end - len(phrase)
                if (
                    start >= 0
                    and reachable[start]
                    and tuple(tokens[start:end]) == phrase
                ):
                    reachable[end] = True
                    break
        return reachable[len(tokens)]

    def appears_in(self, tokens: list[str]) -> bool:
        """Whether any listed phrase occurs as a contiguous run of tokens."""
        for phrase in self._tokens:
            width = len(phrase)
            for start in range(len(tokens) - width + 1):
                if tuple(tokens[start : start + width]) == phrase:
                    return True
        return False


class PhraseAwareUserTurnStartStrategy(MinWordsUserTurnStartStrategy):
    """Minimum-words turn start with ignore and break-in phrase lists."""

    def __init__(
        self,
        *,
        min_words: int,
        ignore_phrases: list[str] | None = None,
        break_in_phrases: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(min_words=min_words, **kwargs)
        self._ignore = PhraseList(ignore_phrases)
        self._break_in = PhraseList(break_in_phrases)
        # Words of the finals that fell short while the bot was speaking, and
        # when the last of them arrived (see _handle_transcription).
        self._carried_words = 0
        self._carried_at = 0.0
        # When a final last carried a number, bot speaking or not.
        self._number_heard_at: float | None = None

    async def handle_user_turn_started(self):
        self._carried_words = 0
        await super().handle_user_turn_started()

    async def _handle_transcription(
        self, frame: TranscriptionFrame | InterimTranscriptionFrame
    ) -> ProcessFrameResult:
        tokens = normalize_transcript(frame.text).split()
        is_interim = isinstance(frame, InterimTranscriptionFrame)
        if not is_interim and carries_a_number(tokens):
            self._number_heard_at = time.monotonic()
        if not self._bot_speaking:
            return await super()._handle_transcription(frame)

        if self._ignore.covers(tokens):
            logger.debug(
                f"{self} ignoring back-channel {frame.text!r} while bot speaking "
                f"interim_transcription={is_interim}"
            )
            await self.trigger_reset_aggregation()
            return ProcessFrameResult.CONTINUE
        if self._break_in.appears_in(tokens):
            logger.debug(
                f"{self} break-in phrase in {frame.text!r} while bot speaking "
                f"interim_transcription={is_interim}"
            )
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP
        if carries_a_number(tokens) or self._continues_a_number(tokens):
            logger.debug(
                f"{self} number in {frame.text!r} while bot speaking "
                f"interim_transcription={is_interim}"
            )
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP
        # A transcriber cuts one sentence into short finals, and each was
        # counted alone: "Hi," (one word) and "StubHub support." (two) both
        # fell under a threshold of three, so a caller's first sentence, said
        # as the greeting ended, was dropped whole (copy runs 6472, 6483) and
        # the line sat silent until the idle prompt. Finals that arrive within
        # a few seconds of each other are one sentence: their words add up.
        # Interims are left out -- each repeats the one before it.
        if not is_interim and tokens:
            now = time.monotonic()
            if now - self._carried_at > _CARRY_SECONDS:
                self._carried_words = 0
            self._carried_at = now
            if self._carried_words + len(tokens) >= self._min_words > len(tokens):
                logger.debug(
                    f"{self} {frame.text!r} completes {self._carried_words} carried "
                    f"words while bot speaking"
                )
                self._carried_words = 0
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
            self._carried_words += len(tokens)
        return await super()._handle_transcription(frame)


    def _continues_a_number(self, tokens: list[str]) -> bool:
        """A lone digit word said while a number is being read out.

        Copy run 7042 (2026-11-21, AssemblyAI): the caller read 5512084 a
        digit at a time and the agent read each part back. The transcriber
        wrote "5," "2," "8," as digits but "one" and "zero" as words, each
        said over a read-back, and a single word is under the three-word
        threshold -- both were dropped, twice, and the agent heard 5528. On
        its own "one" stays a word; a few seconds after a digit it is a digit.
        """
        if len(tokens) != 1 or tokens[0] not in _DIGIT_WORDS:
            return False
        if self._number_heard_at is None:
            return False
        return time.monotonic() - self._number_heard_at <= _NUMBER_CONTEXT_SECONDS


def create_turn_start_strategy(
    *,
    min_words: int,
    ignore_phrases: list[str] | None = None,
    break_in_phrases: list[str] | None = None,
    **kwargs,
) -> MinWordsUserTurnStartStrategy:
    """The plain minimum-words strategy, or the phrase-aware one when a list is set.

    With both lists empty the plain class is returned, so a workflow that has
    never configured a phrase runs exactly the code it ran before.
    """
    ignore = normalize_phrases(ignore_phrases)
    break_in = normalize_phrases(break_in_phrases)
    if not ignore and not break_in:
        return MinWordsUserTurnStartStrategy(min_words=min_words, **kwargs)
    return PhraseAwareUserTurnStartStrategy(
        min_words=min_words,
        ignore_phrases=ignore,
        break_in_phrases=break_in,
        **kwargs,
    )


def resolve_turn_start_phrases(run_configs: dict) -> tuple[list[str], list[str]]:
    """The configured (ignore, break-in) lists, normalised; empty when unset."""
    ignore = run_configs.get("turn_start_ignore_phrases")
    break_in = run_configs.get("turn_start_break_in_phrases")
    return (
        normalize_phrases(ignore if isinstance(ignore, list) else None),
        normalize_phrases(break_in if isinstance(break_in, list) else None),
    )
