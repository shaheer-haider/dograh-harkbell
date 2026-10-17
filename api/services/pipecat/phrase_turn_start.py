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
- Anything else is judged by the word count.

Both sides are normalised the same way — lower-cased, hyphens to spaces,
punctuation stripped — so "Mm-hmm." and "mm hmm" are the same phrase.
"""

import re

from loguru import logger

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy

_SEPARATORS = re.compile(r"[-_/]+")
_NOT_WORD = re.compile(r"[^\w\s]+")


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

    async def _handle_transcription(
        self, frame: TranscriptionFrame | InterimTranscriptionFrame
    ) -> ProcessFrameResult:
        if not self._bot_speaking:
            return await super()._handle_transcription(frame)

        tokens = normalize_transcript(frame.text).split()
        is_interim = isinstance(frame, InterimTranscriptionFrame)
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
        return await super()._handle_transcription(frame)


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
