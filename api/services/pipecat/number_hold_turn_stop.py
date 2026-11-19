"""Harkbell: never end a caller's turn in the middle of a number they are reading out.

A transcriber that commits on a short silence (``elevenlabs_manual_stt.py``,
0.2 s) hands the turn analyzer every pause, and a caller reading an order
number slowly pauses after every digit. The turn analyzer judges the audio,
and a digit said on a falling tone sounds finished: on the Mirror Labs copy
"seven", then "three", each became a turn of its own, the agent answered the
first ("Seven -- go on") and the digits said over it were lost (run 6537,
filed 7251 for 73920516).

So when the turn so far ends on a short run of digits -- fewer than an order
number has -- the turn is held until ``NUMBER_HOLD_SECS`` of silence after the
caller's last word, instead of ending at the analyzer's word. The caller
speaking again cancels the hold, as it cancels any pending end of turn. A
whole number ("seven three nine two zero five one six", "658744957") ends the
turn at once, and so does a small number after a word that makes it a time
("at eight", "by 10").
"""

import asyncio
import re
import time

from loguru import logger

from pipecat.frames.frames import TranscriptionFrame, VADUserStoppedSpeakingFrame
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)

# The silence a caller gets, after a short run of digits, before the turn ends.
# The provider's own 1.5 s pause held the copy's slowly read number in one piece.
NUMBER_HOLD_SECS = 1.5
# The shortest number that counts as complete: the copy's order numbers have 8
# to 12 digits, and a caller who has said that many has said a number.
WHOLE_NUMBER_DIGITS = 8

_DIGIT_WORDS = frozenset(
    "zero oh one two three four five six seven eight nine".split()
)
# Words after which one or two digits are a time or a count, not an order number.
_TIME_WORDS = frozenset(
    "at by around about before after until till from to past".split()
)
# A lone digit word is a number being read out only after one of these, or as
# the whole turn; "that one", "I have two" and "oh" are words.
_NUMBER_CUES = frozenset(
    "is number order code reference ref case confirmation id".split()
)
_EDGE_PUNCTUATION = re.compile(r"^[^\w:]+|[^\w:]+$")


def _tokens(text: str) -> list[str]:
    # "658-744" and "6,587" are digits; "8:00" keeps its colon, so it reads as a time.
    text = re.sub(r"(?<=\d)[,-](?=\d)", "", text.lower())
    text = text.replace("-", " ")
    return [t for t in (_EDGE_PUNCTUATION.sub("", raw) for raw in text.split()) if t]


def unfinished_number(text: str) -> bool:
    """Whether a turn ends on a run of digits too short to be a whole number."""
    tokens = _tokens(text)
    digits = 0
    first = len(tokens)
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index]
        if token.isdigit():
            digits += len(token)
        elif token in _DIGIT_WORDS:
            digits += 1
        else:
            break
        first = index
    if digits == 0 or digits >= WHOLE_NUMBER_DIGITS:
        return False
    before = tokens[first - 1] if first > 0 else ""
    if digits <= 2 and before in _TIME_WORDS:
        return False
    run = tokens[first:]
    if run == ["oh"]:
        return False
    if len(run) == 1 and run[0] in _DIGIT_WORDS and before and before not in _NUMBER_CUES:
        return False
    return True


class NumberHoldTurnAnalyzerUserTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """The turn analyzer's stop strategy, holding the turn open on a part-read number."""

    def __init__(self, *, hold_secs: float = NUMBER_HOLD_SECS, **kwargs):
        super().__init__(**kwargs)
        self._hold_secs = hold_secs
        self._turn_text = ""
        self._speech_ended_at: float | None = None
        self._hold_task: asyncio.Task | None = None
        self._hold_expired = False

    async def _reset(self):
        self._turn_text = ""
        await super()._reset()

    async def _discard_pending_end_of_turn(self):
        self._hold_expired = False
        self._speech_ended_at = None
        if self._hold_task:
            await self.task_manager.cancel_task(self._hold_task)
            self._hold_task = None
        await super()._discard_pending_end_of_turn()

    async def cleanup(self):
        if self._hold_task:
            await self.task_manager.cancel_task(self._hold_task)
            self._hold_task = None
        await super().cleanup()

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        # Set first: the parent ends by asking whether the turn may stop.
        self._speech_ended_at = frame.timestamp - frame.stop_secs
        await super()._handle_vad_user_stopped_speaking(frame)

    async def _handle_transcription(self, frame: TranscriptionFrame):
        if frame.text and frame.text.strip():
            self._turn_text = f"{self._turn_text} {frame.text.strip()}".strip()
        await super()._handle_transcription(frame)

    async def _maybe_trigger_user_turn_stopped(self):
        if self._turn_complete and not self._hold_expired and unfinished_number(self._turn_text):
            ended = self._speech_ended_at if self._speech_ended_at is not None else time.time()
            remaining = ended + self._hold_secs - time.time()
            if remaining > 0:
                if self._hold_task is None:
                    logger.debug(
                        f"{self}: holding the turn {remaining:.2f}s, it ends on part of a number: "
                        f"{self._turn_text[-40:]!r}"
                    )
                    self._hold_task = self.task_manager.create_task(
                        self._hold(remaining), f"{self}::number_hold"
                    )
                return
        await super()._maybe_trigger_user_turn_stopped()

    async def _hold(self, seconds: float):
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        finally:
            self._hold_task = None
        self._hold_expired = True
        await self._maybe_trigger_user_turn_stopped()
