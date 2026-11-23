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

Since ``-harkbell.38``:

- An order code with letters in it is held the same way until it has as many
  characters as a whole number: "VS-441-B as in Bravo" is seven, and the "nine
  two" a caller adds after a breath belongs to it (run 6573 read back and filed
  VS441B for VS441B92). Letters count when the transcriber writes them as a
  code -- a capital on its own, "B as in Bravo", "VS", "441B" -- and a turn
  ending on "the NFL" or "row F" is not held: it has no digit and no spelling.
- The strategy says when the transcriber commits (``transcript_commit.py``): a
  hold at a pause the analyzer judges unfinished or that ends on part of a
  number, a commit when the turn is over. The turn then ends on that commit's
  own final, never on an earlier one. The number check reads the interim
  transcript too, since nothing is final until the commit.

Since ``-harkbell.39`` a turn always ends (copy run 6660: "Hi, Vivid Seats
support." went unanswered for 30 s after the event loop stalled for 1.2 s
mid-sentence):

- A VAD stop that comes when no audio has arrived for ``AUDIO_GAP_SECS`` is
  the audio stopping, not the caller: the VAD controller forces one after a
  second without frames. It is not judged as a pause. The transcriber holds
  its commit and the turn stays open for the rest of the sentence.
- Every VAD stop arms a backstop: if nothing has ended the turn after the turn
  analyzer's own silence window and ``SILENCE_BACKSTOP_MARGIN_SECS``, with the
  caller still silent, the strategy ends it. The analyzer's own silence timer
  is armed only by audio it has been told is speech, and the burst of audio a
  stall releases can reach it after VAD has already called that audio over.
- A final that arrives while the caller is speaking again answers a commit
  from before they did, so it does not stand for this pause's commit.

Since ``-harkbell.44`` it runs for every transcriber the engine ends turns for,
not only the manually committed one (``steer_commits=False`` there). Copy run
7042 (2026-11-21, AssemblyAI): the caller read 5512084 a digit at a time and
each digit was a turn of its own, answered with a read-back the next digits
landed on. The VAD heard none of those digits, so each turn ended on Pipecat's
fallback -- a transcript with no VAD stop is a finished turn 0.42 s later --
which never asked whether the words were part of a number. Without steering:

- Words that arrive while the VAD is silent, and not as the final of speech it
  just heard stop, are speech it missed. The hold restarts from them: the
  caller is still reading, and the silence is counted from their last word.
- An interim with new words while a hold is pending is the caller still
  speaking: the hold starts again in full.

Since ``-harkbell.46`` the transcriber's words can end a turn the analyzer
judges unfinished (without steering). Copy runs 7246-7251 (2026-11-23): on a
line where the VAD heard most of the caller, Smart Turn called a finished
sentence unfinished on most turns -- "Hi, this is Shahid. From StubHub
support, I am calling about an order." -- and the turn waited out the
analyzer's two seconds: replies came 2.8-3.9 s after the caller's last word,
where they had come in 1.2-1.4 s the day before on a line the VAD barely
heard. Now, while the analyzer holds a turn open and the caller is silent,
the words are timed the way the manually committed transcriber times them
(``words_quiet_seconds``): 0.7 s after a sentence that has ended, 2.0 s after a
dash, comma, filler, joining word or part-read number, or words with no full
stop after them. Words still changing
restart it; the caller speaking again cancels it.
"""

import asyncio
import re
import time

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)

from api.services.pipecat.transcript_commit import (
    TranscriptCommitHoldFrame,
    TranscriptCommitRequestFrame,
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
# How long the final of a requested commit is waited for before the turn ends
# on what it has. A commit with no speech in it has no final at all.
COMMIT_FINAL_WAIT_SECS = 1.0
# A VAD stop this long after the last audio frame is the audio stopping, not
# the caller. Frames arrive every 20 ms; the VAD controller forces a stop after
# a whole second without one.
AUDIO_GAP_SECS = 0.5
# How long past the turn analyzer's own silence window a silent caller waits
# before the strategy ends the turn itself.
SILENCE_BACKSTOP_MARGIN_SECS = 0.25
# The analyzer's silence window when it does not say (pipecat's Smart Turn default).
DEFAULT_ANALYZER_STOP_SECS = 3.0
# A final this soon after the speech end the VAD reported is that speech's own
# final; one later than this is words the VAD never heard.
OWN_FINAL_SECS = 1.0


def _tokens(text: str, *, lower: bool = True) -> list[str]:
    # "658-744" and "6,587" are digits; "8:00" keeps its colon, so it reads as a time.
    text = re.sub(r"(?<=\d)[,-](?=\d)", "", text.lower() if lower else text)
    text = text.replace("-", " ")
    return [t for t in (_EDGE_PUNCTUATION.sub("", raw) for raw in text.split()) if t]


def _spelled_letter(token: str) -> bool:
    """A letter on its own, as a transcriber writes one being spelled: "V", "B"."""
    return len(token) == 1 and token.isalpha() and token.isupper() and token not in ("I", "A")


def unfinished_code(text: str) -> bool | None:
    """Whether a turn ends on an order code with letters in it, too short to be whole.

    None when the run at the end has no letters: that is a number, and
    ``unfinished_number`` judges it by its own rules.

    Read from the end: spelled letters ("V S", "B as in Bravo"), capitals of
    two or three ("VS"), letters and digits in one token ("441B"), digits and
    digit words. The code is whole at ``WHOLE_NUMBER_DIGITS`` characters. A run
    with no letters is left to ``unfinished_number``'s own rules; a run with
    letters but no digit is held only when it is spelled, so "the NFL" and "row
    F" end the turn.
    """
    tokens = _tokens(text, lower=False)
    digits = letters = single_letters = 0
    spelled = False
    index = len(tokens) - 1
    while index >= 0:
        token = tokens[index]
        low = token.lower()
        if (
            index >= 3
            and token.isalpha()
            and tokens[index - 1].lower() == "in"
            and tokens[index - 2].lower() == "as"
            and len(tokens[index - 3]) == 1
            and tokens[index - 3].isalpha()
        ):
            # "B as in Bravo": one letter, said the way a code is spelled.
            letters += 1
            single_letters += 1
            spelled = True
            index -= 4
            continue
        if low.isdigit():
            digits += len(low)
        elif low in _DIGIT_WORDS:
            digits += 1
        elif _spelled_letter(token):
            letters += 1
            single_letters += 1
        elif token.isalnum() and any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
            digits += sum(c.isdigit() for c in token)
            letters += sum(c.isalpha() for c in token)
        elif token.isalpha() and token.isupper() and 2 <= len(token) <= 3:
            letters += len(token)
        else:
            break
        index -= 1
    if letters == 0:
        return None
    if digits + letters >= WHOLE_NUMBER_DIGITS:
        return False
    return digits > 0 or spelled or single_letters >= 2


def unfinished_number(text: str) -> bool:
    """Whether a turn ends on a run of digits, or an order code, too short to be whole."""
    code = unfinished_code(text)
    if code is not None:
        return code
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
    """The turn analyzer's stop strategy, holding the turn open on a part-read number.

    With ``steer_commits`` (the default) it also tells a manually committed
    transcriber when to commit, and ends the turn on that commit's final.
    """

    def __init__(
        self,
        *,
        hold_secs: float = NUMBER_HOLD_SECS,
        steer_commits: bool = True,
        final_wait_secs: float = COMMIT_FINAL_WAIT_SECS,
        silence_backstop_secs: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._hold_secs = hold_secs
        self._steer_commits = steer_commits
        self._final_wait_secs = final_wait_secs
        self._silence_backstop_secs = silence_backstop_secs
        self._turn_text = ""
        self._interim_text = ""
        self._speech_ended_at: float | None = None
        # When the speech the VAD last reported stopping ended (wall clock).
        self._vad_ended_at: float | None = None
        self._hold_task: asyncio.Task | None = None
        self._hold_expired = False
        # This pause's commit has been asked for, and its final not yet seen.
        self._commit_requested = False
        self._awaiting_final = False
        self._final_wait_task: asyncio.Task | None = None
        # When the last audio frame reached the strategy (monotonic).
        self._last_audio_at: float | None = None
        self._backstop_task: asyncio.Task | None = None
        # Ends a turn the analyzer holds open once the words have stopped.
        self._quiet_task: asyncio.Task | None = None

    @property
    def steers_commits(self) -> bool:
        """Whether this strategy tells the transcriber when to commit."""
        return self._steer_commits

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, InputAudioRawFrame):
            self._last_audio_at = time.monotonic()
        elif isinstance(frame, InterimTranscriptionFrame) and frame.text and frame.text.strip():
            text = frame.text.strip()
            if (
                not self._steer_commits
                and not self._vad_user_speaking
                and self._hold_task is not None
                and text != self._interim_text
            ):
                # The caller is still reading the number the hold is for.
                await self._cancel_task("_hold_task")
                self._speech_ended_at = time.time()
                self._hold_task = self.task_manager.create_task(
                    self._hold(self._hold_secs), f"{self}::number_hold"
                )
            changed = text != self._interim_text
            self._interim_text = text
            result = await super().process_frame(frame)
            if changed:
                await self._arm_words_quiet()
            return result
        return await super().process_frame(frame)

    def _words_the_vad_missed(self) -> bool:
        if self._steer_commits or self._vad_user_speaking:
            return False
        if self._vad_ended_at is None:
            return True
        return time.time() - self._vad_ended_at > OWN_FINAL_SECS

    def _backstop_seconds(self) -> float:
        if self._silence_backstop_secs is not None:
            return self._silence_backstop_secs
        params = getattr(self._turn_analyzer, "params", None)
        stop_secs = getattr(params, "stop_secs", None) or DEFAULT_ANALYZER_STOP_SECS
        return stop_secs + SILENCE_BACKSTOP_MARGIN_SECS

    def _audio_gap(self) -> float | None:
        """How long the audio has been missing, when that is why VAD stopped."""
        if self._last_audio_at is None:
            return None
        missing = time.monotonic() - self._last_audio_at
        return missing if missing >= AUDIO_GAP_SECS else None

    async def _reset(self):
        self._turn_text = ""
        self._interim_text = ""
        await super()._reset()

    async def _discard_pending_end_of_turn(self):
        self._hold_expired = False
        self._speech_ended_at = None
        self._vad_ended_at = None
        self._commit_requested = False
        self._awaiting_final = False
        await self._cancel_task("_hold_task")
        await self._cancel_task("_final_wait_task")
        await self._cancel_task("_backstop_task")
        await self._cancel_task("_quiet_task")
        await super()._discard_pending_end_of_turn()

    async def cleanup(self):
        await self._cancel_task("_hold_task")
        await self._cancel_task("_final_wait_task")
        await self._cancel_task("_backstop_task")
        await self._cancel_task("_quiet_task")
        await super().cleanup()

    async def _cancel_task(self, name: str):
        task = getattr(self, name)
        if task:
            setattr(self, name, None)
            await self.task_manager.cancel_task(task)

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        # Set first: the parent ends by asking whether the turn may stop.
        self._speech_ended_at = frame.timestamp - frame.stop_secs
        self._vad_ended_at = self._speech_ended_at
        gap = self._audio_gap()
        if gap is not None:
            # The audio stopped, not the caller: judging the pause would end
            # the turn mid-sentence on whatever came before the gap.
            logger.info(
                f"{self}: no audio for {gap:.2f}s while the caller was speaking; "
                "keeping the turn open for the rest of it"
            )
            self._vad_user_speaking = False
            self._vad_stopped = True
            self._turn_complete = False
        else:
            await super()._handle_vad_user_stopped_speaking(frame)
        if self._steer_commits and not self._turn_complete:
            # The analyzer heard more coming: the transcriber keeps listening.
            await self.push_frame(TranscriptCommitHoldFrame(), FrameDirection.UPSTREAM)
        await self._cancel_task("_backstop_task")
        seconds = self._backstop_seconds()
        self._backstop_task = self.task_manager.create_task(
            self._silence_backstop(seconds), f"{self}::silence_backstop"
        )
        await self._arm_words_quiet()

    async def _silence_backstop(self, seconds: float):
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        finally:
            self._backstop_task = None
        if self._vad_user_speaking:
            return
        if not self._turn_complete:
            logger.info(
                f"{self}: {seconds:.2f}s of silence with no end of turn from the analyzer; "
                "ending the turn"
            )
            self._turn_complete = True
        await self._maybe_trigger_user_turn_stopped()

    async def _handle_transcription(self, frame: TranscriptionFrame):
        missed = bool(frame.text and frame.text.strip()) and self._words_the_vad_missed()
        if missed:
            # Speech the VAD never heard: its silence starts now, and an
            # earlier hold or timeout no longer speaks for the turn.
            await self._cancel_task("_hold_task")
            self._hold_expired = False
            self._timeout_expired = False
            self._speech_ended_at = time.time()
        if frame.text and frame.text.strip():
            self._turn_text = f"{self._turn_text} {frame.text.strip()}".strip()
        self._interim_text = ""
        # A final nobody asked for is the transcriber committing on its own
        # (no verdict reached it in time): that is this pause's commit. One
        # that arrives while the caller is speaking again answers a commit
        # from before they did, and this pause's is still to come.
        if not self._vad_user_speaking:
            self._commit_requested = True
        if self._awaiting_final:
            self._awaiting_final = False
            await self._cancel_task("_final_wait_task")
        await super()._handle_transcription(frame)
        if missed and self._turn_complete and self._timeout_task is None:
            # Pipecat's fallback arms its own timer only before any VAD stop
            # this turn; after one, nothing else would judge these words.
            await self._maybe_trigger_user_turn_stopped()
        await self._arm_words_quiet()

    async def _arm_words_quiet(self):
        """Time the words while the analyzer holds the turn open (since .46)."""
        await self._cancel_task("_quiet_task")
        if self._steer_commits or self._vad_user_speaking or self._turn_complete:
            return
        heard = self._heard()
        if not heard:
            return
        from api.services.pipecat.elevenlabs_manual_stt import (
            WORDS_UNFINISHED_QUIET_SECONDS,
            words_quiet_seconds,
        )

        seconds = words_quiet_seconds(heard)
        if heard.rstrip()[-1:] not in ".?!":
            # A transcriber that punctuates writes no full stop after words
            # that trail off ("Uh, the customer"): the analyzer may be right.
            seconds = max(seconds, WORDS_UNFINISHED_QUIET_SECONDS)
        self._quiet_task = self.task_manager.create_task(
            self._words_quiet(seconds), f"{self}::words_quiet"
        )

    async def _words_quiet(self, seconds: float):
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        finally:
            self._quiet_task = None
        if self._vad_user_speaking or self._turn_complete:
            return
        logger.info(
            f"{self}: the words stopped {seconds:.1f}s ago and the analyzer still "
            f"holds the turn; ending it: {self._heard()[-60:]!r}"
        )
        self._turn_complete = True
        await self._maybe_trigger_user_turn_stopped()

    def _heard(self) -> str:
        return f"{self._turn_text} {self._interim_text}".strip()

    async def _maybe_trigger_user_turn_stopped(self):
        if self._turn_complete and not self._hold_expired and unfinished_number(self._heard()):
            ended = self._speech_ended_at if self._speech_ended_at is not None else time.time()
            remaining = ended + self._hold_secs - time.time()
            if remaining > 0:
                if self._hold_task is None:
                    logger.debug(
                        f"{self}: holding the turn {remaining:.2f}s, it ends on part of a number: "
                        f"{self._heard()[-40:]!r}"
                    )
                    self._hold_task = self.task_manager.create_task(
                        self._hold(remaining), f"{self}::number_hold"
                    )
                    if self._steer_commits:
                        await self.push_frame(TranscriptCommitHoldFrame(), FrameDirection.UPSTREAM)
                return
        if self._steer_commits and self._turn_complete:
            if not self._commit_requested:
                # The turn is over: the transcriber commits it, and the turn
                # ends on that final rather than on one from earlier in it.
                self._commit_requested = True
                self._awaiting_final = True
                await self.push_frame(TranscriptCommitRequestFrame(), FrameDirection.UPSTREAM)
                self._final_wait_task = self.task_manager.create_task(
                    self._wait_for_final(), f"{self}::commit_final"
                )
            if self._awaiting_final:
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

    async def _wait_for_final(self):
        try:
            await asyncio.sleep(self._final_wait_secs)
        except asyncio.CancelledError:
            return
        finally:
            self._final_wait_task = None
        # A commit with no speech in it has no final: end on what there is.
        self._awaiting_final = False
        await self._maybe_trigger_user_turn_stopped()
