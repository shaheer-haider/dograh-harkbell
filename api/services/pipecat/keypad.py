"""A caller's keypad presses, delivered to the agent as a short user turn.

Telephony serializers already turn each keypress into an ``InputDTMFFrame``,
but nothing in the call pipeline consumed it: a caller who pressed 1 was
ignored and the agent never learned it happened. Pipecat's ``DTMFAggregator``
turns presses into a ``TranscriptionFrame``, so the agent "hears" the digits as
``[keypad] 1``.

A transcript alone is not a turn, though. Every interruption policy the engine
installs judges transcripts by their words, and a keypress has two: a
``turn_start_min_words`` of three, the greeting's own threshold or a phrase
list would drop it as too short while the bot speaks, and a transcript with no
voice activity around it would wait out the turn analyzer's STT timeout before
it finished. So two turn strategies, placed ahead of the configured ones, make
a keypad transcript a complete user turn by itself: it starts a turn (which
interrupts the bot through the user aggregator, under the engine's own
interruption policy) and ends it at once.

Transports that carry no keypad (web calls) never produce an ``InputDTMFFrame``,
so on them all of this is inert.
"""

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputDTMFFrame,
    LLMMessagesAppendFrame,
    TranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
)
from pipecat.processors.aggregators.dtmf_aggregator import DTMFAggregator
from pipecat.processors.frame_processor import FrameDirection
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import BaseUserTurnStartStrategy
from pipecat.turns.user_stop import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

# What the agent reads for a keypress: "[keypad] 1", "[keypad] 12#". Workflow
# prompts match on this text, so it is a contract and must not change casually.
# Brackets are something a speech recogniser never writes, so a spoken
# "keypad one" can never be mistaken for a press.
KEYPAD_PREFIX = "[keypad] "

# How long to wait after a press for another before handing the digits over.
# A menu choice is one digit, and Pipecat's default two seconds is dead air on
# the line before the agent can react; '#' still ends an entry at once.
KEYPAD_IDLE_TIMEOUT_SECS = 1.0


def is_keypad_transcript(frame: Frame) -> bool:
    """Whether ``frame`` is a keypad entry rather than recognised speech."""
    return isinstance(frame, TranscriptionFrame) and frame.text.startswith(
        KEYPAD_PREFIX
    )


class KeypadAggregator(DTMFAggregator):
    """``DTMFAggregator`` with the engine's prefix and timeout, logging, and mute.

    Two departures from the stock aggregator:

    - It does not broadcast an interruption on the first digit. That one is
      unconditional, so it would cut speech the engine made uninterruptible
      (a transition notice, a node with interruptions off) and bypass the
      engine's own ``should_interrupt``. The keypad turn interrupts instead,
      through the user aggregator, which honours both.
    - While the caller's input is muted the user aggregator drops every
      transcript, so an entry finished then would be lost. It is held, and
      delivered as soon as the mute lifts.
    """

    def __init__(self, **kwargs):
        super().__init__(
            timeout=KEYPAD_IDLE_TIMEOUT_SECS, prefix=KEYPAD_PREFIX, **kwargs
        )
        self._user_muted = False
        self._held = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # The user aggregator broadcasts its mute changes, so the upstream copy
        # passes through here.
        if isinstance(frame, UserMuteStartedFrame):
            self._user_muted = True
        elif isinstance(frame, UserMuteStoppedFrame):
            self._user_muted = False
        await super().process_frame(frame, direction)
        if isinstance(frame, UserMuteStoppedFrame) and self._held:
            held, self._held = self._held, ""
            logger.info(f"Keypad entry released as the mute lifted: {held!r}")
            in_progress = self._aggregation
            self._aggregation = held + in_progress
            # An entry still being typed carries the held digits out with it
            # when it finishes; otherwise they go now.
            if not in_progress:
                await self._flush_aggregation()

    async def _handle_dtmf_frame(self, frame: InputDTMFFrame):
        digit = frame.button.value
        logger.info(f"Keypad digit received: {digit!r}")
        self._aggregation += digit
        if frame.button == self._termination_digit:
            await self._flush_aggregation()
        else:
            self._digit_event.set()

    async def _flush_aggregation(self):
        if not self._aggregation:
            return
        if self._user_muted:
            self._held += self._aggregation
            self._aggregation = ""
            logger.info(f"Keypad entry held while the caller is muted: {self._held!r}")
            return
        logger.info(f"Keypad entry complete: {self._aggregation!r}")
        await super()._flush_aggregation()


class KeypadUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """Start a user turn on a keypad entry, whatever the word threshold.

    Returns STOP so the strategies after it never see the entry: a minimum-words
    strategy would otherwise reset the aggregation that holds it.
    """

    # The entry that started the turn, for ``bind_realtime_keypad``.
    last_entry: str | None = None

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if is_keypad_transcript(frame):
            self.last_entry = frame.text
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP
        return ProcessFrameResult.CONTINUE


class KeypadUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """End the user turn on a keypad entry: an entry is complete when flushed.

    If the caller is also speaking, the turn controller keeps the turn open and
    the entry is answered together with the speech when that ends.
    """

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if is_keypad_transcript(frame):
            await self.trigger_user_turn_stopped()
            return ProcessFrameResult.STOP
        return ProcessFrameResult.CONTINUE


def with_keypad_turns(strategies: UserTurnStrategies) -> UserTurnStrategies:
    """``strategies`` with the keypad strategies in front of the configured ones."""
    return UserTurnStrategies(
        start=[KeypadUserTurnStartStrategy(), *(strategies.start or [])],
        stop=[KeypadUserTurnStopStrategy(), *(strategies.stop or [])],
    )


def keypad_start_strategies(strategies: UserTurnStrategies | None) -> list:
    """The keypad start strategies in ``strategies``, for a temporary policy to keep."""
    if strategies is None:
        return []
    return [
        s for s in strategies.start or [] if isinstance(s, KeypadUserTurnStartStrategy)
    ]


def bind_realtime_keypad(user_aggregator) -> None:
    """Hand a keypad entry to a realtime model as a user turn it answers.

    Harkbell. A speech-to-speech model hears the caller's audio itself, and a
    keypress is not audio it is given: the telephony serializer takes the tone
    out and sends an ``InputDTMFFrame`` instead. In a cascade call the keypad
    transcript ends a user turn and the aggregator sends the context to the
    model. In realtime mode the aggregator sends nothing at a turn's end -- the
    provider answers its own audio -- so the press was recorded and never
    answered: a caller who pressed 1 for urgent help heard nothing change.

    The entry is sent the way the idle check is: a one-off user message with a
    request to answer it. From the turn-started event, because by then the
    aggregator has broadcast the interruption the keypad turn causes, and a
    frame pushed after it reaches the service after it -- the other order would
    cancel the answer to the key.
    """

    async def answer_keypad_entry(aggregator, strategy) -> None:
        if not isinstance(strategy, KeypadUserTurnStartStrategy):
            return
        entry, strategy.last_entry = strategy.last_entry, None
        if not entry:
            return
        logger.info(f"Keypad entry sent to the realtime model: {entry!r}")
        await aggregator.push_frame(
            LLMMessagesAppendFrame([{"role": "user", "content": entry}], run_llm=True)
        )

    user_aggregator.add_event_handler("on_user_turn_started", answer_keypad_entry)
