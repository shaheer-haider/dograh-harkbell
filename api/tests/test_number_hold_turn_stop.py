import asyncio
import time

import pytest
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager

from api.services.pipecat.number_hold_turn_stop import (
    NumberHoldTurnAnalyzerUserTurnStopStrategy,
    unfinished_number,
)
from api.services.pipecat.transcript_commit import (
    TranscriptCommitHoldFrame,
    TranscriptCommitRequestFrame,
)


@pytest.mark.parametrize(
    "text",
    [
        "The order number is seven",  # copy run 6537: one slow digit
        "The order number is seven. Three.",
        "Seven.",
        "The order number is six five eight seven,",  # the first of two breaths
        "It's 658-744",
        "Section 104",
    ],
)
def test_a_turn_ending_on_part_of_a_number_is_held(text):
    assert unfinished_number(text)


@pytest.mark.parametrize(
    "text",
    [
        "The order number is seven three nine two zero five one six.",
        "The first is 658-744-957, and the second is 658-744-958.",
        "Order 73920516",
        "The event is tonight at eight.",
        "Both are for tonight at 8:00.",
        "It's at 10",
        "Yes, that one.",
        "I have two",
        "Oh.",
        "Yes, correct.",
        "The tickets were never delivered.",
        "",
    ],
)
def test_a_whole_number_a_time_or_no_number_is_not_held(text):
    assert not unfinished_number(text)


# -harkbell.38: run 6573 read back VS441B for VS441B92, the "nine two" said
# after a breath.
@pytest.mark.parametrize(
    "text",
    [
        "The order is VS-441-B as in Bravo.",
        "The order is V S 4 4 1",
        "It's V S",
        "B as in Bravo",
        "Order VS441B",
        "The code is 4B",
    ],
)
def test_a_turn_ending_on_part_of_a_code_with_letters_is_held(text):
    assert unfinished_number(text)


@pytest.mark.parametrize(
    "text",
    [
        "The order is VS-441-B as in Bravo. Nine two.",
        "The order is V S 441 B 92",
        "Order VS441B92",
        "It's for the NFL",
        "I'm calling from the USA",
        "Row F",
        "Section B",
        "My order is A",
    ],
)
def test_a_whole_code_or_a_word_in_capitals_is_not_held(text):
    assert not unfinished_number(text)


class _Analyzer(BaseTurnAnalyzer):
    """Says the turn is complete whenever the caller stops (or, told so, that it is not)."""

    def __init__(self, state: EndOfTurnState = EndOfTurnState.COMPLETE):
        super().__init__()
        self.state = state

    @property
    def speech_triggered(self) -> bool:
        return False

    @property
    def params(self):
        return None

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        return EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self):
        return self.state, None

    def clear(self):
        pass


async def _strategy(hold_secs: float, analyzer: _Analyzer | None = None, **kwargs):
    strategy = NumberHoldTurnAnalyzerUserTurnStopStrategy(
        turn_analyzer=analyzer or _Analyzer(), hold_secs=hold_secs, **kwargs
    )
    task_manager = TaskManager()
    await strategy.setup(
        FrameProcessorSetup(
            clock=SystemClock(), task_manager=task_manager, pipeline_worker=None
        )
    )
    stopped: list[float] = []

    @strategy.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(strategy, params):
        stopped.append(time.time())

    pushed: list = []

    @strategy.event_handler("on_push_frame")
    async def on_push_frame(strategy, frame, direction):
        pushed.append(type(frame).__name__)

    strategy.pushed = pushed
    return strategy, stopped


def _interim(text: str) -> InterimTranscriptionFrame:
    return InterimTranscriptionFrame(text=text, user_id="", timestamp="")


def _final(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="", finalized=True)


@pytest.mark.asyncio
async def test_a_finished_sentence_ends_the_turn_at_once():
    strategy, stopped = await _strategy(hold_secs=0.4)
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await strategy.process_frame(_final("The tickets were never delivered."))
    assert len(stopped) == 1
    await strategy.cleanup()


@pytest.mark.asyncio
async def test_part_of_a_number_waits_for_the_hold_then_ends():
    strategy, stopped = await _strategy(hold_secs=0.4)
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await strategy.process_frame(_final("The order number is seven"))
    assert stopped == []
    await asyncio.sleep(0.4)
    assert len(stopped) == 1
    await strategy.cleanup()


@pytest.mark.asyncio
async def test_the_next_digit_cancels_the_hold_and_the_whole_number_ends_the_turn():
    strategy, stopped = await _strategy(hold_secs=0.4)
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await strategy.process_frame(_final("The order number is seven three nine"))
    await asyncio.sleep(0.1)
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await asyncio.sleep(0.4)
    assert stopped == []
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await strategy.process_frame(_final("two zero five one six."))
    assert len(stopped) == 1
    await strategy.cleanup()


# -harkbell.38: the strategy says when the transcriber commits. Run 6583:
# "It's in two days. No, sorry, it's in two hours." was committed at every
# pause and reached the agent as "... Sorry, i / In two hours."


@pytest.mark.asyncio
async def test_a_pause_the_analyzer_calls_unfinished_holds_the_commit():
    analyzer = _Analyzer(EndOfTurnState.INCOMPLETE)
    strategy, stopped = await _strategy(hold_secs=0.4, analyzer=analyzer)
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(_interim("It's in two days. No, sorry,"))
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    assert strategy.pushed == ["TranscriptCommitHoldFrame"]
    assert stopped == []
    await strategy.cleanup()


@pytest.mark.asyncio
async def test_a_finished_turn_asks_for_the_commit_and_ends_on_its_final():
    strategy, stopped = await _strategy(hold_secs=0.4)
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(_interim("It's in two days. No, sorry, it's in two hours"))
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    assert strategy.pushed == ["TranscriptCommitRequestFrame"]
    assert stopped == []
    await strategy.process_frame(_final("It's in two days. No, sorry, it's in two hours."))
    assert len(stopped) == 1
    # Asked once per pause.
    assert strategy.pushed == ["TranscriptCommitRequestFrame"]
    await strategy.cleanup()


@pytest.mark.asyncio
async def test_an_earlier_final_never_ends_the_turn_before_the_commit_it_asked_for():
    strategy, stopped = await _strategy(hold_secs=0.4, final_wait_secs=5.0)
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(_final("Hello."))
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(_interim("The tickets never arrived"))
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await asyncio.sleep(0.05)
    assert strategy.pushed == ["TranscriptCommitRequestFrame"]
    assert stopped == []
    await strategy.process_frame(_final("The tickets never arrived."))
    assert len(stopped) == 1
    await strategy.cleanup()


@pytest.mark.asyncio
async def test_a_commit_with_no_final_ends_the_turn_after_the_wait():
    strategy, stopped = await _strategy(hold_secs=0.4, final_wait_secs=0.2)
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(_final("Hello."))
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await asyncio.sleep(0.05)
    assert stopped == []
    await asyncio.sleep(0.3)
    assert len(stopped) == 1
    await strategy.cleanup()


@pytest.mark.asyncio
async def test_part_of_a_number_in_the_interim_holds_the_commit_until_the_hold_ends():
    strategy, stopped = await _strategy(hold_secs=0.4)
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(_interim("The order is VS-441-B as in Bravo"))
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    assert strategy.pushed == ["TranscriptCommitHoldFrame"]
    await asyncio.sleep(0.45)
    assert strategy.pushed == ["TranscriptCommitHoldFrame", "TranscriptCommitRequestFrame"]
    assert stopped == []
    await strategy.process_frame(_final("The order is VS-441-B as in Bravo."))
    assert len(stopped) == 1
    await strategy.cleanup()


@pytest.mark.asyncio
async def test_the_rest_of_the_code_said_within_the_hold_keeps_one_commit():
    strategy, stopped = await _strategy(hold_secs=0.4)
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(_interim("The order is VS-441-B as in Bravo"))
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await asyncio.sleep(0.1)
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(_interim("The order is VS-441-B as in Bravo, nine two"))
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    assert strategy.pushed == ["TranscriptCommitHoldFrame", "TranscriptCommitRequestFrame"]
    await strategy.process_frame(_final("The order is VS-441-B as in Bravo, 92."))
    assert len(stopped) == 1
    await strategy.cleanup()


@pytest.mark.asyncio
async def test_without_steering_it_asks_the_transcriber_for_nothing():
    strategy, stopped = await _strategy(hold_secs=0.4, steer_commits=False)
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await strategy.process_frame(_final("The tickets were never delivered."))
    assert len(stopped) == 1
    assert strategy.pushed == []
    await strategy.cleanup()

