import asyncio
import time

import pytest
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
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


class _Analyzer(BaseTurnAnalyzer):
    """Says the turn is complete whenever the caller stops."""

    @property
    def speech_triggered(self) -> bool:
        return False

    @property
    def params(self):
        return None

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        return EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self):
        return EndOfTurnState.COMPLETE, None

    def clear(self):
        pass


async def _strategy(hold_secs: float):
    strategy = NumberHoldTurnAnalyzerUserTurnStopStrategy(
        turn_analyzer=_Analyzer(), hold_secs=hold_secs
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

    return strategy, stopped


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
