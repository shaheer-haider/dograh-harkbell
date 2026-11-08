"""A caller's keypress reaches the model as a "[keypad] <digits>" user turn.

These run the real pieces the call pipeline uses — the keypad aggregator, the
LLM user aggregator, the engine's phrase-aware start strategy, the Smart Turn
stop strategy and the greeting controller — and watch for the context frame
that runs the model.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    EndFrame,
    Frame,
    InputDTMFFrame,
    InterruptionFrame,
    LLMContextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_mute import (
    CallbackUserMuteStrategy,
    FunctionCallUserMuteStrategy,
)
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import (
    ExternalUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from api.services.pipecat.greeting import GreetingController
from api.services.pipecat.keypad import (
    KEYPAD_IDLE_TIMEOUT_SECS,
    KEYPAD_PREFIX,
    KeypadAggregator,
    KeypadUserTurnStartStrategy,
    KeypadUserTurnStopStrategy,
    is_keypad_transcript,
    keypad_start_strategies,
    with_keypad_turns,
)
from api.services.pipecat.phrase_turn_start import create_turn_start_strategy
from api.services.pipecat.pipeline_builder import build_pipeline
from api.services.pipecat.worker_runner import run_pipeline_worker

# Harkbell's production turn settings (config 116 onwards).
TURN_START_MIN_WORDS = 3
IGNORE = ["uh-huh", "mm-hmm", "yeah", "okay", "right", "one", "1"]
BREAK_IN = ["stop", "hold on", "wait"]


def keypad_text(digits: str) -> str:
    return f"{KEYPAD_PREFIX}{digits}"


def configured_strategies(*, keypad: bool = True) -> UserTurnStrategies:
    """What run_pipeline builds for a min_words workflow with Smart Turn."""
    strategies = UserTurnStrategies(
        start=[
            create_turn_start_strategy(
                min_words=TURN_START_MIN_WORDS,
                ignore_phrases=IGNORE,
                break_in_phrases=BREAK_IN,
            )
        ],
        stop=[
            TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=LocalSmartTurnAnalyzerV3(
                    params=SmartTurnParams(stop_secs=2)
                )
            )
        ],
    )
    return with_keypad_turns(strategies) if keypad else strategies


class Sink(FrameProcessor):
    """Records what reaches the generation stage."""

    def __init__(self):
        super().__init__()
        self.contexts: asyncio.Queue[list[dict]] = asyncio.Queue()
        self.downstream: list[Frame] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            self.downstream.append(frame)
            if isinstance(frame, LLMContextFrame):
                await self.contexts.put([dict(m) for m in frame.context.get_messages()])
        await self.push_frame(frame, direction)


class Call:
    """Keypad aggregator → user aggregator → sink, as in the call pipeline."""

    def __init__(self, *, keypad_turns: bool = True, greeting: bool = False):
        self.muted = False
        self.context = LLMContext()
        self.keypad = KeypadAggregator()
        self.user = LLMUserAggregator(
            self.context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=configured_strategies(keypad=keypad_turns),
                user_mute_strategies=[
                    FunctionCallUserMuteStrategy(),
                    CallbackUserMuteStrategy(should_mute_callback=self._mute),
                ],
                should_interrupt=lambda: True,
                user_idle_timeout=0,
            ),
        )
        self.greeting = None
        if greeting:
            self.playback = SimpleNamespace(greeting_pending=True, greeting=None)
            self.greeting = GreetingController(
                self.playback, lambda: self.context, is_screening=lambda: False
            )
            # The greeting's own threshold above a keypress's two words.
            self.greeting.configure(
                min_words=TURN_START_MIN_WORDS,
                ignore_phrases=IGNORE,
                break_in_phrases=BREAK_IN,
            )
            self.greeting.bind(self.user)
        self.sink = Sink()
        self.worker = PipelineWorker(
            Pipeline([self.keypad, self.user, self.sink]),
            enable_rtvi=False,
            cancel_on_idle_timeout=False,
        )

    async def _mute(self, _frame) -> bool:
        return self.muted

    async def send(self, frame: Frame):
        await self.worker.queue_frame(frame)

    async def press(self, *buttons: KeypadEntry):
        for button in buttons:
            await self.send(InputDTMFFrame(button))

    async def next_context(self, timeout: float) -> list[dict]:
        return await asyncio.wait_for(self.sink.contexts.get(), timeout)

    def user_messages(self, messages: list[dict]) -> list[str]:
        return [m["content"] for m in messages if m.get("role") == "user"]


@pytest.fixture
async def call(request):
    harness = Call(**getattr(request, "param", {}))
    ready = asyncio.Event()

    @harness.worker.event_handler("on_pipeline_started")
    async def started(*_):
        ready.set()

    runner = asyncio.create_task(run_pipeline_worker(harness.worker))
    try:
        await asyncio.wait_for(ready.wait(), 5)
        yield harness
    finally:
        if not runner.done():
            await harness.worker.queue_frame(EndFrame())
        await asyncio.wait_for(runner, 5)


# --- the pieces ----------------------------------------------------------------


def test_keypad_transcripts_are_recognised_by_their_prefix():
    assert KEYPAD_PREFIX == "[keypad] "
    assert KEYPAD_IDLE_TIMEOUT_SECS == 1.0
    assert is_keypad_transcript(TranscriptionFrame("[keypad] 1", "", ""))
    assert not is_keypad_transcript(TranscriptionFrame("keypad one", "", ""))
    assert not is_keypad_transcript(Frame())


@pytest.mark.asyncio
async def test_strategies_own_a_keypad_entry_and_pass_speech_on():
    start, stop = KeypadUserTurnStartStrategy(), KeypadUserTurnStopStrategy()
    started, stopped = AsyncMock(), AsyncMock()
    start.add_event_handler("on_user_turn_started", started)
    stop.add_event_handler("on_user_turn_stopped", stopped)
    entry = TranscriptionFrame("[keypad] 1", "", "", finalized=True)
    assert await start.process_frame(entry) is ProcessFrameResult.STOP
    assert await stop.process_frame(entry) is ProcessFrameResult.STOP
    assert started.await_count == 1 and stopped.await_count == 1
    speech = TranscriptionFrame("hello there", "", "")
    assert await start.process_frame(speech) is ProcessFrameResult.CONTINUE
    assert await stop.process_frame(speech) is ProcessFrameResult.CONTINUE
    assert started.await_count == 1 and stopped.await_count == 1


def test_keypad_strategies_go_in_front_of_the_configured_ones():
    configured = UserTurnStrategies(
        start=[MinWordsUserTurnStartStrategy(min_words=3)],
        stop=[ExternalUserTurnStopStrategy()],
    )
    wrapped = with_keypad_turns(configured)
    assert isinstance(wrapped.start[0], KeypadUserTurnStartStrategy)
    assert wrapped.start[1:] == configured.start
    assert isinstance(wrapped.stop[0], KeypadUserTurnStopStrategy)
    assert wrapped.stop[1:] == configured.stop
    assert keypad_start_strategies(wrapped) == [wrapped.start[0]]
    assert keypad_start_strategies(None) == []


def test_the_call_pipeline_puts_the_keypad_aggregator_right_after_recognition():
    transport = SimpleNamespace(
        input=lambda: FrameProcessor(name="input"),
        output=lambda: FrameProcessor(name="output"),
    )
    stt = FrameProcessor(name="stt")
    user = FrameProcessor(name="user")
    pipeline = build_pipeline(
        transport,
        stt,
        FrameProcessor(name="audio"),
        user,
        FrameProcessor(name="assistant"),
        FrameProcessor(name="monitor"),
        [FrameProcessor(name="generation")],
        FrameProcessor(name="metrics"),
        FrameProcessor(name="funnel"),
    )
    processors = pipeline.processors[1:-1]  # drop the pipeline's own source/sink
    index = processors.index(stt)
    assert isinstance(processors[index + 1], KeypadAggregator)
    assert processors[index + 2] is user


@pytest.mark.asyncio
async def test_the_greeting_policy_keeps_the_keypad_turn_in_front():
    normal = with_keypad_turns(
        UserTurnStrategies(
            start=[MinWordsUserTurnStartStrategy(min_words=9)],
            stop=[ExternalUserTurnStopStrategy()],
        )
    )
    installed = []
    controller = SimpleNamespace(
        user_turn_strategies=normal,
        has_active_user_turn=False,
        update_strategies=AsyncMock(side_effect=installed.append),
    )
    user = SimpleNamespace(
        user_turn_controller=controller,
        add_event_handler=lambda *_: None,
    )
    for screening in (False, True):
        greeting = GreetingController(
            SimpleNamespace(greeting_pending=not screening, greeting=None),
            lambda: None,
            is_screening=lambda screening=screening: screening,
        )
        greeting.bind(user)
        await greeting._update_strategies(user, Frame())
        start = installed[-1].start
        assert start[0] is normal.start[0]
        assert isinstance(start[1], MinWordsUserTurnStartStrategy)
        assert installed[-1].stop is normal.stop


# --- through the real aggregators ------------------------------------------------


@pytest.mark.asyncio
async def test_a_press_with_the_bot_silent_reaches_the_model(call):
    await call.press(KeypadEntry.ONE)
    messages = await call.next_context(KEYPAD_IDLE_TIMEOUT_SECS + 1.5)
    assert call.user_messages(messages) == [keypad_text("1")]


@pytest.mark.asyncio
async def test_pound_ends_an_entry_at_once(call):
    loop = asyncio.get_running_loop()
    pressed = loop.time()
    await call.press(KeypadEntry.ONE, KeypadEntry.TWO, KeypadEntry.POUND)
    messages = await call.next_context(0.5)
    assert loop.time() - pressed < KEYPAD_IDLE_TIMEOUT_SECS
    assert call.user_messages(messages) == [keypad_text("12#")]


@pytest.mark.asyncio
async def test_a_press_while_the_bot_speaks_interrupts_it_and_reaches_the_model(call):
    await call.send(BotStartedSpeakingFrame())
    # Control: two spoken words are below the three-word threshold and dropped.
    await call.send(TranscriptionFrame("hello there", "caller", "", finalized=True))
    await asyncio.sleep(0.2)
    assert call.sink.contexts.empty()
    await call.press(KeypadEntry.ONE)
    messages = await call.next_context(KEYPAD_IDLE_TIMEOUT_SECS + 1.5)
    assert call.user_messages(messages) == [keypad_text("1")]
    kinds = [type(f) for f in call.sink.downstream]
    # The keypad turn's own interruption, ahead of the inference it runs.
    assert InterruptionFrame in kinds
    assert kinds.index(InterruptionFrame) < kinds.index(LLMContextFrame)


@pytest.mark.asyncio
@pytest.mark.parametrize("call", [{"keypad_turns": False}], indirect=True)
async def test_without_the_keypad_strategies_the_threshold_drops_the_press(call):
    # The risk this patch closes: "[keypad] 1" is two words.
    await call.send(BotStartedSpeakingFrame())
    await call.press(KeypadEntry.ONE)
    with pytest.raises(asyncio.TimeoutError):
        await call.next_context(KEYPAD_IDLE_TIMEOUT_SECS + 1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("call", [{"greeting": True}], indirect=True)
async def test_a_press_during_the_greeting_reaches_the_model(call):
    await call.send(BotStartedSpeakingFrame())
    await asyncio.sleep(0.05)
    # The greeting's own three-word policy is in force, keypad turn in front.
    start = call.user.user_turn_controller.user_turn_strategies.start
    assert isinstance(start[0], KeypadUserTurnStartStrategy)
    assert start[1]._min_words == TURN_START_MIN_WORDS and start[1]._bot_speaking
    await call.press(KeypadEntry.TWO)
    messages = await call.next_context(KEYPAD_IDLE_TIMEOUT_SECS + 1.5)
    assert call.user_messages(messages) == [keypad_text("2")]
    assert InterruptionFrame in [type(f) for f in call.sink.downstream]


@pytest.mark.asyncio
async def test_an_entry_finished_while_muted_is_delivered_when_the_mute_lifts(call):
    call.muted = True
    await call.send(Frame())  # the user aggregator re-evaluates its mute
    await call.press(KeypadEntry.THREE)
    await asyncio.sleep(KEYPAD_IDLE_TIMEOUT_SECS + 0.5)
    assert call.sink.contexts.empty()
    call.muted = False
    await call.send(Frame())
    messages = await call.next_context(1.5)
    assert call.user_messages(messages) == [keypad_text("3")]


@pytest.mark.asyncio
async def test_no_press_no_turn(call):
    # A web call carries no keypad: the aggregator only passes frames on.
    await call.send(TranscriptionFrame("", "caller", ""))
    with pytest.raises(asyncio.TimeoutError):
        await call.next_context(KEYPAD_IDLE_TIMEOUT_SECS + 0.5)


@pytest.mark.asyncio
async def test_a_held_entry_joins_one_still_being_typed_when_the_mute_lifts(call):
    call.muted = True
    await call.send(Frame())
    await call.press(KeypadEntry.THREE)
    await asyncio.sleep(KEYPAD_IDLE_TIMEOUT_SECS + 0.3)  # "3" is now held
    await call.press(KeypadEntry.FOUR)  # "4" is mid-entry when the mute lifts
    call.muted = False
    await call.send(Frame())
    messages = await call.next_context(KEYPAD_IDLE_TIMEOUT_SECS + 1.0)
    assert call.user_messages(messages) == [keypad_text("34")]
