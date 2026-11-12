"""A line the caller cut off stays in the context as far as it played, marked.

Harkbell T-670: "Hello?" over the greeting restarted it four times on one
call, because neither a cut-off greeting nor the sentence a reply was cut off
in left anything in the context. These drive real Pipecat aggregation, TTS and
output queues through an interruption and read the context afterwards.
"""

import asyncio
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    EndFrame,
    LLMContextFrame,
    SpeechBoundaryFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.tests import MockLLMService
from pipecat.tests.mock_transport import MockOutputTransport
from pipecat.tests.mock_tts_service import MockTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_start import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from api.services.pipecat.cut_off import (
    CUT_OFF_MARKER,
    harkbell_context_aggregators,
    mark_cut_off,
    sentences_started,
)
from api.services.pipecat.speech_playback import PlaybackOutcome
from api.services.pipecat.worker_runner import run_pipeline_worker
from api.services.workflow.pipecat_engine import PipecatEngine

GREETING = "Hi, this is Emma at Mirror Labs. How can I help you today?"
CHUNK = 640  # 20 ms of 16 kHz mono PCM
# The output transport plays audio in 40 ms chunks: a second is 25 of them.
PER_SECOND = 25
WORD_GAP = 0.3


def test_the_marker_follows_what_was_said():
    assert mark_cut_off("Hi, this is Emma at ") == f"Hi, this is Emma at — {CUT_OFF_MARKER}"
    assert CUT_OFF_MARKER == "[cut off by the caller]"


@pytest.mark.parametrize(
    ("fraction", "kept"),
    [
        (0.0, ""),
        (0.01, "Hi, this is Emma at Mirror Labs."),
        (0.5, "Hi, this is Emma at Mirror Labs."),
        (0.6, GREETING),
        (1.0, GREETING),
    ],
)
def test_a_recording_keeps_the_sentences_its_audio_reached(fraction, kept):
    assert sentences_started(GREETING, fraction) == kept


class WordTimedTTS(MockTTSService):
    """A speech service that reports each word as it plays, like ElevenLabs.

    A word every ``WORD_GAP`` seconds, each sentence's three seconds of audio
    after the last. Word frames are released on the clock rather than behind
    the audio, so these tests cut the reply off at a time, not at a chunk.
    """

    def __init__(self, **kwargs):
        kwargs["mock_audio_duration_ms"] = 3000
        super().__init__(push_text_frames=False, **kwargs)
        self._offset = 0.0

    async def run_tts(self, text, context_id):
        await self.start_word_timestamps()
        words = text.split()
        await self.add_word_timestamps(
            [(word, self._offset + index * WORD_GAP) for index, word in enumerate(words)],
            context_id,
        )
        self._offset += 3.0
        async for frame in super().run_tts(text, context_id):
            yield frame


class MeteredOutput(MockOutputTransport):
    """An output whose playback can be stopped after a number of audio chunks."""

    def __init__(self):
        super().__init__(
            params=TransportParams(
                audio_out_enabled=True,
                audio_out_sample_rate=16000,
                audio_out_end_silence_secs=0,
            )
        )
        self.written = 0
        self.allowance: int | None = None
        self.held = asyncio.Event()
        self._more = asyncio.Event()

    def stop_after(self, chunks: int) -> None:
        self.allowance = chunks
        self._more.clear()

    def play_on(self) -> None:
        self.allowance = None
        self._more.set()

    async def write_audio_frame(self, frame):
        while self.allowance is not None and self.written >= self.allowance:
            self.held.set()
            await self._more.wait()
        self.written += 1
        return await super().write_audio_frame(frame)


class Call:
    """User aggregator -> LLM -> TTS -> output -> Harkbell assistant aggregator."""

    def __init__(self, *, reply: str = GREETING, word_timed: bool = False):
        self.context = LLMContext()
        self.context.add_message({"role": "system", "content": "Answer the phone."})
        self.engine = PipecatEngine(workflow=None, call_context_vars={})
        self.engine.set_context(self.context)
        self.llm = MockLLMService(
            mock_chunks=MockLLMService.create_text_chunks(reply), chunk_delay=0
        )
        tts_class = WordTimedTTS if word_timed else MockTTSService
        # One second of audio per sentence, written 20 ms at a time.
        self.tts = tts_class(mock_audio_duration_ms=1000, chunk_size=CHUNK, frame_delay=0)
        self.output = MeteredOutput()
        self.user, self.assistant = harkbell_context_aggregators(
            self.context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    start=[ExternalUserTurnStartStrategy()],
                    stop=[ExternalUserTurnStopStrategy()],
                )
            ),
            assistant_params=LLMAssistantAggregatorParams(),
        )
        self.turns = []

        @self.assistant.event_handler("on_assistant_turn_stopped")
        async def stopped(_aggregator, message):
            self.turns.append(message)

        self.worker = PipelineWorker(
            Pipeline([self.user, self.llm, self.tts, self.output, self.assistant]),
            enable_rtvi=False,
        )
        self.engine.call_worker = self.worker
        self.engine.active_agent.worker = SimpleNamespace(queue_frame=self.tts.queue_frame)
        self.engine.active_agent.llm = self.llm
        self.engine.set_transport_output(self.output)
        self.engine.greeting.bind(self.user)

    def assistant_messages(self) -> list[str]:
        return [
            m["content"]
            for m in self.context.get_messages()
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content")
        ]

    async def reply(self) -> None:
        await self.llm.queue_frame(LLMContextFrame(self.context))

    async def cut_off_after(self, chunks: int) -> None:
        """Let ``chunks`` of audio play, then have the caller speak over it."""
        self.output.stop_after(chunks)
        await asyncio.wait_for(self.output.held.wait(), 3)
        await self.user.broadcast_interruption()
        await asyncio.sleep(0.2)
        self.output.play_on()

    async def cut_off_at(self, seconds: float) -> None:
        """Have the caller speak over the reply ``seconds`` after it was asked for."""
        await asyncio.sleep(seconds)
        await self.user.broadcast_interruption()

    async def settle(self) -> None:
        await asyncio.sleep(0.3)


@pytest.fixture
async def call(request):
    harness = Call(**getattr(request, "param", {}))
    ready = asyncio.Event()

    @harness.worker.event_handler("on_pipeline_started")
    async def started(*_):
        ready.set()

    runner = asyncio.create_task(run_pipeline_worker(harness.worker))
    try:
        await asyncio.wait_for(ready.wait(), 3)
        yield harness
    finally:
        harness.output.play_on()
        if not runner.done():
            await harness.worker.queue_frame(EndFrame())
        await asyncio.wait_for(runner, 5)


@pytest.mark.asyncio
async def test_a_reply_cut_in_its_first_sentence_keeps_that_sentence(call):
    await call.reply()
    # 0.4 s into the first sentence's second of audio.
    await call.cut_off_after(10)
    await call.settle()
    assert call.assistant_messages() == [f"Hi, this is Emma at Mirror Labs. — {CUT_OFF_MARKER}"]
    # The transcript and the turn events carry what was said, unmarked.
    assert call.turns[-1].interrupted
    assert call.turns[-1].content == "Hi, this is Emma at Mirror Labs."


@pytest.mark.asyncio
async def test_a_reply_cut_in_its_second_sentence_keeps_both(call):
    await call.reply()
    # 1.4 s: the first sentence's second has played, the second has begun.
    await call.cut_off_after(PER_SECOND + 10)
    await call.settle()
    assert call.assistant_messages() == [f"{GREETING} — {CUT_OFF_MARKER}"]


@pytest.mark.asyncio
@pytest.mark.parametrize("call", [{"word_timed": True}], indirect=True)
async def test_with_word_timings_the_words_that_played_are_kept(call):
    await call.reply()
    # A word every 0.3 s: 0.75 s in, three have played.
    await call.cut_off_at(0.75)
    await call.settle()
    assert call.assistant_messages() == [f"Hi, this is — {CUT_OFF_MARKER}"]


@pytest.mark.asyncio
async def test_a_reply_that_finishes_is_kept_unmarked(call):
    await call.reply()
    async with asyncio.timeout(5):
        while not call.turns:
            await asyncio.sleep(0.05)
    assert call.assistant_messages() == [GREETING]
    assert not call.turns[-1].interrupted


@pytest.mark.asyncio
async def test_speech_meant_to_stay_out_of_the_context_stays_out_when_cut(call):
    # A transition notice: spoken, never part of the conversation.
    await call.tts.queue_frame(
        TTSSpeakFrame("You are talking to an AI and this call is recorded.", append_to_context=False)
    )
    await call.cut_off_after(10)
    await call.settle()
    assert call.assistant_messages() == []


@pytest.mark.asyncio
async def test_a_greeting_cut_off_keeps_what_played(call):
    speech = await call.engine.queue_speech(GREETING, greeting=True, timeout=10)
    await call.cut_off_after(10)
    await call.settle()
    assert speech.outcome is PlaybackOutcome.INTERRUPTED
    # A TTSSpeakFrame is one unit to the speech service: it had begun.
    assert call.assistant_messages() == [f"{GREETING} — {CUT_OFF_MARKER}"]


@pytest.mark.asyncio
@pytest.mark.parametrize("call", [{"word_timed": True}], indirect=True)
async def test_a_greeting_with_word_timings_keeps_its_played_words(call):
    speech = await call.engine.queue_speech(GREETING, greeting=True, timeout=10)
    await call.cut_off_at(0.75)
    await call.settle()
    assert speech.outcome is PlaybackOutcome.INTERRUPTED
    assert call.assistant_messages() == [f"Hi, this is — {CUT_OFF_MARKER}"]


@pytest.mark.asyncio
async def test_a_greeting_cut_off_before_any_audio_leaves_nothing(call):
    call.output.stop_after(0)
    speech = await call.engine.queue_speech(GREETING, greeting=True, timeout=10)
    await asyncio.wait_for(call.output.held.wait(), 3)
    await call.user.broadcast_interruption()
    await call.settle()
    assert speech.outcome is PlaybackOutcome.INTERRUPTED
    assert call.assistant_messages() == []


@pytest.mark.asyncio
async def test_a_greeting_that_plays_through_is_kept_whole(call):
    speech = await call.engine.queue_speech(GREETING, greeting=True, timeout=10)
    assert await asyncio.wait_for(speech.wait(), 5)
    await call.settle()
    assert call.assistant_messages() == [GREETING]


@pytest.mark.asyncio
async def test_a_recorded_greeting_keeps_the_sentences_its_audio_reached(call):
    # Two seconds of recording behind a two-sentence transcript.
    audio = b"\x01\x00" * 32000
    # Queued the way queue_speech queues a recording, straight to the output.
    speech = call.engine.speech_playback.create(timeout=10, greeting=True)
    speech.text = GREETING
    speech.audio_secs = 2.0
    queue = call.output.queue_frame
    await queue(SpeechBoundaryFrame(speech.id, beginning=True))
    await queue(TTSStartedFrame(context_id="recording"))
    transcript = TTSTextFrame(text=GREETING, aggregated_by="recording", context_id="recording")
    transcript.append_to_context = False
    await queue(transcript)
    await queue(TTSAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1))
    await queue(TTSStoppedFrame(context_id="recording"))
    await queue(SpeechBoundaryFrame(speech.id, beginning=False))
    # 0.4 s of 2 s: the first sentence had begun, the second had not.
    await call.cut_off_after(10)
    await call.settle()
    assert speech.outcome is PlaybackOutcome.INTERRUPTED
    assert call.assistant_messages() == [f"Hi, this is Emma at Mirror Labs. — {CUT_OFF_MARKER}"]


@pytest.mark.asyncio
async def test_a_generated_greeting_cut_off_is_kept_once(call):
    # The model's own opening, tracked as the greeting.
    speech = call.engine.speech_playback.expect_response(source=call.llm, greeting=True)
    await call.reply()
    await call.cut_off_after(10)
    await call.settle()
    assert speech.outcome is PlaybackOutcome.INTERRUPTED
    # GreetingController commits it; the aggregator must not add a second copy.
    assert call.assistant_messages() == [f"Hi, this is Emma at Mirror Labs. — {CUT_OFF_MARKER}"]


def test_committing_a_cut_off_line_is_logged_at_info():
    from loguru import logger

    from api.services.pipecat.cut_off import commit_cut_off_line
    from pipecat.utils.run_context import run_id_var

    seen = []
    sink = logger.add(lambda message: seen.append(str(message)), level="INFO")
    # Reset afterwards: the run id is a context variable other tests read.
    token = run_id_var.set("4603")
    try:
        context = LLMContext()
        commit_cut_off_line(context, "Hi, this is Emma at", since=0)
    finally:
        run_id_var.reset(token)
        logger.remove(sink)
    assert context.messages == [
        {"role": "assistant", "content": f"Hi, this is Emma at — {CUT_OFF_MARKER}"}
    ]
    assert any("[run 4603] Kept a line the caller cut off: 19 chars" in line for line in seen)
