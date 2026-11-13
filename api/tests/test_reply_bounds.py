"""A reply's cost and length are bounded, and a call ends while one is mid-flight.

Production run 4817 (``-harkbell.21``, 2026-11-13): a silent caller, the first
silence prompt, a 3.9 s first token -- and then the reply said nothing for 64 s
with no stall warning, the caller hung up, and the whole engine process
stopped answering (health checks included) until ECS killed it. The stall
guard only measures silence; a stream that keeps sending chunks the filters
hold back never trips it, and every chunk made the filter search everything
held so far -- with a run of marks, a search that alone took seconds.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pipecat.frames.frames import LLMContextFrame, LLMTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.services.minimax.llm import MiniMaxLLMService
from pipecat.tests import MockTTSService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.utils.enums import EndTaskReason

import api.services.pipecat.minimax_llm as minimax_llm
from api.services.pipecat import spoken_reasoning
from api.services.pipecat.agent_bridge import AgentBridgeProcessor
from api.services.pipecat.cut_off import harkbell_context_aggregators
from api.services.pipecat.minimax_llm import HarkbellMiniMaxLLMService
from api.services.pipecat.pipeline_builder import create_agent_worker
from api.services.pipecat.reasks import ReAskCount, ReAskGuard
from api.services.pipecat.spoken_reasoning import SpokenReasoningFilter
from api.services.pipecat.termination_funnel_processor import (
    TerminationFunnelProcessor,
)
from api.services.pipecat.worker_runner import (
    create_worker_runner,
    run_worker_runner,
    wait_for_pipeline_worker_started,
)
from api.services.workflow.pipecat_engine import PipecatEngine
from api.services.workflow.pipecat_engine_callbacks import IDLE_PROMPTS
from api.tests.telephony.test_telnyx_transfer_teardown import (
    _Strategies,
    _TelnyxSocket,
)


def _chunk(text: str | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="c",
        choices=[Choice(index=0, delta=ChoiceDelta(content=text, role="assistant"), finish_reason=None)],
        created=0,
        model="m",
        object="chat.completion.chunk",
    )


class _Stream:
    """A streaming response: chunks on a schedule, closed when abandoned."""

    def __init__(self, chunks, *, first_after=0.0, every=0.0, endless=None):
        self._chunks = list(chunks)
        self._first_after = first_after
        self._every = every
        self._endless = endless
        self._sent = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(self._first_after if self._sent == 0 else self._every)
        self._sent += 1
        if self._chunks:
            return self._chunks.pop(0)
        if self._endless is not None:
            return _chunk(self._endless)
        raise StopAsyncIteration

    async def close(self):
        self.closed = True


class _Heartbeat:
    """How long the event loop went without running anything else."""

    def __init__(self):
        self.longest = 0.0
        self._task = None

    async def _beat(self):
        last = time.perf_counter()
        while True:
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            self.longest = max(self.longest, now - last)
            last = now

    def __enter__(self):
        self._task = asyncio.create_task(self._beat())
        return self

    def __exit__(self, *exc):
        self._task.cancel()


# ── the filters' cost ─────────────────────────────────────────────────────


def test_a_run_of_marks_is_searched_once_not_from_every_mark():
    started = time.perf_counter()
    assert spoken_reasoning._SENTENCE_END.search("." * 50_000) is None
    assert time.perf_counter() - started < 0.2
    # What it finds is unchanged.
    assert spoken_reasoning._SENTENCE_END.search("Wait... what? Yes").group() == "... "


def test_a_reply_that_never_ends_a_sentence_is_let_through_and_stays_cheap():
    reply = "la " * 20_000
    reasoning = SpokenReasoningFilter(reasks=ReAskGuard(ReAskCount()))
    said, slowest, started = "", 0.0, time.perf_counter()
    for i in range(0, len(reply), 7):
        before = time.perf_counter()
        said += reasoning.feed(reply[i : i + 7])
        slowest = max(slowest, time.perf_counter() - before)
    said += reasoning.finish()
    assert said == reply
    assert slowest < 0.02
    assert time.perf_counter() - started < 2.0


def test_a_degenerate_run_of_marks_does_not_hold_the_loop():
    reasoning = SpokenReasoningFilter()
    slowest = 0.0
    for _ in range(20_000):
        before = time.perf_counter()
        reasoning.feed("..")
        slowest = max(slowest, time.perf_counter() - before)
    reasoning.finish()
    assert slowest < 0.02


def test_a_long_reply_is_judged_for_a_re_ask_at_its_end_only():
    guard = ReAskGuard(ReAskCount())
    started = time.perf_counter()
    for _ in range(3_000):
        assert guard.pass_on("Sorry about the wait, there is a queue today. ")
    assert time.perf_counter() - started < 1.0
    assert not guard.is_reask
    assert guard.pass_on("Sorry, could you say that again?")
    assert guard.is_reask


# ── the service ───────────────────────────────────────────────────────────


def _service() -> HarkbellMiniMaxLLMService:
    with patch.object(HarkbellMiniMaxLLMService, "create_client"):
        return HarkbellMiniMaxLLMService(api_key="key")


async def _said(service, streams, context) -> str:
    """Run one response; return what reached speech."""
    pushed = []

    async def push_frame(frame, direction=None):
        if isinstance(frame, LLMTextFrame):
            pushed.append(frame.text)

    async def completions(self, context):
        return streams.pop(0)

    service.push_frame = push_frame
    with patch.object(MiniMaxLLMService, "get_chat_completions", completions):
        await service._process_context(context)
    return "".join(pushed)


def _idle_context() -> LLMContext:
    return LLMContext(
        messages=[
            {"role": "assistant", "content": "Hi, this is Emma at Mirror Labs. How can I help you?"},
            {"role": "user", "content": IDLE_PROMPTS[0]},
        ]
    )


@pytest.mark.asyncio
async def test_a_first_token_just_after_the_deadline_is_asked_again_and_answered():
    late = _Stream([_chunk("Hello? Are you still there?")], first_after=0.5)
    answer = _Stream([_chunk("Hello? Are you still there?")])
    with patch.object(minimax_llm, "FIRST_OUTPUT_SECS", 0.3), _Heartbeat() as beat:
        said = await asyncio.wait_for(_said(_service(), [late, answer], _idle_context()), 5)
    assert said == "Hello? Are you still there?"
    assert late.closed and answer.closed
    assert beat.longest < 0.1


@pytest.mark.asyncio
async def test_a_reply_that_streams_without_end_is_ended_and_never_holds_the_loop():
    runaway = _Stream([], every=0.001, endless="..")
    with patch.object(minimax_llm, "RESPONSE_MAX_SECS", 0.5), _Heartbeat() as beat:
        started = time.perf_counter()
        await asyncio.wait_for(_said(_service(), [runaway], _idle_context()), 5)
    assert time.perf_counter() - started < 1.5
    assert runaway.closed
    assert beat.longest < 0.1


# ── a hangup while the silence prompt's reply is mid-flight ──────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["late_first_token", "runaway"])
async def test_a_hangup_during_the_silence_prompts_reply_ends_the_call(reply):
    strategies = _Strategies()
    transport = FastAPIWebsocketTransport(
        websocket=_TelnyxSocket(),
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            serializer=TelnyxFrameSerializer(
                stream_id="stream",
                call_control_id="v3:caller-leg",
                api_key="key",
                outbound_encoding="PCMU",
                inbound_encoding="PCMU",
                transfer_strategy=strategies,
                hangup_strategy=strategies,
            ),
            allowed_origins=[],
        ),
    )
    runner = create_worker_runner()
    llm = _service()
    tts = MockTTSService(mock_audio_duration_ms=40, frame_delay=0)
    context = _idle_context()
    engine = PipecatEngine(llm=llm, context=context, workflow=None, call_context_vars={})
    engine.active_agent.is_child = True
    user, assistant = harkbell_context_aggregators(
        context,
        user_params=LLMUserAggregatorParams(user_idle_timeout=0),
        assistant_params=LLMAssistantAggregatorParams(),
    )
    monitor = engine.call_monitor
    monitor.bind_user(user, idle_timeout=30)
    funnel = TerminationFunnelProcessor()

    async def dispose(reason, error=None):
        await engine.end_call_with_reason(reason, abort_immediately=True)

    funnel.set_termination_handler(dispose)
    bridge = AgentBridgeProcessor(
        bus=runner.bus,
        worker_name="call-1",
        selected_visit=lambda: engine.selected_visit_id,
        allow_inference=lambda: engine.agent_can_generate(engine.active_agent),
    )
    worker = PipelineWorker(
        Pipeline([transport.input(), funnel, user, monitor, bridge, transport.output(), assistant]),
        name="call-1",
        params=PipelineParams(audio_out_sample_rate=8000),
        enable_rtvi=False,
        idle_timeout_secs=None,
    )
    child = create_agent_worker(
        Pipeline([llm, tts]),
        name=engine.active_agent.visit_id,
        call_worker_name=worker.name,
        audio_config=SimpleNamespace(transport_in_sample_rate=8000, transport_out_sample_rate=8000),
    )
    engine.active_agent.worker = child
    engine.call_worker = worker
    engine.set_transport_output(transport.output())

    if reply == "late_first_token":
        # Abandoned just past the deadline, then asked again, and the second
        # attempt is still thinking when the caller hangs up.
        streams = [
            _Stream([_chunk("Hello?")], first_after=0.5),
            _Stream([_chunk("Hello?")], first_after=30),
        ]
    else:
        streams = [_Stream([], every=0.001, endless="..")]

    async def completions(self, context):
        return streams.pop(0) if streams else _Stream([_chunk("Goodbye.")])

    with (
        patch.object(minimax_llm, "FIRST_OUTPUT_SECS", 0.3),
        patch.object(MiniMaxLLMService, "get_chat_completions", completions),
        _Heartbeat() as beat,
    ):
        task = asyncio.create_task(run_worker_runner(runner, worker))
        await wait_for_pipeline_worker_started(worker, timeout=10, run_task=task)
        await worker.add_workers(child)
        await wait_for_pipeline_worker_started(child, timeout=3)
        assert await engine.activate_agent(engine.active_agent, timeout=1)
        monitor.activate()
        engine.expect_response()
        await worker.queue_frame(LLMContextFrame(context))
        await asyncio.sleep(1.0)
        # As on_client_disconnected does when the caller hangs up.
        await asyncio.wait_for(
            engine.end_call_with_reason(EndTaskReason.USER_HANGUP.value, abort_immediately=True),
            10,
        )
        await asyncio.wait_for(task, 10)

    assert engine._gathered_context["call_status"] == EndTaskReason.USER_HANGUP.value
    assert child.has_finished()
    assert beat.longest < 0.25
