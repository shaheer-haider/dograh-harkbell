"""Production run 4603: "Hello?" over the greeting after a recorded notice.

The copy of Mirror Labs opens with ``take_pathway_without_model``: the start
node's pathway plays the recorded-line notice (transition audio 405, about
4 s), and the next node's first generation -- the greeting -- follows it. The
caller said "Hello?" while the greeting was still playing, and the agent
said the whole greeting again (Harkbell T-670, engine ``-harkbell.15``).

This plays that sequence through real aggregation, a word-timed speech
service like ElevenLabs, and a real-time output, with the production turn
start (three words, "hello" a break-in phrase), and reads what the model was
given for the reply to "Hello?".
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.tests import MockTTSService
from pipecat.tests.mock_llm_service import ContextCapturingMockLLM, MockLLMService
from pipecat.tests.mock_transport import MockTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from api.services.pipecat.cut_off import CUT_OFF_MARKER, harkbell_context_aggregators
from api.services.pipecat.phrase_turn_start import create_turn_start_strategy
from api.services.workflow.dto import ReactFlowDTO
from api.services.workflow.pipecat_engine import PipecatEngine
from api.services.workflow.workflow_graph import WorkflowGraph
from api.tests.pipecat_test_utils import run_engine_test_pipeline
from api.tests.test_take_pathway_without_model import screening_definition

GREETING = "Hi, this is Emma at Mirror Labs. Thanks for calling. How can I help you?"
NOTICE_SECS = 1.0
SENTENCE_SECS = 1.0
WORD_GAP = 0.15


class ElevenLabsLike(MockTTSService):
    """Word timings like the dialogue relay's: each word reported as it plays."""

    def __init__(self, **kwargs):
        super().__init__(
            push_text_frames=False,
            mock_audio_duration_ms=int(SENTENCE_SECS * 1000),
            chunk_size=640,
            frame_delay=0,
            **kwargs,
        )
        self._offset = 0.0

    async def run_tts(self, text, context_id):
        await self.start_word_timestamps()
        await self.add_word_timestamps(
            [
                (word, self._offset + index * WORD_GAP)
                for index, word in enumerate(text.split())
            ],
            context_id,
        )
        self._offset += SENTENCE_SECS
        async for frame in super().run_tts(text, context_id):
            yield frame


async def _run_4603(*, main_greeting=None, hello_after=1.5 * SENTENCE_SECS + 0.3):
    graph = WorkflowGraph(
        ReactFlowDTO.model_validate(
            screening_definition(notice_audio=True, main_greeting=main_greeting)
        )
    )

    async def fetch_recording(*, recording_pk=None, recording_id=None):
        return SimpleNamespace(
            audio=b"\x01\x00" * int(16000 * NOTICE_SECS),
            transcript="You are talking to an AI and this call is recorded.",
        )

    reply = MockLLMService.create_text_chunks("Yes, I'm here. How can I help you?")
    llm = ContextCapturingMockLLM(
        # With a greeting of its own the step says it, and the model's first
        # turn is the reply to "Hello?".
        mock_steps=[reply]
        if main_greeting
        else [MockLLMService.create_text_chunks(GREETING), reply],
        chunk_delay=0.001,
    )
    replies = 1 if main_greeting else 2
    tts = ElevenLabsLike()
    transport = MockTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_out_end_silence_secs=0,
        ),
    )
    context = LLMContext()
    engine = PipecatEngine(
        llm=llm, context=context, workflow=graph, call_context_vars={}, workflow_run_id=1
    )
    # Production's turn start: three words, "hello" always cuts in.
    user, assistant = harkbell_context_aggregators(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[
                    create_turn_start_strategy(
                        min_words=3,
                        ignore_phrases=["yeah", "uh huh"],
                        break_in_phrases=["hello", "hold on"],
                    )
                ],
                stop=[ExternalUserTurnStopStrategy()],
            ),
            should_interrupt=engine.should_interrupt_user_turn,
        ),
        assistant_params=LLMAssistantAggregatorParams(),
    )
    task = PipelineWorker(
        Pipeline([transport.input(), user, llm, tts, transport.output(), assistant]),
        params=PipelineParams(),
        enable_rtvi=False,
    )
    engine.call_worker = task
    engine.set_fetch_recording_audio(fetch_recording)
    loop = asyncio.get_running_loop()
    timeline = {}

    async def open_the_call():
        await engine.set_node(graph.start_node_id)
        timeline["connect"] = loop.time()
        await engine.queue_node_opening(
            node_id=graph.start_node_id, previous_node_id=None, generate_if_no_greeting=True
        )

        async def caller():
            # The notice, then part of the greeting.
            await asyncio.sleep(NOTICE_SECS + hello_after)
            for frame in (
                ProposedUserStartedSpeakingFrame(),
                InterimTranscriptionFrame("Hello", "caller", ""),
                TranscriptionFrame("Hello?", "caller", "", finalized=True),
                ProposedUserStoppedSpeakingFrame(),
            ):
                await task.queue_frame(frame)
                await asyncio.sleep(0.05)
            async with asyncio.timeout(8):
                while len(llm.captured_contexts) < replies:
                    await asyncio.sleep(0.02)
            await asyncio.sleep(0.2)
            await task.cancel()

        asyncio.create_task(caller())

    with patch(
        "api.db:db_client.get_organization_id_by_workflow_run_id",
        new_callable=AsyncMock,
        return_value=1,
    ):
        await run_engine_test_pipeline(
            task, engine, transport, on_ready=open_the_call, timeout=20
        )
    return llm


@pytest.mark.asyncio
async def test_the_reply_to_hello_sees_the_greeting_as_cut_off():
    llm = await _run_4603()
    assert len(llm.captured_contexts) == 2
    messages = llm.get_context_at_step(1)["messages"]
    said = [m["content"] for m in messages if m.get("role") == "assistant" and m.get("content")]
    heard = [m["content"] for m in messages if m.get("role") == "user"]
    assert heard == ["Hello?"]
    # The first sentence and a half had played: what played, marked.
    assert len(said) == 1, said
    assert said[0].endswith(f"— {CUT_OFF_MARKER}"), said
    assert said[0].startswith("Hi, this is Emma at Mirror Labs."), said
    assert "How can I help you?" not in said[0], said
    # And it sits in front of the caller's words.
    roles = [m.get("role") for m in messages if m.get("role") in ("assistant", "user") and m.get("content")]
    assert roles == ["assistant", "user"]


@pytest.mark.asyncio
async def test_a_configured_greeting_after_the_notice_is_kept_as_cut_off_too():
    # Config 192: the screening main step opens with its own greeting.
    # A configured greeting is one utterance to the speech service, so here
    # its whole second of audio carries all three sentences: cut in at half.
    llm = await _run_4603(main_greeting=GREETING, hello_after=0.55)
    assert len(llm.captured_contexts) == 1
    messages = llm.get_context_at_step(0)["messages"]
    said = [m["content"] for m in messages if m.get("role") == "assistant" and m.get("content")]
    assert len(said) == 1, said
    assert said[0].startswith("Hi, this is"), said
    assert said[0].endswith(f"— {CUT_OFF_MARKER}"), said
    assert "How can I help you?" not in said[0], said
    assert not any(m.get("tool_calls") for m in messages)
