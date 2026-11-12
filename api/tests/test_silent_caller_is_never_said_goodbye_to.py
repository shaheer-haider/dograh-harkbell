"""A caller who has not spoken is never ended by the model, but silence still ends the call.

Production run 4615 (``-harkbell.16``): the model said goodbye three times
before the caller had said a word; the early-end guard refused twice and let
the third through, so the caller heard "thanks for calling Mirror Labs, have
a good day" and was never greeted. Here the model asks to say goodbye three
times in its first turn, then once more when the engine's silence prompt runs
it: nothing ends the call but the engine's own second silence.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.tests import MockTTSService
from pipecat.tests.mock_llm_service import ContextCapturingMockLLM, MockLLMService
from pipecat.tests.mock_transport import MockTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.enums import EndTaskReason

from api.services.workflow.dto import ReactFlowDTO
from api.services.workflow.pipecat_engine import PipecatEngine
from api.services.workflow.pipecat_engine_callbacks import IDLE_PROMPTS
from api.services.workflow.workflow_graph import WorkflowGraph
from api.tests.pipecat_test_utils import run_engine_test_pipeline

GREETING = "Hi, this is Emma at Mirror Labs. How can I help you?"


def _graph(greeting: str | None) -> WorkflowGraph:
    start = {
        "name": "Start",
        "prompt": "Answer the phone.",
        "allow_interrupt": True,
        "add_global_prompt": False,
    }
    if greeting:
        start["greeting"] = greeting
    return WorkflowGraph(
        ReactFlowDTO.model_validate(
            {
                "nodes": [
                    {"id": "start", "type": "startCall", "position": {"x": 0, "y": 0}, "data": start},
                    {
                        "id": "end",
                        "type": "endCall",
                        "position": {"x": 0, "y": 200},
                        "data": {"name": "End", "prompt": "Say goodbye.", "add_global_prompt": False},
                    },
                ],
                "edges": [
                    {
                        "id": "start-end",
                        "source": "start",
                        "target": "end",
                        "data": {"label": "Finish", "condition": "When they are done."},
                    }
                ],
            }
        )
    )


def _goodbyes(count: int, first: int = 0):
    return MockLLMService.create_multiple_function_call_chunks(
        [
            {"name": "say_goodbye", "arguments": {}, "tool_call_id": f"bye{first + n}"}
            for n in range(count)
        ]
    )


async def _silent_call(greeting: str | None):
    llm = ContextCapturingMockLLM(
        mock_steps=[
            # Its first turn, before the caller has said anything.
            _goodbyes(3),
            # Run again by the engine's first silence prompt, it tries again.
            _goodbyes(1, first=3),
            MockLLMService.create_text_chunks("Goodbye."),
        ],
        chunk_delay=0.001,
    )
    tts = MockTTSService(mock_audio_duration_ms=40, frame_delay=0)
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
    graph = _graph(greeting)
    engine = PipecatEngine(
        llm=llm, context=context, workflow=graph, call_context_vars={}, workflow_run_id=1
    )
    user, assistant = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(),
        assistant_params=LLMAssistantAggregatorParams(),
    )
    monitor = engine.call_monitor
    task = PipelineWorker(
        Pipeline([transport.input(), user, monitor, llm, tts, transport.output(), assistant]),
        params=PipelineParams(),
        enable_rtvi=False,
    )
    engine.call_worker = task
    engine.set_transport_output(transport.output())
    monitor.bind_user(user, idle_timeout=0.4)
    ended = []

    async def end_call_with_reason(reason, **_kwargs):
        ended.append(reason)
        await task.cancel()

    engine.end_call_with_reason = end_call_with_reason

    async def open_the_call():
        await engine.set_node(graph.start_node_id)
        tool = SimpleNamespace(definition={"config": {}})
        llm.register_function(
            "say_goodbye",
            engine._custom_tool_manager._create_end_call_handler(tool, "say_goodbye"),
        )
        # The model opens, as it does at a step with no greeting of its own.
        engine.expect_response()
        await engine.active_agent.run_llm(context)

    with patch(
        "api.db:db_client.get_organization_id_by_workflow_run_id",
        new_callable=AsyncMock,
        return_value=1,
    ):
        await run_engine_test_pipeline(task, engine, transport, on_ready=open_the_call, timeout=15)
    return llm, tts, ended


def _ran_only_for_itself_and_silence(llm):
    """No turn came from a refusal: the model ran for its own first turn, then
    for the engine's silence prompts (the second one's may be cut short by the
    end of the call)."""
    assert len(llm.captured_contexts) in (2, 3)
    for step, prompt in ((1, IDLE_PROMPTS[0]), (2, IDLE_PROMPTS[1])):
        captured = llm.get_context_at_step(step)
        if captured is not None:
            assert captured["messages"][-1] == {"role": "user", "content": prompt}


@pytest.mark.asyncio
async def test_three_goodbyes_before_the_caller_speaks_greet_them_once_and_silence_ends_it():
    llm, tts, ended = await _silent_call(GREETING)
    # Greeted once, by the engine, and never said goodbye to by the model.
    assert tts.received_texts.count(GREETING) == 1
    assert "Goodbye." not in tts.received_texts[: tts.received_texts.index(GREETING) + 1]
    _ran_only_for_itself_and_silence(llm)
    tool_results = [
        m["content"]
        for m in llm.get_context_at_step(1)["messages"]
        if m.get("role") == "tool"
    ]
    assert len(tool_results) == 3 and all("refused" in r for r in tool_results)
    # The engine's second silence ended the call.
    assert ended == [EndTaskReason.USER_IDLE_MAX_DURATION_EXCEEDED.value]


@pytest.mark.asyncio
async def test_with_no_greeting_to_say_the_call_still_waits_and_silence_ends_it():
    llm, tts, ended = await _silent_call(None)
    assert GREETING not in tts.received_texts
    _ran_only_for_itself_and_silence(llm)
    assert ended == [EndTaskReason.USER_IDLE_MAX_DURATION_EXCEEDED.value]
