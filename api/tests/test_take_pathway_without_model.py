"""A start node with ``take_pathway_without_model`` moves on without the model.

Harkbell T-669: the screening line's start step says nothing of its own, and
the model's first reply is only the tool call that plays the recorded-line
notice -- so a slow minute at the model provider was 41 s of silence before
the caller heard anything. With the flag the engine takes that one edge
itself: the notice plays at once, muted, and the next node's first reply is
generated behind it.
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

from api.services.pipecat.realtime_feedback_observer import (
    warn_if_slow_llm_first_token,
)
from api.services.pipecat.worker_runner import run_pipeline_worker
from api.services.workflow.dto import (
    ReactFlowDTO,
    StartCallNodeData,
    sanitize_workflow_definition,
)
from api.services.workflow.pipecat_engine import PipecatEngine
from api.services.workflow.text_chat_runner import (
    _ResponseWindowState,
    _TaskQueueProxy,
    _TextChatCaptureProcessor,
    _wait_for_quiescence,
)
from api.services.workflow.workflow_graph import WorkflowGraph
from api.tests.pipecat_test_utils import run_engine_test_pipeline

NOTICE = "You are talking to an AI and this call is recorded."
GREETING = "Hi, this is Emma at Mirror Labs. How can I help?"
START_PROMPT = "START PROMPT: take your one pathway silently."
MAIN_PROMPT = "MAIN PROMPT: greet the caller."


def screening_definition(
    *, flag=True, edges=None, notice_audio=False, main_greeting=None
) -> dict:
    """Harkbell's screening shape: an empty start step, one edge, the notice on it."""
    start_data = {
        "name": "harkbell-start",
        "prompt": START_PROMPT,
        "greeting": "",
        "allow_interrupt": False,
        "add_global_prompt": False,
    }
    if flag is not None:
        start_data["take_pathway_without_model"] = flag
    edge_data = {"label": "Begin", "condition": "Always, at once."}
    if notice_audio:
        edge_data |= {
            "transition_speech_type": "audio",
            "transition_speech_recording_id": "7",
        }
    else:
        edge_data["transition_speech"] = NOTICE
    return {
        "nodes": [
            {
                "id": "harkbell-start",
                "type": "startCall",
                "position": {"x": 0, "y": 0},
                "data": start_data,
            },
            {
                "id": "harkbell-main",
                "type": "agentNode",
                "position": {"x": 0, "y": 200},
                "data": {
                    "name": "harkbell-main",
                    "prompt": MAIN_PROMPT,
                    "allow_interrupt": True,
                    "add_global_prompt": False,
                    **({"greeting": main_greeting} if main_greeting is not None else {}),
                },
            },
            {
                "id": "end",
                "type": "endCall",
                "position": {"x": 0, "y": 400},
                "data": {
                    "name": "End",
                    "prompt": "Say goodbye.",
                    "add_global_prompt": False,
                },
            },
        ],
        "edges": edges
        if edges is not None
        else [
            {
                "id": "begin",
                "source": "harkbell-start",
                "target": "harkbell-main",
                "data": edge_data,
            },
            {
                "id": "finish",
                "source": "harkbell-main",
                "target": "end",
                "data": {"label": "End Call", "condition": "When they are done."},
            },
        ],
    }


def screening_graph(**kwargs) -> WorkflowGraph:
    return WorkflowGraph(ReactFlowDTO.model_validate(screening_definition(**kwargs)))


# ── the contract ──────────────────────────────────────────────────────────


def test_the_key_survives_a_save_and_reaches_the_graph():
    definition = screening_definition()
    saved = sanitize_workflow_definition(definition)
    assert saved["nodes"][0]["data"]["take_pathway_without_model"] is True
    dumped = ReactFlowDTO.model_validate(saved).model_dump(mode="json")
    assert dumped["nodes"][0]["data"]["take_pathway_without_model"] is True
    graph = WorkflowGraph(ReactFlowDTO.model_validate(dumped))
    assert graph.nodes["harkbell-start"].take_pathway_without_model


@pytest.mark.parametrize("flag", [None, False])
def test_absent_or_false_is_off(flag):
    graph = screening_graph(flag=flag)
    assert not graph.nodes["harkbell-start"].take_pathway_without_model
    assert StartCallNodeData(name="s", prompt="p").take_pathway_without_model is False


def test_the_key_stays_out_of_the_editor_spec():
    from api.services.workflow.node_specs import get_spec

    names = {prop.name for prop in get_spec("startCall").properties}
    assert "take_pathway_without_model" not in names


def test_a_flagged_start_with_two_pathways_does_not_validate():
    definition = screening_definition()
    definition["edges"].append(
        {
            "id": "straight-to-end",
            "source": "harkbell-start",
            "target": "end",
            "data": {"label": "Hang Up", "condition": "Never."},
        }
    )
    with pytest.raises(ValueError) as raised:
        WorkflowGraph(ReactFlowDTO.model_validate(definition))
    errors = raised.value.args[0]
    assert [error["field"] for error in errors] == ["data.take_pathway_without_model"]


def _engine(graph: WorkflowGraph, **kwargs) -> PipecatEngine:
    return PipecatEngine(
        workflow=graph, call_context_vars={}, context=LLMContext(), **kwargs
    )


def test_only_a_fresh_opening_of_a_flagged_start_qualifies():
    engine = _engine(screening_graph())
    start = "harkbell-start"
    edge = engine._pathway_without_model(
        start, None, opening_context=None, wait_for_playback=False
    )
    assert edge is not None and edge.target == "harkbell-main"
    # Resuming the start node, an answer-supervised opening, and other nodes.
    assert not engine._pathway_without_model(
        start, start, opening_context=None, wait_for_playback=False
    )
    assert not engine._pathway_without_model(
        start, None, opening_context=LLMContext(), wait_for_playback=True
    )
    assert not engine._pathway_without_model(
        "harkbell-main", None, opening_context=None, wait_for_playback=False
    )
    assert not _engine(screening_graph(flag=False))._pathway_without_model(
        start, None, opening_context=None, wait_for_playback=False
    )


def test_a_flag_the_graph_cannot_honour_falls_back_to_the_model():
    # Built unflagged so it validates, then flagged as a stale definition might be.
    definition = screening_definition(flag=False)
    definition["nodes"] = [n for n in definition["nodes"] if n["id"] != "harkbell-main"]
    definition["edges"] = [definition["edges"][0] | {"target": "end"}]
    graph = WorkflowGraph(ReactFlowDTO.model_validate(definition))
    graph.nodes["harkbell-start"].take_pathway_without_model = True
    engine = _engine(graph)
    assert not engine._pathway_without_model(
        "harkbell-start", None, opening_context=None, wait_for_playback=False
    )


def test_a_realtime_call_takes_the_pathway_without_the_model_too():
    # -harkbell.24: the model opened the screening line in words of its own
    # (copy run 4859), so the engine opens a realtime call as it opens any other.
    realtime = _engine(screening_graph(), is_realtime=True)
    edge = realtime._pathway_without_model(
        "harkbell-start", None, opening_context=None, wait_for_playback=False
    )
    assert edge is not None and edge.target == "harkbell-main"


# ── a call ────────────────────────────────────────────────────────────────


async def _run_call(graph: WorkflowGraph, llm, *, fetch_recording=None, until):
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
    engine = PipecatEngine(
        llm=llm,
        context=context,
        workflow=graph,
        call_context_vars={},
        workflow_run_id=1,
    )
    user, assistant = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(),
        assistant_params=LLMAssistantAggregatorParams(),
    )
    task = PipelineWorker(
        Pipeline([transport.input(), user, llm, tts, transport.output(), assistant]),
        params=PipelineParams(),
        enable_rtvi=False,
    )
    engine.call_worker = task
    if fetch_recording is not None:
        engine.set_fetch_recording_audio(fetch_recording)

    async def open_the_call():
        # As event_handlers.py opens a call.
        await engine.set_node(graph.start_node_id)
        await engine.queue_node_opening(
            node_id=graph.start_node_id,
            previous_node_id=None,
            generate_if_no_greeting=True,
        )

        async def finish():
            async with asyncio.timeout(5):
                while not until(llm, tts):
                    await asyncio.sleep(0.02)
            await asyncio.sleep(0.2)
            await task.cancel()

        asyncio.create_task(finish())

    with patch(
        "api.db:db_client.get_organization_id_by_workflow_run_id",
        new_callable=AsyncMock,
        return_value=1,
    ):
        await run_engine_test_pipeline(task, engine, transport, on_ready=open_the_call)
    return engine, tts, context


def _spoke(*texts):
    return lambda llm, tts: all(t in tts.received_texts for t in texts)


@pytest.mark.asyncio
async def test_the_call_opens_with_the_notice_and_the_next_nodes_reply():
    llm = ContextCapturingMockLLM(
        mock_steps=[MockLLMService.create_text_chunks(GREETING)], chunk_delay=0.001
    )
    engine, tts, context = await _run_call(
        screening_graph(), llm, until=_spoke(NOTICE, "Hi, this is Emma at Mirror Labs.")
    )

    # One generation, and it was the next node's.
    assert len(llm.captured_contexts) == 1
    assert MAIN_PROMPT in llm.get_system_prompt_at_step(0)
    assert START_PROMPT not in llm.get_system_prompt_at_step(0)
    # It read the history a model-made transition leaves.
    messages = llm.get_context_at_step(0)["messages"]
    call = next(m for m in messages if m.get("tool_calls"))
    assert call["tool_calls"][0]["function"]["name"] == "begin"
    result = next(m for m in messages if m.get("role") == "tool")
    assert result["tool_call_id"] == call["tool_calls"][0]["id"]
    assert result["content"] == '{"status": "done"}'
    # The notice first, then the reply.
    assert tts.received_texts[0] == NOTICE
    assert tts.received_texts[1:3] == ["Hi, this is Emma at Mirror Labs.", "How can I help?"]
    assert engine.active_agent.current_node.id == "harkbell-main"
    # The notice is spoken, never part of the conversation.
    assert all(
        NOTICE not in str(m.get("content") or "")
        for m in context.messages
        if isinstance(m, dict) and m.get("role") == "assistant"
    )


@pytest.mark.asyncio
async def test_a_recorded_notice_plays_from_its_recording():
    fetched = []

    async def fetch_recording(*, recording_pk=None, recording_id=None):
        fetched.append(recording_pk)
        return SimpleNamespace(audio=b"\x01\x00" * 1600, transcript=NOTICE)

    llm = ContextCapturingMockLLM(
        mock_steps=[MockLLMService.create_text_chunks(GREETING)], chunk_delay=0.001
    )
    engine, tts, _context = await _run_call(
        screening_graph(notice_audio=True),
        llm,
        fetch_recording=fetch_recording,
        until=_spoke("How can I help?"),
    )
    assert fetched == [7]
    assert NOTICE not in tts.received_texts
    assert len(llm.captured_contexts) == 1
    assert MAIN_PROMPT in llm.get_system_prompt_at_step(0)
    assert engine.active_agent.current_node.id == "harkbell-main"


@pytest.mark.asyncio
async def test_without_the_flag_the_model_still_takes_the_pathway():
    llm = ContextCapturingMockLLM(
        mock_steps=[
            MockLLMService.create_function_call_chunks(
                function_name="begin", arguments={}, tool_call_id="call_begin"
            ),
            MockLLMService.create_text_chunks(GREETING),
        ],
        chunk_delay=0.001,
    )
    engine, tts, _context = await _run_call(
        screening_graph(flag=False), llm, until=_spoke("How can I help?")
    )
    assert len(llm.captured_contexts) == 2
    assert START_PROMPT in llm.get_system_prompt_at_step(0)
    assert MAIN_PROMPT in llm.get_system_prompt_at_step(1)
    assert tts.received_texts[0] == NOTICE
    assert engine.active_agent.current_node.id == "harkbell-main"


# ── a text chat ───────────────────────────────────────────────────────────


async def _text_chat_opening(graph: WorkflowGraph, llm):
    """The opening of a text chat, wired as text_chat_runner wires it."""
    context = LLMContext()
    engine = PipecatEngine(
        llm=llm,
        context=context,
        workflow=graph,
        call_context_vars={},
        workflow_run_id=1,
        run_transition_variable_extraction_in_background=False,
    )
    window = _ResponseWindowState()
    capture = _TextChatCaptureProcessor(window, context, engine)
    _user, assistant = LLMContextAggregatorPair(
        context, assistant_params=LLMAssistantAggregatorParams()
    )

    @assistant.event_handler("on_assistant_turn_started")
    async def started(_aggregator):
        window.note_assistant_turn_started()

    @assistant.event_handler("on_assistant_turn_stopped")
    async def stopped(_aggregator, message):
        window.note_assistant_turn_stopped(message.content or "")

    task = PipelineWorker(Pipeline([llm, capture, assistant]), enable_rtvi=False)
    runner = asyncio.create_task(run_pipeline_worker(task))
    engine.call_worker = task
    engine.set_transport_output(_TaskQueueProxy(task.queue_frame))
    ready = asyncio.Event()

    @task.event_handler("on_pipeline_started")
    async def pipeline_started(*_):
        ready.set()

    try:
        await asyncio.wait_for(ready.wait(), 3)
        with patch(
            "api.db:db_client.get_organization_id_by_workflow_run_id",
            new_callable=AsyncMock,
            return_value=1,
        ):
            await engine.initialize()
        await engine.set_node(graph.start_node_id)
        engine.call_monitor.activate()
        marker = capture.activity_count
        # The start node has no greeting, so the runner expects the model.
        window.note_direct_context_request()
        opening = await engine.queue_node_opening(
            node_id=graph.start_node_id,
            previous_node_id=None,
            generate_if_no_greeting=True,
        )
        if opening.action != "llm":
            # As the runner does when the opening asked the model for nothing.
            window.pending_context_requests = max(0, window.pending_context_requests - 1)
        await _wait_for_quiescence(
            capture_processor=capture,
            response_window=window,
            runner_task=runner,
            activity_marker=marker,
            timeout_seconds=5,
        )
    finally:
        await task.cancel()
        await asyncio.wait_for(runner, 5)
    return engine, opening, window


@pytest.mark.asyncio
async def test_a_text_chat_opens_with_the_notice_then_the_next_nodes_greeting():
    llm = ContextCapturingMockLLM(
        mock_steps=[MockLLMService.create_text_chunks(GREETING)], chunk_delay=0.001
    )
    engine, opening, window = await _text_chat_opening(screening_graph(), llm)
    assert opening.action == "llm"
    assert window.outputs == [NOTICE, GREETING]
    assert len(llm.captured_contexts) == 1
    assert MAIN_PROMPT in llm.get_system_prompt_at_step(0)
    assert engine.active_agent.current_node.id == "harkbell-main"


# ── seeing a slow minute ──────────────────────────────────────────────────


def test_a_slow_first_token_is_a_warning():
    from loguru import logger

    seen = []
    sink = logger.add(lambda message: seen.append(str(message)), level="WARNING")
    try:
        assert warn_if_slow_llm_first_token(
            kind="llm", ttfb_seconds=8.4, processor="LLM#1", model="minimax-m3"
        )
        assert not warn_if_slow_llm_first_token(
            kind="llm", ttfb_seconds=2.9, processor="LLM#1", model="minimax-m3"
        )
        assert not warn_if_slow_llm_first_token(
            kind="tts", ttfb_seconds=8.4, processor="TTS#1", model=None
        )
    finally:
        logger.remove(sink)
    assert len(seen) == 1
    assert "Slow language model first token: 8400 ms" in seen[0]
