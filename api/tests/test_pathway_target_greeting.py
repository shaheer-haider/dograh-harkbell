"""Production run 4615: a step entered by a start pathway opens with its own greeting.

On ``-harkbell.16`` the screening main step's first turn was the model's, and
the model read the pathway's tool call and ``{"status": "done"}`` as a check
to report on -- "No issues found with the recording line notice." -- and then
said goodbye to a caller who had not yet said a word. A step with a greeting
of its own now says it after the notice, and the model waits for the caller.
"""

import pytest
from pipecat.tests.mock_llm_service import ContextCapturingMockLLM, MockLLMService

from api.services.workflow.dto import ReactFlowDTO
from api.services.workflow.workflow_graph import WorkflowGraph
from api.tests.test_take_pathway_without_model import (
    NOTICE,
    _run_call,
    _text_chat_opening,
    screening_definition,
)

# Config 192's shape: the pre-call fetch may fill the keypad line, or not.
MAIN_GREETING = "Hi, this is Emma at Mirror Labs. {{keypad_announcement}} How can I help you?"
SPOKEN = "Hi, this is Emma at Mirror Labs.  How can I help you?"


def _graph(**kwargs):
    return WorkflowGraph(ReactFlowDTO.model_validate(screening_definition(**kwargs)))


def _unused_model():
    return ContextCapturingMockLLM(
        mock_steps=[MockLLMService.create_text_chunks("No issues found with the recording line notice.")],
        chunk_delay=0.001,
    )


@pytest.mark.asyncio
async def test_run_4615_the_step_greets_and_the_model_waits_for_the_caller():
    llm = _unused_model()
    engine, tts, context = await _run_call(
        _graph(main_greeting=MAIN_GREETING),
        llm,
        until=lambda llm, tts: SPOKEN in tts.received_texts,
    )
    assert tts.received_texts == [NOTICE, SPOKEN]
    assert llm.captured_contexts == []
    assert engine.active_agent.current_node.id == "harkbell-main"
    # Nothing for a model to report on, and the greeting as said.
    assert not any(isinstance(m, dict) and m.get("tool_calls") for m in context.messages)
    assert [m["content"] for m in context.messages if m.get("role") == "assistant"] == [SPOKEN]
    # The greeting is the call's greeting to everything that knows one.
    assert engine.speech_playback.greeting is not None
    assert engine.speech_playback.greeting.text == SPOKEN


@pytest.mark.asyncio
async def test_a_recorded_notice_is_followed_by_the_greeting():
    from types import SimpleNamespace

    async def fetch_recording(*, recording_pk=None, recording_id=None):
        return SimpleNamespace(audio=b"\x01\x00" * 3200, transcript=NOTICE)

    llm = _unused_model()
    engine, tts, _context = await _run_call(
        _graph(main_greeting=MAIN_GREETING, notice_audio=True),
        llm,
        fetch_recording=fetch_recording,
        until=lambda llm, tts: SPOKEN in tts.received_texts,
    )
    assert tts.received_texts == [SPOKEN]
    assert llm.captured_contexts == []


@pytest.mark.asyncio
async def test_without_a_greeting_the_model_still_opens_the_step():
    llm = ContextCapturingMockLLM(
        mock_steps=[MockLLMService.create_text_chunks("Hi, how can I help?")],
        chunk_delay=0.001,
    )
    _engine, tts, _context = await _run_call(
        _graph(), llm, until=lambda llm, tts: "Hi, how can I help?" in tts.received_texts
    )
    assert len(llm.captured_contexts) == 1
    messages = llm.get_context_at_step(0)["messages"]
    assert any(m.get("tool_calls") for m in messages)
    assert tts.received_texts == [NOTICE, "Hi, how can I help?"]


@pytest.mark.asyncio
async def test_a_text_chat_opens_with_the_notice_then_the_steps_greeting():
    llm = _unused_model()
    engine, opening, window = await _text_chat_opening(_graph(main_greeting=MAIN_GREETING), llm)
    assert opening.action == "greeting"
    assert window.outputs == [NOTICE, SPOKEN]
    assert llm.captured_contexts == []
    assert engine.active_agent.current_node.id == "harkbell-main"
