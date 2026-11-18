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


# ── A greeting with a variable in it, and a recording for each way it comes out ──────────
# Copy runs 5758 and 5761 (2026-11-18): the keypad sentence makes the greeting different
# per call, so it could not be recorded, and the realtime model was asked to say it as
# written. The smaller model dropped the key sentence once and thought aloud in front of
# it once in eight calls. A node may now carry a recording per rendering.

TEMPLATE = "{{initial_context.keypad_announcement}}Hi, this is Emma at Mirror Labs. How can I help you?"
PLAIN = "Hi, this is Emma at Mirror Labs. How can I help you?"
KEYED = "For events in the next 48 hours, press 1. " + PLAIN
RECORDINGS = [
    {"text": KEYED, "recording_id": "11"},
    {"text": PLAIN, "recording_id": "12"},
]


def _graph_with_recordings(recordings=RECORDINGS, greeting=TEMPLATE):
    definition = screening_definition(main_greeting=greeting)
    main = next(n for n in definition["nodes"] if n["id"] == "harkbell-main")
    main["data"]["greeting_recordings"] = recordings
    return WorkflowGraph(ReactFlowDTO.model_validate(definition))


def _engine_for(graph, *, announcement=None, audio=True):
    from pipecat.processors.aggregators.llm_context import LLMContext

    from api.services.workflow.pipecat_engine import PipecatEngine

    engine = PipecatEngine(
        llm=_unused_model(),
        context=LLMContext(),
        workflow=graph,
        call_context_vars=(
            {"keypad_announcement": announcement} if announcement is not None else {}
        ),
        workflow_run_id=1,
    )
    if audio:

        async def fetch_recording(*, recording_pk=None, recording_id=None):
            return None

        engine.set_fetch_recording_audio(fetch_recording)
        engine._transport_output = object()
    return engine


def test_the_recording_of_this_calls_greeting_is_the_one_that_plays():
    graph = _graph_with_recordings()
    in_hours = _engine_for(graph, announcement="For events in the next 48 hours, press 1. ")
    assert in_hours.get_node_greeting("harkbell-main") == ("audio", "11")
    # No announcement on this call -- out of hours, or a website call.
    assert _engine_for(graph).get_node_greeting("harkbell-main") == ("audio", "12")
    assert _engine_for(graph, announcement="").get_node_greeting("harkbell-main") == ("audio", "12")
    # Spaces are the template's, not the words': a doubled or missing one still matches.
    spaced = _engine_for(graph, announcement="For events in the next 48 hours,  press 1.  ")
    assert spaced.get_node_greeting("harkbell-main") == ("audio", "11")


def test_a_greeting_no_recording_was_made_of_is_spoken_as_before():
    graph = _graph_with_recordings()
    changed = _engine_for(graph, announcement="Press 9 for the box office. ")
    assert changed.get_node_greeting("harkbell-main") == (
        "text",
        "Press 9 for the box office. " + PLAIN,
    )
    # No list, an empty one, or entries that are not usable: the text, never a guess.
    # (An entry that is not an object at all is refused when the workflow is saved.)
    for recordings in (None, [], [{"text": PLAIN}], [{"recording_id": "12"}]):
        engine = _engine_for(_graph_with_recordings(recordings=recordings))
        assert engine.get_node_greeting("harkbell-main") == ("text", PLAIN), recordings


def test_a_text_chat_is_given_the_words_not_a_recording():
    engine = _engine_for(_graph_with_recordings(), audio=False)
    assert engine.get_node_greeting("harkbell-main") == ("text", PLAIN)


def test_a_configured_recording_still_wins_over_the_list():
    definition = screening_definition(main_greeting=TEMPLATE)
    main = next(n for n in definition["nodes"] if n["id"] == "harkbell-main")
    main["data"] |= {
        "greeting_type": "audio",
        "greeting_recording_id": "40",
        "greeting_recordings": RECORDINGS,
    }
    engine = _engine_for(WorkflowGraph(ReactFlowDTO.model_validate(definition)))
    assert engine.get_node_greeting("harkbell-main") == ("audio", "40")


@pytest.mark.asyncio
async def test_on_a_call_the_recording_plays_and_nothing_is_synthesised_for_the_greeting():
    from types import SimpleNamespace

    asked = []

    async def fetch_recording(*, recording_pk=None, recording_id=None):
        asked.append(recording_pk if recording_pk is not None else recording_id)
        return SimpleNamespace(audio=b"\x01\x00" * 3200, transcript=PLAIN)

    llm = _unused_model()
    engine, tts, context = await _run_call(
        _graph_with_recordings(),
        llm,
        fetch_recording=fetch_recording,
        until=lambda llm, tts: 12 in asked,
    )
    # The notice is this fixture's text; the greeting came from recording 12.
    assert tts.received_texts == [NOTICE]
    assert asked == [12]
    assert llm.captured_contexts == []
    assert [m["content"] for m in context.messages if m.get("role") == "assistant"] == [PLAIN]
