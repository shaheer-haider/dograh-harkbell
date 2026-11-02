"""A tool call the model writes as text is run or dropped, never spoken.

The case these come from: a staging call where MiniMax answered "yes, this
number is fine" by putting ``{"destination": "assistant", "tool":
"leave_a_message", "params": {...}}`` in the content channel, and the caller
heard it read aloud. The LLM here is Pipecat's OpenAI-shaped mock streaming
exactly that text, so the recovered call goes through the real service's
dispatcher rather than a stand-in.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext

from api.services.pipecat.spoken_tool_call_filter import (
    SpokenToolCallFilter,
    tool_call_from_object,
)
from pipecat.tests import MockLLMService, run_test
from pipecat.tests.utils import SleepFrame

LEAKED_CALL = json.dumps(
    {
        "destination": "assistant",
        "tool": "leave_a_message",
        "params": {
            "customer_name": "Sophie Turner",
            "reason": "Wants a balayage {or highlights}, saw it on Instagram.",
            "customer_phone": "",
        },
    }
)


def _schema(name: str) -> FunctionSchema:
    return FunctionSchema(
        name=name, description=f"{name} tool", properties={}, required=[]
    )


def _context(*tool_names: str) -> LLMContext:
    return LLMContext(
        messages=[{"role": "user", "content": "Yeah, this one's fine."}],
        tools=ToolsSchema(standard_tools=[_schema(name) for name in tool_names]),
    )


class _Calls:
    def __init__(self):
        self.received = []

    async def __call__(self, params):
        self.received.append((params.function_name, dict(params.arguments)))
        await params.result_callback({"status": "done"})


def _one_turn(context: LLMContext) -> list:
    # The pause keeps the test's EndFrame behind the response: an EndFrame
    # reaching the LLM first discards a deferred transition on its own.
    return [LLMContextFrame(context), SleepFrame(sleep=0.3)]


def _spoken(frames) -> str:
    return "".join(f.text for f in frames if isinstance(f, LLMTextFrame))


async def _run_llm(text: str, *, chunk_size: int, advertised=("leave_a_message",)):
    """Stream ``text`` from a mock LLM through the filter."""
    llm = MockLLMService(
        mock_chunks=MockLLMService.create_text_chunks(text, chunk_size=chunk_size),
        chunk_delay=0.001,
    )
    handler = _Calls()
    llm.register_function("leave_a_message", handler)
    context = _context(*advertised)
    pipeline = Pipeline(
        [llm, SpokenToolCallFilter(llm=llm, context_provider=lambda: context)]
    )
    down, _ = await run_test(pipeline, frames_to_send=_one_turn(context))
    return down, handler


@pytest.mark.asyncio
async def test_a_bare_tool_call_is_run_and_not_spoken():
    down, handler = await _run_llm(LEAKED_CALL, chunk_size=len(LEAKED_CALL))

    assert _spoken(down) == ""
    assert handler.received == [
        (
            "leave_a_message",
            {
                "customer_name": "Sophie Turner",
                "reason": "Wants a balayage {or highlights}, saw it on Instagram.",
                "customer_phone": "",
            },
        )
    ]
    results = [f for f in down if isinstance(f, FunctionCallResultFrame)]
    assert [r.function_name for r in results] == ["leave_a_message"]
    assert results[0].tool_call_id.startswith("call_spoken_")


@pytest.mark.asyncio
async def test_a_tool_call_after_a_sentence_keeps_the_sentence():
    down, handler = await _run_llm(
        f"Perfect, I'll pass that on. {LEAKED_CALL}", chunk_size=1000
    )

    assert _spoken(down) == "Perfect, I'll pass that on. "
    assert [name for name, _ in handler.received] == ["leave_a_message"]


@pytest.mark.asyncio
async def test_a_tool_call_split_across_streamed_chunks_is_caught():
    text = f"Thanks, Sophie.\n{LEAKED_CALL}\nAnything else?"
    for chunk_size in (1, 3, 7, 16):
        down, handler = await _run_llm(text, chunk_size=chunk_size)

        assert _spoken(down) == "Thanks, Sophie.\n\nAnything else?", chunk_size
        assert "destination" not in _spoken(down)
        assert [name for name, _ in handler.received] == ["leave_a_message"]


@pytest.mark.asyncio
async def test_speech_with_braces_passes_untouched():
    for text in (
        "Pick a slot {morning or evening} and I'll hold it.",
        'She said {"name": "Sophie"} on the form, so I used that.',
        "Your reference ends in a brace {",
        'It literally says {"',
        "{ } and {} are not calls.",
    ):
        down, handler = await _run_llm(text, chunk_size=4)
        assert _spoken(down) == text
        assert handler.received == []


@pytest.mark.asyncio
async def test_a_tool_the_node_does_not_offer_is_dropped_not_run():
    down, handler = await _run_llm(LEAKED_CALL, chunk_size=9, advertised=("end_call",))

    assert _spoken(down) == ""
    assert handler.received == []
    assert not [f for f in down if isinstance(f, FunctionCallResultFrame)]


@pytest.mark.asyncio
async def test_a_spoken_call_beside_a_native_one_is_dropped():
    llm = MockLLMService(
        mock_chunks=MockLLMService.create_mixed_chunks(
            LEAKED_CALL, "end_call", {}, tool_call_id="call_native"
        ),
        chunk_delay=0.001,
    )
    spoken_handler, native_handler = _Calls(), _Calls()
    llm.register_function("leave_a_message", spoken_handler)
    llm.register_function("end_call", native_handler)
    context = _context("leave_a_message", "end_call")
    pipeline = Pipeline(
        [llm, SpokenToolCallFilter(llm=llm, context_provider=lambda: context)]
    )

    down, _ = await run_test(pipeline, frames_to_send=_one_turn(context))

    assert _spoken(down) == ""
    assert spoken_handler.received == []
    assert [name for name, _ in native_handler.received] == ["end_call"]


@pytest.mark.asyncio
async def test_a_transition_deferred_for_removed_text_still_runs():
    # The service counts the removed JSON as speech and defers a lone
    # transition until the bot stops speaking. Nothing reached the TTS, so
    # without the filter releasing it the call would sit on this node forever.
    llm = MockLLMService(
        mock_chunks=MockLLMService.create_mixed_chunks(
            LEAKED_CALL, "go_to_messages", {}, tool_call_id="call_transition"
        ),
        chunk_delay=0.001,
    )
    transition = _Calls()
    llm.register_function("go_to_messages", transition, is_node_transition=True)
    context = _context("go_to_messages")
    pipeline = Pipeline(
        [llm, SpokenToolCallFilter(llm=llm, context_provider=lambda: context)]
    )

    down, _ = await run_test(pipeline, frames_to_send=_one_turn(context))

    assert _spoken(down) == ""
    assert [name for name, _ in transition.received] == ["go_to_messages"]


@pytest.mark.asyncio
async def test_an_interruption_discards_a_held_call():
    dispatch = AsyncMock()
    llm = SimpleNamespace(
        _functions={"leave_a_message": object()},
        _run_or_defer_function_calls=dispatch,
        _tool_response_generation=1,
    )
    context = _context("leave_a_message")
    flt = SpokenToolCallFilter(llm=llm, context_provider=lambda: context)

    half = len(LEAKED_CALL) // 2
    down, _ = await run_test(
        flt,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame(LEAKED_CALL[:half]),
            # An interruption is a system frame and would overtake the text
            # without the pause.
            SleepFrame(sleep=0.1),
            InterruptionFrame(),
            SleepFrame(sleep=0.1),
            LLMFullResponseEndFrame(),
        ],
    )

    assert _spoken(down) == ""
    dispatch.assert_not_awaited()
    assert flt.holding == ""


@pytest.mark.asyncio
async def test_a_recovered_call_waits_for_the_speech_before_it():
    # Handed to the service's own dispatcher with text_generated set, so a lone
    # transition is deferred until "One moment." has finished playing.
    dispatch = AsyncMock()
    llm = SimpleNamespace(
        _functions={"leave_a_message": object()},
        _run_or_defer_function_calls=dispatch,
        _tool_response_generation=7,
    )
    context = _context("leave_a_message")
    flt = SpokenToolCallFilter(llm=llm, context_provider=lambda: context)

    down, _ = await run_test(
        flt,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("One moment. "),
            LLMTextFrame(LEAKED_CALL),
            LLMFullResponseEndFrame(),
        ],
    )

    assert _spoken(down) == "One moment. "
    dispatch.assert_awaited_once()
    (calls,), kwargs = dispatch.await_args
    assert [c.function_name for c in calls] == ["leave_a_message"]
    assert calls[0].context is context
    assert kwargs == {"text_generated": True, "response_generation": 7}


@pytest.mark.asyncio
async def test_without_a_dispatcher_the_call_is_only_removed():
    flt = SpokenToolCallFilter()
    down, _ = await run_test(
        flt,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("One moment. "),
            LLMTextFrame(LEAKED_CALL),
            LLMFullResponseEndFrame(),
        ],
    )
    assert _spoken(down) == "One moment. "


@pytest.mark.asyncio
async def test_an_unfinished_call_at_the_end_of_a_response_is_not_spoken():
    flt = SpokenToolCallFilter()
    down, _ = await run_test(
        flt,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Sure. "),
            LLMTextFrame(LEAKED_CALL[:40]),
            LLMFullResponseEndFrame(),
        ],
    )
    assert _spoken(down) == "Sure. "


def test_the_call_shapes_models_write():
    assert tool_call_from_object(json.loads(LEAKED_CALL))[0] == "leave_a_message"
    assert tool_call_from_object(
        {"name": "book", "arguments": '{"day": "Friday"}'}
    ) == ("book", {"day": "Friday"})
    assert tool_call_from_object(
        {"type": "function", "function": {"name": "book", "arguments": {}}}
    ) == ("book", {})
    assert tool_call_from_object({"tool": "end_call"}) == ("end_call", {})
    # A name with nothing to call is a person, not a call.
    assert tool_call_from_object({"name": "Sophie"}) is None
    assert tool_call_from_object({"tool": ""}) is None
    assert tool_call_from_object(["tool"]) is None
