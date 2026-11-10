import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.services.pipecat.minimax_llm import HarkbellMiniMaxLLMService, written_json_calls
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.minimax.llm import MiniMaxLLMService

# Harkbell round t656b, g3: the reply to "I wanna buy a gift card", repeated on
# every turn after it.
G3 = '<function_calls>\n<invoke name="select_tool">\n<parameter name="tool">question</parameter>\n</invoke>\n</function_calls>'


def _service() -> HarkbellMiniMaxLLMService:
    with patch.object(HarkbellMiniMaxLLMService, "create_client"):
        service = HarkbellMiniMaxLLMService(api_key="test-key")
    service.push_frame = AsyncMock()
    return service


def _chunk(content):
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


async def _stream(pieces):
    for piece in pieces:
        yield _chunk(piece)


async def _visible(pieces: list[str]) -> str:
    service = _service()
    out = []
    async for chunk in service._drop_tool_markup(_stream(pieces)):
        if chunk.choices[0].delta.content:
            out.append(chunk.choices[0].delta.content)
    return "".join(out)


@pytest.mark.asyncio
@pytest.mark.parametrize("pieces", [[G3], ["<", "function_", "calls>\n<invoke", ' name="x">'], [" \n<tool_call>"], ["<invoke name=\"end_call\">"]])
async def test_a_reply_that_is_a_written_tool_call_says_nothing(pieces):
    assert await _visible(pieces) == ""


@pytest.mark.asyncio
async def test_prose_before_markup_is_kept_and_the_rest_dropped():
    assert await _visible(["Sure, one moment. ", "<function_calls>", "<invoke name=\"question\">"]) == "Sure, one moment. "
    assert await _visible(["Let me check.<tool_call>{}"]) == "Let me check."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pieces",
    [["We sell gift cards ", "from fifty dollars."], ["[warmly] ", "Got it."], ["<", "3 you too"], ["Anything < 2 inches ", "is the minimum."]],
)
async def test_ordinary_replies_pass_untouched(pieces):
    assert await _visible(pieces) == "".join(pieces)


@pytest.mark.asyncio
async def test_a_reply_that_ends_on_a_lone_angle_bracket_is_still_said():
    assert await _visible(["<"]) == "<"


@pytest.mark.asyncio
async def test_each_response_starts_fresh():
    service = _service()
    async for _ in service._drop_tool_markup(_stream([G3])):
        pass
    service._reset_markup_state()
    out = []
    async for chunk in service._drop_tool_markup(_stream(["Sorry about that — ", "a gift card, sure."])):
        if chunk.choices[0].delta.content:
            out.append(chunk.choices[0].delta.content)
    assert "".join(out) == "Sorry about that — a gift card, sure."


@pytest.mark.asyncio
@pytest.mark.parametrize("pieces", [['{"name": "say_goodbye", "arguments": {}}'], ["{", '"name', '": "say_goodbye"', ', "arguments": {}}'], ['{ "name": "end_call", "arguments": {}}']])
async def test_a_tool_call_written_as_json_says_nothing(pieces):
    assert await _visible(pieces) == ""


@pytest.mark.asyncio
async def test_a_reply_that_opens_with_a_brace_but_is_prose_is_said():
    assert await _visible(["{", "laughs} okay"]) == "{laughs} okay"


# Production's twin of Mirror Labs, run 3881 (Harkbell T-663): the reply to
# "Tomorrow at seven thirty." was this, and the caller got "```json ```".
ROUTE_ARGS = {"order_number": "55120933", "event_date_as_said": "Tomorrow at seven thirty"}
T663 = ["```", "json", "\n", "{", '"name": "route_call", "arguments": ', '{"order_number": "55120933", ', '"event_date_as_said": "Tomorrow at seven thirty"}}', "\n```"]


def _context(*names: str) -> LLMContext:
    return LLMContext(
        messages=[],
        tools=ToolsSchema(
            standard_tools=[FunctionSchema(name=n, description=n, properties={}, required=[]) for n in names]
        ),
    )


async def _run(pieces: list[str], *, registered=("route_call", "end_call"), context=None, structured=False):
    service = _service()
    for name in registered:
        service.register_function(name, AsyncMock())
    if context is None:
        context = _context(*registered)
    visible, calls = [], []

    async def stream():
        for piece in pieces:
            yield _chunk(piece)
        if structured:
            chunk = _chunk(None)
            chunk.choices[0].delta.tool_calls = [MagicMock()]
            yield chunk

    async for chunk in service._drop_tool_markup(stream(), context):
        delta = chunk.choices[0].delta
        if delta.content:
            visible.append(delta.content)
        if delta.tool_calls and not structured:
            call = delta.tool_calls[0]
            calls.append((call.index, call.function.name, json.loads(call.function.arguments), call.id))
    return "".join(visible), calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pieces",
    [
        T663,
        ["```json\n", '{"name": "route_call"', ', "arguments": {"order_number": "55120933", "event_date_as_said": "Tomorrow at seven thirty"}}\n```'],
        ["```json\n{\n  ", '"name": "route_call",\n  "arguments": {"order_number": "55120933", "event_date_as_said": "Tomorrow at seven thirty"}\n}\n```'],
        ['{"name": "route_call", "arguments": "{\\"order_number\\": \\"55120933\\", \\"event_date_as_said\\": \\"Tomorrow at seven thirty\\"}"}'],
    ],
)
async def test_a_fenced_written_call_says_nothing_and_runs_as_the_call(pieces):
    visible, calls = await _run(pieces)
    assert visible == ""
    assert [(index, name, args) for index, name, args, _ in calls] == [(0, "route_call", ROUTE_ARGS)]
    assert calls[0][3].startswith("call_")


@pytest.mark.asyncio
async def test_a_written_call_to_a_tool_this_turn_was_not_offered_only_says_nothing():
    assert await _run(T663, context=_context("end_call")) == ("", [])
    assert await _run(T663, registered=("end_call",)) == ("", [])


@pytest.mark.asyncio
async def test_a_written_call_after_words_is_dropped_and_not_run():
    # Staging run 3831 (T-662): reasoning in words, then a route_call made of
    # the tool's own examples. Words before the call mean it is not run.
    visible, calls = await _run(["Wait — the caller is Priya. ", "```json\n", '{"name": "route_call", "arguments": {"order_number": "VG-55120"}}', "\n```"])
    assert visible == "Wait — the caller is Priya. "
    assert calls == []


@pytest.mark.asyncio
async def test_two_different_written_calls_run_neither():
    two = ['{"name": "route_call", "arguments": {"order_number": "1"}}\n{"name": "route_call", "arguments": {"order_number": "2"}}']
    assert await _run(two) == ("", [])


@pytest.mark.asyncio
async def test_the_same_written_call_twice_runs_once():
    twice = ['{"name": "end_call", "arguments": {}}{"name": "end_call", "arguments": {}}']
    visible, calls = await _run(twice)
    assert (visible, [(name, args) for _, name, args, _ in calls]) == ("", [("end_call", {})])


@pytest.mark.asyncio
async def test_a_written_call_beside_a_structured_one_is_not_run_again():
    visible, calls = await _run(['{"name": "end_call", "arguments": {}}'], structured=True)
    assert (visible, calls) == ("", [])


@pytest.mark.asyncio
async def test_an_xml_written_call_is_dropped_and_not_run():
    assert await _run([G3]) == ("", [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pieces",
    [
        ["Sure. ", "<", "function_calls>\n<invoke"],
        ["Sure. <func", "tion_calls>"],
        ["Sure. {", '"name": "end_call", "arguments": {}}'],
        ["Sure. ", "`", "``json\n{}"],
    ],
)
async def test_an_opener_split_across_chunks_after_words_is_still_caught(pieces):
    assert await _visible(pieces) == "Sure. "


@pytest.mark.asyncio
async def test_words_inside_a_fence_are_said_without_it():
    visible = await _visible(["```\n", "Sure, the event ", "is tomorrow.\n```"])
    assert visible.strip() == "Sure, the event is tomorrow."


@pytest.mark.asyncio
@pytest.mark.parametrize("pieces", [["```"], ["```json"], ["`"], ["```json\n\n```"]])
async def test_a_fence_with_nothing_in_it_says_nothing(pieces):
    assert await _visible(pieces) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("pieces", [["Okay {", "laughs} sure"], ["Call me at <", "3pm"], ["A {big} one"]])
async def test_a_held_tail_that_was_not_an_opener_is_said(pieces):
    assert await _visible(pieces) == "".join(pieces)


def test_written_json_calls_reads_what_minimax_writes():
    assert written_json_calls('```json\n{"name": "x", "parameters": {"a": 1}}\n```') == [("x", {"a": 1})]
    assert written_json_calls('{"arguments": {"a": 1}, "name": "x"}') == [("x", {"a": 1})]
    assert written_json_calls('{"name": "x", "arguments": ""}') == [("x", {})]
    assert written_json_calls('{"name": "x", "arguments": {"a": ') == []
    assert written_json_calls('{"name": 3, "arguments": {}}') == []
    assert written_json_calls("no call here {}") == []


@pytest.mark.asyncio
async def test_pipecat_runs_the_written_call_through_its_own_tool_path():
    service = _service()
    service.register_function("route_call", AsyncMock())
    service._run_or_defer_function_calls = AsyncMock()

    async def completions(self, context):
        return _stream(T663)

    with patch.object(MiniMaxLLMService, "get_chat_completions", completions):
        await service._process_context(_context("route_call"))

    calls = service._run_or_defer_function_calls.await_args.args[0]
    assert [(c.function_name, c.arguments) for c in calls] == [("route_call", ROUTE_ARGS)]
    assert service._run_or_defer_function_calls.await_args.kwargs["text_generated"] is False


# Harkbell T-667: a response stream that says nothing (copy run 4008: headers,
# then no token for a minute) is asked again; one that goes quiet mid-reply ends.


class _Stream:
    """A fake response stream: (delay, content) pairs, then the end."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.steps:
            raise StopAsyncIteration
        delay, content = self.steps.pop(0)
        await asyncio.sleep(delay)
        return _chunk(content)

    async def aclose(self):
        self.closed = True


@pytest.fixture
def fast_stalls(monkeypatch):
    import api.services.pipecat.minimax_llm as module

    monkeypatch.setattr(module, "FIRST_OUTPUT_SECS", 0.05)
    monkeypatch.setattr(module, "IDLE_SECS", 0.05)
    monkeypatch.setattr(module, "STALL_RETRIES", 2)


async def _through_guard(streams):
    service = _service()
    handed = []

    async def completions(self, context):
        stream = streams.pop(0)
        handed.append(stream)
        if isinstance(stream, float):
            await asyncio.sleep(stream)  # the request itself hangs
        return stream

    out = []
    with patch.object(MiniMaxLLMService, "get_chat_completions", completions):
        async for chunk in service._unstalled(_context()):
            if chunk.choices[0].delta.content:
                out.append(chunk.choices[0].delta.content)
    return "".join(out), handed


@pytest.mark.asyncio
async def test_a_stream_that_says_nothing_is_asked_again(fast_stalls):
    stalled = _Stream([(0.0, None), (1.0, "never")])
    text, handed = await _through_guard([stalled, _Stream([(0.0, "Sure, "), (0.0, "one moment.")])])
    assert text == "Sure, one moment."
    assert len(handed) == 2 and stalled.closed


@pytest.mark.asyncio
async def test_a_request_that_never_answers_is_asked_again(fast_stalls):
    text, handed = await _through_guard([1.0, _Stream([(0.0, "Hello.")])])
    assert text == "Hello." and len(handed) == 2


@pytest.mark.asyncio
async def test_after_two_stalls_the_last_attempt_waits_as_before(fast_stalls):
    slow = _Stream([(0.2, "Slow, "), (0.0, "but here.")])
    text, handed = await _through_guard([_Stream([(1.0, "x")]), _Stream([(1.0, "y")]), slow])
    assert text == "Slow, but here." and len(handed) == 3


@pytest.mark.asyncio
async def test_a_stream_that_goes_quiet_mid_reply_ends_with_what_it_said(fast_stalls):
    quiet = _Stream([(0.0, "The team will "), (1.0, "never arrives")])
    text, handed = await _through_guard([quiet, _Stream([(0.0, "a second answer")])])
    assert text == "The team will " and len(handed) == 1 and quiet.closed


@pytest.mark.asyncio
async def test_an_ordinary_stream_passes_through_once_and_is_closed(fast_stalls):
    ok = _Stream([(0.01, "We open "), (0.01, "at ten.")])
    text, handed = await _through_guard([ok])
    assert text == "We open at ten." and len(handed) == 1 and ok.closed
