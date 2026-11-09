from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.services.pipecat.minimax_llm import HarkbellMiniMaxLLMService

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
    service = _service()
    service._push_llm_text = AsyncMock()
    async for _ in service._drop_tool_markup(_stream(["<"])):
        pass
    service._push_llm_text.assert_awaited_once_with("<")


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
