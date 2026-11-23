"""A reply to an answer that asks for no words says none beside its tool call.

Copy run 7251 (2026-11-23): route_call put an urgent order through, its answer
said to call transfer_call and say nothing, and the model said a sentence of
its own first -- the caller waited seven seconds for the transfer.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.minimax.llm import MiniMaxLLMService

from api.services.pipecat.minimax_llm import (
    HarkbellMiniMaxLLMService,
    _content_chunk,
    _tool_call_chunk,
    answers_a_silent_reply,
)

SAID = "Thanks. So the buyer can't get in, and the event's tonight at eight — let me get that to the right place."


def _context(silent: bool) -> LLMContext:
    context = LLMContext()
    context.add_message({"role": "user", "content": "The buyer can't get in, the event is tonight at 8."})
    context.add_message(
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "route_call", "arguments": "{}"}}],
        }
    )
    data = {"result": "routed", "route": "transfer_and_email", "instructions": "Call transfer_call now."}
    if silent:
        data["silent_reply"] = True
    context.add_message(
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"status": "success", "data": data})}
    )
    return context


def _service() -> HarkbellMiniMaxLLMService:
    with patch.object(HarkbellMiniMaxLLMService, "create_client"):
        service = HarkbellMiniMaxLLMService(api_key="test-key")
    service.push_frame = AsyncMock()
    service.run_function_calls = AsyncMock()
    return service


async def _said(context, text: str, *, call: bool) -> str:
    service = _service()

    async def completions(self, context):
        async def stream():
            for i in range(0, len(text), 7):
                yield _content_chunk(text[i : i + 7])
            if call:
                yield _tool_call_chunk(0, "transfer_call", {})

        return stream()

    with patch.object(MiniMaxLLMService, "get_chat_completions", completions):
        await service._process_context(context)
    return "".join(
        c.args[0].text for c in service.push_frame.await_args_list if isinstance(c.args[0], LLMTextFrame)
    )


def test_the_answer_asking_for_no_words_is_read_off_the_context():
    assert answers_a_silent_reply(_context(True).messages)
    assert not answers_a_silent_reply(_context(False).messages)
    later = _context(True)
    later.add_message({"role": "user", "content": "Hello?"})
    assert not answers_a_silent_reply(later.messages)


@pytest.mark.asyncio
async def test_words_beside_the_transfer_are_dropped():
    assert await _said(_context(True), SAID, call=True) == ""


@pytest.mark.asyncio
async def test_a_reply_with_no_tool_call_is_still_said():
    assert await _said(_context(True), SAID, call=False) == SAID


@pytest.mark.asyncio
async def test_any_other_answer_keeps_the_models_words():
    assert await _said(_context(False), SAID, call=True) == SAID
