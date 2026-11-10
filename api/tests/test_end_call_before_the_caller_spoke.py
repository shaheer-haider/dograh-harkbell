"""Harkbell T-664: an end-call tool cannot end a call the caller has not spoken on yet."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.services.workflow.pipecat_engine_custom_tools import (
    EARLY_END_REFUSAL,
    CustomToolManager,
    _caller_has_spoken,
)


def _ctx(*messages):
    return SimpleNamespace(messages=list(messages))


def test_an_empty_conversation_has_no_speaker():
    assert not _caller_has_spoken(_ctx())
    assert not _caller_has_spoken(_ctx({"role": "system", "content": "You are a receptionist."}))
    # The model's own tool call says nothing.
    assert not _caller_has_spoken(
        _ctx({"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]}, {"role": "tool", "content": "{}"})
    )
    assert not _caller_has_spoken(_ctx({"role": "user", "content": "   "}))


def test_only_the_callers_words_count():
    assert _caller_has_spoken(_ctx({"role": "user", "content": "hi"}))
    assert _caller_has_spoken(_ctx({"role": "user", "content": [{"type": "text", "text": "hello?"}]}))
    # Copy run 4181: the recorded-line notice was already in the conversation
    # as the agent's words, and the call still ended at hello.
    assert not _caller_has_spoken(
        _ctx({"role": "assistant", "content": "This call may be recorded."}, {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]})
    )


def test_an_unreadable_context_never_holds_a_call_open():
    assert _caller_has_spoken(None)
    assert _caller_has_spoken(SimpleNamespace())


def _params(context):
    return SimpleNamespace(context=context, arguments={}, result_callback=AsyncMock())


@pytest.mark.asyncio
async def test_refused_twice_then_allowed():
    manager = CustomToolManager.__new__(CustomToolManager)
    inner = AsyncMock()
    guarded = manager._refuse_before_anyone_speaks(inner, "say_goodbye")

    for _ in range(2):
        params = _params(_ctx())
        await guarded(params)
        result = params.result_callback.await_args
        assert result.args[0] == {"status": "refused", "message": EARLY_END_REFUSAL}
        assert result.kwargs["properties"].run_llm is True
    inner.assert_not_awaited()

    # A model that insists a third time is not looped forever.
    await guarded(_params(_ctx()))
    inner.assert_awaited_once()


@pytest.mark.asyncio
async def test_once_the_caller_has_spoken_the_tool_ends_the_call():
    manager = CustomToolManager.__new__(CustomToolManager)
    inner = AsyncMock()
    guarded = manager._refuse_before_anyone_speaks(inner, "end_call")
    params = _params(_ctx({"role": "user", "content": "That's all, bye."}))
    await guarded(params)
    inner.assert_awaited_once_with(params)
    params.result_callback.assert_not_awaited()
