"""Harkbell T-664: an end-call tool cannot end a call the caller has not spoken on yet."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.services.workflow.pipecat_engine_callbacks import IDLE_PROMPTS
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
async def test_refused_every_time_without_running_the_model_again():
    # Run 4615: the third goodbye used to go through, to a caller who had not
    # said a word. And the refusal no longer asks the model for another turn,
    # so it cannot loop on it.
    manager = CustomToolManager.__new__(CustomToolManager)
    inner = AsyncMock()
    guarded = manager._refuse_before_anyone_speaks(inner, "say_goodbye")

    for _ in range(5):
        params = _params(_ctx())
        await guarded(params)
        result = params.result_callback.await_args
        assert result.args[0] == {"status": "refused", "message": EARLY_END_REFUSAL}
        assert result.kwargs["properties"].run_llm is False
    inner.assert_not_awaited()


def test_the_engines_own_silence_prompts_are_not_the_caller_speaking():
    for prompt in IDLE_PROMPTS:
        assert not _caller_has_spoken(_ctx({"role": "user", "content": prompt}))


def _engine(*, greets=True):
    monitor = SimpleNamespace(cancel=MagicMock(), await_caller=MagicMock())
    opened = SimpleNamespace(action="greeting") if greets else None
    return SimpleNamespace(
        call_monitor=monitor,
        active_agent=SimpleNamespace(current_node=SimpleNamespace(id="main")),
        queue_node_greeting=AsyncMock(return_value=opened),
    )


def _manager(engine):
    manager = CustomToolManager.__new__(CustomToolManager)
    manager._engine = engine
    return manager


@pytest.mark.asyncio
async def test_an_ungreeted_caller_is_greeted_once_and_then_waited_for():
    engine = _engine()
    guarded = _manager(engine)._refuse_before_anyone_speaks(AsyncMock(), "say_goodbye")
    for _ in range(3):
        await guarded(_params(_ctx()))
    engine.queue_node_greeting.assert_awaited_once()
    assert engine.queue_node_greeting.await_args.args == ("main",)
    engine.call_monitor.cancel.assert_called_once()
    assert engine.call_monitor.await_caller.call_count == 2


@pytest.mark.asyncio
async def test_a_caller_already_spoken_to_is_only_waited_for():
    engine = _engine()
    guarded = _manager(engine)._refuse_before_anyone_speaks(AsyncMock(), "say_goodbye")
    # Run 4615's first reply: words, but no greeting.
    await guarded(_params(_ctx({"role": "assistant", "content": "No issues found with the recording line notice."})))
    engine.queue_node_greeting.assert_not_awaited()
    engine.call_monitor.await_caller.assert_called_once()


@pytest.mark.asyncio
async def test_a_step_with_no_greeting_says_nothing_and_waits():
    engine = _engine(greets=False)
    guarded = _manager(engine)._refuse_before_anyone_speaks(AsyncMock(), "end_call")
    await guarded(_params(_ctx()))
    engine.queue_node_greeting.assert_awaited_once()
    engine.call_monitor.await_caller.assert_called_once()


@pytest.mark.asyncio
async def test_once_the_caller_has_spoken_the_tool_ends_the_call():
    manager = CustomToolManager.__new__(CustomToolManager)
    inner = AsyncMock()
    guarded = manager._refuse_before_anyone_speaks(inner, "end_call")
    params = _params(_ctx({"role": "user", "content": "That's all, bye."}))
    await guarded(params)
    inner.assert_awaited_once_with(params)
    params.result_callback.assert_not_awaited()
