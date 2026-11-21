"""An HTTP tool answer with a sentence for the line is said without the model.

Copy run 7101 (2026-11-21): route_call's email answer was a sentence written
out, and the model was run again only to say it -- 2.5 s from the caller's
last word to the agent's first.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.services.workflow import pipecat_engine_custom_tools as custom_tools
from api.services.workflow.pipecat_engine_custom_tools import CustomToolManager

EMAIL = "The team handles this one by email, so please email them at brokers at mirrorlabs dot co."


def _manager(*, realtime: bool):
    engine = SimpleNamespace(
        _is_realtime=realtime,
        _call_context_vars={},
        _gathered_context={},
        active_agent=None,
        queue_text_message=AsyncMock(return_value=True),
    )
    manager = CustomToolManager(engine)
    manager.get_organization_id = AsyncMock(return_value=1)
    return manager, engine


def _tool():
    return SimpleNamespace(
        name="route_call",
        tool_uuid="route-call",
        definition={"type": "http_api", "config": {"method": "POST", "url": "https://x.example/route"}},
    )


async def _run(manager, monkeypatch, answer):
    monkeypatch.setattr(custom_tools, "execute_http_tool", AsyncMock(return_value=answer))
    params = SimpleNamespace(arguments={}, result_callback=AsyncMock())
    await manager._create_http_tool_handler(_tool(), "route_call")(params)
    return params.result_callback


@pytest.mark.asyncio
async def test_the_line_says_the_sentence_and_the_model_is_not_run(monkeypatch):
    manager, engine = _manager(realtime=False)
    answer = {"status": "success", "data": {"result": "routed", "line_says": EMAIL, "instructions": "…"}}
    callback = await _run(manager, monkeypatch, answer)
    engine.queue_text_message.assert_awaited_once_with(EMAIL)
    (result,), kwargs = callback.await_args
    assert result == answer
    assert kwargs["properties"].run_llm is False


@pytest.mark.asyncio
async def test_an_answer_without_one_runs_the_model_as_before(monkeypatch):
    manager, engine = _manager(realtime=False)
    answer = {"status": "success", "data": {"result": "routed", "instructions": "say this"}}
    callback = await _run(manager, monkeypatch, answer)
    engine.queue_text_message.assert_not_awaited()
    assert callback.await_args.kwargs == {}


@pytest.mark.asyncio
async def test_a_realtime_line_never_plays_it(monkeypatch):
    manager, engine = _manager(realtime=True)
    answer = {"status": "success", "data": {"result": "routed", "line_says": EMAIL}}
    callback = await _run(manager, monkeypatch, answer)
    engine.queue_text_message.assert_not_awaited()
    assert callback.await_args.kwargs == {}
