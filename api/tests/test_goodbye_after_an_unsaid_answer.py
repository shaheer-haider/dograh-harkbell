"""Harkbell (-harkbell.34): a goodbye may not stand in for an outcome nobody heard.

"It's in two weeks. Okay, thanks, bye." The routing tool answers with what to
tell the caller, the model reads their goodbye in the same turn, and it ends
the call -- the caller leaves believing the order is handled when they were
meant to email (suite runs 5054, 5099, 5164, 5165 on the full realtime model;
copy runs 6159 and 6260 on the smaller one). An end-call tool whose config
sets ``sayAnswerFirst`` refuses once while a tool's answer has gone to the
model and no audio has played since.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from api.schemas.tool import EndCallConfig
from api.services.pipecat.realtime.openai_realtime import (
    DograhOpenAIRealtimeLLMService,
)
from api.services.workflow.pipecat_engine_custom_tools import (
    UNSAID_ANSWER_REFUSAL,
    CustomToolManager,
)


def _service() -> DograhOpenAIRealtimeLLMService:
    service = DograhOpenAIRealtimeLLMService(api_key="test")
    service._api_session_ready = True
    return service


def _tool_result(output: str) -> dict:
    """A tool's result as it leaves for the provider (``send_client_event`` dumps the event)."""
    item = events.ConversationItem(type="function_call_output", call_id="call_1", output=output)
    return events.ConversationItemCreateEvent(item=item).model_dump(exclude_none=True)


def _audio():
    return SimpleNamespace(
        type="response.output_audio.delta",
        item_id="item_1",
        response_id="resp_1",
        content_index=0,
        output_index=0,
        delta="AAAA",
    )


@pytest.mark.asyncio
async def test_a_tools_answer_is_unsaid_until_audio_plays():
    service = _service()
    assert service.tool_answer_unsaid is False
    with patch.object(OpenAIRealtimeLLMService, "_ws_send", AsyncMock()):
        await service._ws_send(_tool_result('{"result": "routed", "route": "email"}'))
    assert service.tool_answer_unsaid is True
    with patch.object(OpenAIRealtimeLLMService, "_handle_evt_audio_delta", AsyncMock()):
        await service._handle_evt_audio_delta(_audio())
    assert service.tool_answer_unsaid is False


@pytest.mark.asyncio
async def test_an_ending_or_a_refusal_holds_nothing_for_the_caller():
    service = _service()
    with patch.object(OpenAIRealtimeLLMService, "_ws_send", AsyncMock()):
        await service._ws_send(_tool_result('{"status": "success", "action": "ending_call"}'))
        assert service.tool_answer_unsaid is False
        await service._ws_send(_tool_result('{"status": "refused", "message": "Not yet."}'))
        assert service.tool_answer_unsaid is False
        # Anything else that is not a tool's answer leaves it alone too.
        await service._ws_send({"type": "response.create"})
        assert service.tool_answer_unsaid is False


@pytest.mark.asyncio
async def test_commentary_that_is_held_back_is_not_the_answer_being_said():
    service = DograhOpenAIRealtimeLLMService(api_key="test", speak_commentary=False)
    service._api_session_ready = True
    service.tool_answer_unsaid = True
    service._commentary_items.add("item_1")
    with patch.object(OpenAIRealtimeLLMService, "_handle_evt_audio_delta", AsyncMock()) as played:
        await service._handle_evt_audio_delta(_audio())
    played.assert_not_awaited()
    assert service.tool_answer_unsaid is True


def _manager(llm):
    manager = CustomToolManager.__new__(CustomToolManager)
    manager._engine = SimpleNamespace(active_agent=SimpleNamespace(llm=llm))
    return manager


def _params():
    return SimpleNamespace(context=None, arguments={}, result_callback=AsyncMock())


@pytest.mark.asyncio
async def test_the_goodbye_is_refused_once_and_the_model_is_run_again():
    llm = SimpleNamespace(tool_answer_unsaid=True)
    inner = AsyncMock()
    guarded = _manager(llm)._refuse_before_the_answer_is_said(inner, "say_goodbye")

    first = _params()
    await guarded(first)
    inner.assert_not_awaited()
    result = first.result_callback.await_args
    assert result.args[0] == {"status": "refused", "message": UNSAID_ANSWER_REFUSAL}
    # No properties: the model gets the refusal and replies to it.
    assert "properties" not in result.kwargs

    # A model that insists ends the call on its second try.
    second = _params()
    await guarded(second)
    inner.assert_awaited_once_with(second)
    second.result_callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_goodbye_after_the_answer_was_said_ends_the_call():
    inner = AsyncMock()
    guarded = _manager(SimpleNamespace(tool_answer_unsaid=False))._refuse_before_the_answer_is_said(
        inner, "say_goodbye"
    )
    params = _params()
    await guarded(params)
    inner.assert_awaited_once_with(params)


@pytest.mark.asyncio
async def test_a_line_that_is_not_realtime_is_never_held():
    # A pipeline model has no such flag: the tool behaves as it always has.
    inner = AsyncMock()
    guarded = _manager(SimpleNamespace())._refuse_before_the_answer_is_said(inner, "end_call")
    params = _params()
    await guarded(params)
    inner.assert_awaited_once_with(params)


def test_only_a_tool_that_asks_for_it_is_guarded():
    manager = CustomToolManager.__new__(CustomToolManager)
    manager._create_plain_end_call_handler = lambda tool, name: "plain"
    manager._refuse_before_anyone_speaks = lambda handler, name: handler
    manager._refuse_before_the_answer_is_said = lambda handler, name: "guarded"
    plain = SimpleNamespace(definition={"config": {"messageType": "none"}})
    asking = SimpleNamespace(definition={"config": {"messageType": "none", "sayAnswerFirst": True}})
    assert manager._create_end_call_handler(plain, "say_goodbye") == "plain"
    assert manager._create_end_call_handler(asking, "say_goodbye") == "guarded"
    # The configuration model knows the key, and it is off unless set.
    assert EndCallConfig().sayAnswerFirst is False
    assert EndCallConfig(sayAnswerFirst=True).sayAnswerFirst is True
