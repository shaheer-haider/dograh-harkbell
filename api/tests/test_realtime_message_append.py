from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.services.openai.realtime import events

from api.services.pipecat.realtime.openai_realtime import (
    DograhOpenAIRealtimeLLMService,
)
from api.services.workflow.pipecat_engine_callbacks import handle_user_idle


@pytest.mark.asyncio
async def test_openai_realtime_messages_append_frame_sends_conversation_item():
    service = DograhOpenAIRealtimeLLMService(api_key="test")
    service._api_session_ready = True
    service.send_client_event = AsyncMock()
    service._send_manual_response_create = AsyncMock()

    await service._handle_messages_append(
        LLMMessagesAppendFrame(
            [{"role": "user", "content": "Are you still there?"}],
            run_llm=True,
        )
    )

    service.send_client_event.assert_awaited_once()
    event = service.send_client_event.await_args.args[0]
    assert isinstance(event, events.ConversationItemCreateEvent)
    assert event.item.role == "user"
    assert event.item.type == "message"
    assert event.item.content == [
        events.ItemContent(type="input_text", text="Are you still there?")
    ]
    service._send_manual_response_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_idle_handler_uses_realtime_append_path():
    engine = SimpleNamespace(
        llm=SimpleNamespace(),
        end_call_with_reason=AsyncMock(),
    )
    aggregator = SimpleNamespace(push_frame=AsyncMock())
    await handle_user_idle(engine, aggregator, 1)

    aggregator.push_frame.assert_awaited_once()
    frame = aggregator.push_frame.await_args.args[0]
    assert isinstance(frame, LLMMessagesAppendFrame)
    assert frame.run_llm is True
    assert frame.messages == [
        {
            "role": "user",
            "content": "The user has been quiet. Politely and briefly ask if they're still there in the language that the user has been speaking so far.",
        }
    ]


@pytest.mark.asyncio
async def test_a_prompt_over_a_live_reply_cancels_it_and_is_answered_after():
    """Harkbell: a keypress while the model is still speaking.

    The provider cancels a reply only for speech it hears, and refuses a
    second response on top of a live one, so the live reply is cancelled and
    the prompt answered once ``response.done`` has arrived for it.
    """
    service = DograhOpenAIRealtimeLLMService(api_key="test")
    service._api_session_ready = True
    service.send_client_event = AsyncMock()
    service._send_manual_response_create = AsyncMock()
    service._current_assistant_response = SimpleNamespace(id="item_1")

    await service._handle_messages_append(
        LLMMessagesAppendFrame([{"role": "user", "content": "[keypad] 1"}], run_llm=True)
    )

    sent = [call.args[0] for call in service.send_client_event.await_args_list]
    assert isinstance(sent[0], events.ConversationItemCreateEvent)
    assert isinstance(sent[1], events.ResponseCancelEvent)
    service._send_manual_response_create.assert_not_awaited()
    assert service._run_llm_after_response_done is True


@pytest.mark.asyncio
async def test_keypad_entry_reaches_a_realtime_model_as_a_prompt_to_answer():
    from api.services.pipecat.keypad import (
        KeypadUserTurnStartStrategy,
        bind_realtime_keypad,
    )

    handlers = {}
    aggregator = SimpleNamespace(
        push_frame=AsyncMock(),
        add_event_handler=lambda name, handler: handlers.setdefault(name, handler),
    )
    bind_realtime_keypad(aggregator)
    strategy = KeypadUserTurnStartStrategy()
    strategy.last_entry = "[keypad] 1"

    await handlers["on_user_turn_started"](aggregator, strategy)
    # A spoken turn's strategy is none of its business.
    await handlers["on_user_turn_started"](aggregator, object())

    aggregator.push_frame.assert_awaited_once()
    frame = aggregator.push_frame.await_args.args[0]
    assert isinstance(frame, LLMMessagesAppendFrame)
    assert frame.messages == [{"role": "user", "content": "[keypad] 1"}]
    assert frame.run_llm is True
    assert strategy.last_entry is None
