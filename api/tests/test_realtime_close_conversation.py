"""Harkbell: after an end-call tool runs, a realtime model hears and says nothing.

The goodbye is the tool's recording. A realtime provider kept listening while
it played, and the tail of a caller's "thank you, bye" gave the model another
turn: "Goodbye." over the recorded one (copy runs 4882, 4884).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from api.services.pipecat.realtime.openai_realtime import (
    DograhOpenAIRealtimeLLMService,
)


def _service() -> DograhOpenAIRealtimeLLMService:
    service = DograhOpenAIRealtimeLLMService(api_key="test")
    service._api_session_ready = True
    service.send_client_event = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_closing_cancels_a_reply_under_way_and_silences_the_caller():
    service = _service()
    await service.close_conversation()

    sent = [call.args[0] for call in service.send_client_event.await_args_list]
    assert any(isinstance(event, events.ResponseCancelEvent) for event in sent)

    frame = InputAudioRawFrame(audio=b"\x01\x02" * 80, sample_rate=24000, num_channels=1)
    prepared = await service._prepare_audio_frame(frame)
    assert prepared.audio == b"\x00" * len(frame.audio)


@pytest.mark.asyncio
async def test_nothing_the_model_still_sends_reaches_the_caller():
    service = _service()
    await service.close_conversation()

    with (
        patch.object(OpenAIRealtimeLLMService, "_handle_evt_audio_delta", AsyncMock()) as audio,
        patch.object(
            OpenAIRealtimeLLMService, "_handle_evt_audio_transcript_delta", AsyncMock()
        ) as transcript,
    ):
        await service._handle_evt_audio_delta(SimpleNamespace(delta="AAAA"))
        await service._handle_evt_audio_transcript_delta(SimpleNamespace(delta="Goodbye."))
        audio.assert_not_awaited()
        transcript.assert_not_awaited()

    service.run_function_calls = AsyncMock()
    await service._handle_evt_function_call_arguments_done(
        SimpleNamespace(arguments="{}", call_id="call_1", response_id="resp_1")
    )
    service.run_function_calls.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_open_conversation_is_untouched():
    service = _service()
    with patch.object(OpenAIRealtimeLLMService, "_handle_evt_audio_delta", AsyncMock()) as audio:
        await service._handle_evt_audio_delta(SimpleNamespace(delta="AAAA"))
        audio.assert_awaited_once()
    frame = InputAudioRawFrame(audio=b"\x01\x02" * 80, sample_rate=24000, num_channels=1)
    assert (await service._prepare_audio_frame(frame)).audio == frame.audio


@pytest.mark.asyncio
async def test_a_response_asked_for_by_hand_pushes_one_start_frame():
    """The provider's own start for it is swallowed; the next response's is not."""
    from pipecat.frames.frames import LLMFullResponseStartFrame
    from pipecat.processors.frame_processor import FrameProcessor

    service = _service()
    service.start_processing_metrics = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    pushed = []

    async def record(self, frame, direction=None):
        pushed.append(frame)

    with patch.object(FrameProcessor, "push_frame", record):
        await service._send_manual_response_create()
        # The provider's "assistant item added" for that same response.
        await service.push_frame(LLMFullResponseStartFrame())
        assert [type(frame) for frame in pushed] == [LLMFullResponseStartFrame]
        # A response the provider starts by itself keeps its one start.
        await service.push_frame(LLMFullResponseStartFrame())
        assert len(pushed) == 2
        # A tool-only response adds no assistant item: its end clears the flag.
        await service._send_manual_response_create()
        service._manual_response_start_pushed = True
        with patch.object(OpenAIRealtimeLLMService, "_handle_evt_response_done", AsyncMock()):
            await service._handle_evt_response_done(
                SimpleNamespace(response=SimpleNamespace(id="r", status="completed"))
            )
        await service.push_frame(LLMFullResponseStartFrame())
        assert len(pushed) == 4


@pytest.mark.asyncio
async def test_the_caller_counts_as_heard_before_their_transcript_arrives():
    """An end-call guard that reads the context alone refused a caller who had spoken."""
    from pipecat.frames.frames import LLMMessagesAppendFrame

    from api.services.workflow.pipecat_engine_custom_tools import CustomToolManager

    service = _service()
    assert service.caller_speech_heard is False
    with patch.object(OpenAIRealtimeLLMService, "_handle_evt_speech_stopped", AsyncMock()):
        await service._handle_evt_speech_stopped(SimpleNamespace())
    assert service.caller_speech_heard is True

    keyed = _service()
    keyed._send_manual_response_create = AsyncMock()
    await keyed._handle_messages_append(
        LLMMessagesAppendFrame([{"role": "user", "content": "[keypad] 1"}], run_llm=True)
    )
    assert keyed.caller_speech_heard is True
    # The idle check is the engine's own words, not the caller's.
    idle = _service()
    idle._send_manual_response_create = AsyncMock()
    await idle._handle_messages_append(
        LLMMessagesAppendFrame([{"role": "user", "content": "The user has been quiet."}], run_llm=True)
    )
    assert idle.caller_speech_heard is False

    manager = CustomToolManager.__new__(CustomToolManager)
    manager._engine = SimpleNamespace(active_agent=SimpleNamespace(llm=service))
    assert manager._realtime_caller_heard() is True
    manager._engine = SimpleNamespace(active_agent=SimpleNamespace(llm=idle))
    assert manager._realtime_caller_heard() is False
    manager._engine = SimpleNamespace(active_agent=SimpleNamespace(llm=object()))
    assert manager._realtime_caller_heard() is False
