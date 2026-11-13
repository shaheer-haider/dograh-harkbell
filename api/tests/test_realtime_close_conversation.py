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
