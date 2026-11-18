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


@pytest.mark.asyncio
async def test_an_exhausted_account_ends_the_call_and_a_passing_error_does_not():
    """A refused session left callers in a minute of silence (staging run 4911)."""
    service = _service()
    service.push_error = AsyncMock()

    def error(**fields):
        return SimpleNamespace(error=SimpleNamespace(**fields))

    await service._handle_evt_error(
        error(type="insufficient_quota", code="credit_balance_exhausted", message="You have no credits remaining.")
    )
    assert service.push_error.await_args.kwargs["force_treat_as_permanent"] is True

    await service._handle_evt_error(error(type="invalid_request_error", code="invalid_api_key", message="Incorrect API key"))
    assert service.push_error.await_args.kwargs["force_treat_as_permanent"] is True

    await service._handle_evt_error(error(type="server_error", code=None, message="The server had an error"))
    assert service.push_error.await_args.kwargs["force_treat_as_permanent"] is False


@pytest.mark.asyncio
async def test_a_click_is_not_the_caller_speaking_and_a_word_is():
    with (
        patch.object(OpenAIRealtimeLLMService, "_handle_evt_speech_started", AsyncMock()),
        patch.object(OpenAIRealtimeLLMService, "_handle_evt_speech_stopped", AsyncMock()),
    ):
        click = _service()
        await click._handle_evt_speech_started(SimpleNamespace(audio_start_ms=1000))
        await click._handle_evt_speech_stopped(SimpleNamespace(audio_end_ms=1120))
        assert click.caller_speech_heard is False

        word = _service()
        await word._handle_evt_speech_started(SimpleNamespace(audio_start_ms=1000))
        await word._handle_evt_speech_stopped(SimpleNamespace(audio_end_ms=1400))
        assert word.caller_speech_heard is True


@pytest.mark.asyncio
async def test_a_prompt_before_the_session_is_ready_is_kept_and_sent_when_it_is():
    from pipecat.frames.frames import LLMMessagesAppendFrame

    service = _service()
    service._api_session_ready = False
    service._send_manual_response_create = AsyncMock()
    frame = LLMMessagesAppendFrame([{"role": "user", "content": "[keypad] 1"}], run_llm=True)
    await service._handle_messages_append(frame)
    service.send_client_event.assert_not_awaited()
    assert service._appends_before_session_ready == [frame]

    await service._handle_evt_session_updated(SimpleNamespace())
    sent = [call.args[0] for call in service.send_client_event.await_args_list]
    assert any(isinstance(event, events.ConversationItemCreateEvent) for event in sent)
    service._send_manual_response_create.assert_awaited_once()
    assert service._appends_before_session_ready == []


@pytest.mark.asyncio
async def test_a_response_refused_over_a_live_one_is_asked_again_when_it_ends():
    service = _service()
    service._manual_response_outstanding = True
    with patch.object(
        OpenAIRealtimeLLMService,
        "_maybe_handle_evt_retrieve_conversation_item_error",
        AsyncMock(return_value=False),
    ):
        handled = await service._maybe_handle_evt_retrieve_conversation_item_error(
            SimpleNamespace(error=SimpleNamespace(code="conversation_already_has_active_response"))
        )
    assert handled is False
    assert service._run_llm_after_response_done is True
    # An error that is not about a response of ours changes nothing.
    other = _service()
    with patch.object(
        OpenAIRealtimeLLMService,
        "_maybe_handle_evt_retrieve_conversation_item_error",
        AsyncMock(return_value=False),
    ):
        await other._maybe_handle_evt_retrieve_conversation_item_error(
            SimpleNamespace(error=SimpleNamespace(code="conversation_already_has_active_response"))
        )
    assert other._run_llm_after_response_done is False


@pytest.mark.asyncio
async def test_nothing_is_asked_of_the_model_after_the_conversation_closed():
    from pipecat.frames.frames import LLMMessagesAppendFrame

    service = _service()
    service._send_manual_response_create = AsyncMock()
    await service.close_conversation()
    service.send_client_event.reset_mock()
    await service._handle_messages_append(
        LLMMessagesAppendFrame([{"role": "user", "content": "The user has been quiet."}], run_llm=True)
    )
    service.send_client_event.assert_not_awaited()
    service._send_manual_response_create.assert_not_awaited()


def _response_done(status: str, error: dict | None = None, response_id: str = "resp_1"):
    """A ``response.done`` as the provider sends it, down to what the engine reads."""
    tokens = SimpleNamespace(cached_tokens=0, audio_tokens=0, cached_tokens_details=None)
    usage = SimpleNamespace(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        input_token_details=tokens,
        output_token_details=SimpleNamespace(audio_tokens=0),
    )
    details = {"type": status, "error": error} if error else None
    return SimpleNamespace(
        response=SimpleNamespace(
            id=response_id, status=status, status_details=details, usage=usage, output=[]
        )
    )


def _service_reporting_errors() -> DograhOpenAIRealtimeLLMService:
    """A service whose real error path runs, with nothing wired around it."""
    service = _service()
    service.start_llm_usage_metrics = AsyncMock()
    service.stop_processing_metrics = AsyncMock()
    service.push_frame = AsyncMock()
    service._workflow_tool_deferral.complete_response = AsyncMock()
    return service


NO_CREDITS = (
    "You have no credits remaining. Add credits to continue using the API at "
    "https://platform.openai.com/settings/organization/billing/."
)


@pytest.mark.asyncio
async def test_a_balance_that_runs_out_mid_call_ends_the_call_at_the_first_failed_reply():
    """Seven failed replies and 76 seconds of silence (production run 5720)."""
    # The message alone, which is all run 5720's log proves the provider sent.
    by_message = _service_reporting_errors()
    await by_message._handle_evt_response_done(_response_done("failed", {"message": NO_CREDITS}))
    assert by_message.is_usable is False

    by_code = _service_reporting_errors()
    await by_code._handle_evt_response_done(
        _response_done("failed", {"type": "insufficient_quota", "code": "insufficient_quota", "message": "x"})
    )
    assert by_code.is_usable is False


@pytest.mark.asyncio
async def test_a_reply_that_fails_in_passing_costs_one_turn_and_the_third_in_a_row_ends_the_call():
    passing = {"type": "server_error", "code": None, "message": "The server had an error"}
    service = _service_reporting_errors()

    await service._handle_evt_response_done(_response_done("failed", passing))
    await service._handle_evt_response_done(_response_done("failed", passing))
    assert service.is_usable is True
    # The caller talking over a reply is not a failure, and does not clear two.
    await service._handle_evt_response_done(_response_done("cancelled"))
    assert service.is_usable is True
    await service._handle_evt_response_done(_response_done("failed", passing))
    assert service.is_usable is False

    recovered = _service_reporting_errors()
    await recovered._handle_evt_response_done(_response_done("failed", passing))
    await recovered._handle_evt_response_done(_response_done("failed", passing))
    await recovered._handle_evt_response_done(_response_done("completed"))
    await recovered._handle_evt_response_done(_response_done("failed", passing))
    assert recovered.is_usable is True


@pytest.mark.asyncio
async def test_an_error_after_a_failed_reply_is_judged_on_its_own():
    """The verdict on one response must not make the next passing error the call's last."""
    service = _service_reporting_errors()
    with patch.object(OpenAIRealtimeLLMService, "_handle_evt_response_done", AsyncMock()):
        # Upstream reported nothing for it: the verdict is dropped with the response.
        await service._handle_evt_response_done(_response_done("failed", {"message": NO_CREDITS}))
    await service.push_error(error_msg="Send error: one dropped packet")
    assert service.is_usable is True


@pytest.mark.asyncio
async def test_a_real_reply_passes_through_and_the_same_reply_failed_ends_the_call():
    """The provider's own ``response.done``, parsed as the receive loop parses it."""
    import json
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "openai_realtime_live_transcribe_session.json"
    done = next(e for e in json.loads(fixture.read_text()) if e["type"] == "response.done")

    service = _service_reporting_errors()
    service._call_event_handler = AsyncMock()
    await service._handle_evt_response_done(events.parse_server_event(json.dumps(done)))
    assert service.is_usable is True
    assert service._failed_responses_in_a_row == 0

    done["response"]["status"] = "failed"
    done["response"]["status_details"] = {"type": "failed", "error": {"message": NO_CREDITS}}
    await service._handle_evt_response_done(events.parse_server_event(json.dumps(done)))
    assert service.is_usable is False
