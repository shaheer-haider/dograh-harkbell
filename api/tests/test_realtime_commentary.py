"""Harkbell (-harkbell.33): a line may play only the model's final answer.

The reasoning realtime models speak in two phases, and each assistant message
says which it is: commentary -- "a short spoken update" before a tool call or
a longer think -- and the final answer. On the smaller model the commentary
was what a caller should never hear, across six prompts: "Okay, thanks for
confirming that; let me think about what to do next" (copy run 6105), "Let me
think this through with you for a moment" (6016). A line whose configuration
sets ``speak_commentary`` false hears none of it.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from api.services.pipecat.realtime.openai_realtime import (
    DograhOpenAIRealtimeLLMService,
)


def _service(speak_commentary: bool) -> DograhOpenAIRealtimeLLMService:
    service = DograhOpenAIRealtimeLLMService(
        api_key="test", speak_commentary=speak_commentary
    )
    service._api_session_ready = True
    service.send_client_event = AsyncMock()
    return service


def _item_added(item_id: str, phase: str | None):
    """The provider's ``conversation.item.added`` for an assistant message, parsed as on the wire."""
    item = {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}
    if phase is not None:
        item["phase"] = phase
    return events.parse_server_event(
        '{"type": "conversation.item.added", "event_id": "e1", "previous_item_id": null, "item": %s}'
        % __import__("json").dumps(item)
    )


def _audio(item_id: str, response_id: str = "resp_1"):
    return SimpleNamespace(
        type="response.output_audio.delta",
        item_id=item_id,
        response_id=response_id,
        content_index=0,
        output_index=0,
        delta="AAAA",
    )


def _words(item_id: str, text: str, response_id: str = "resp_1"):
    return SimpleNamespace(
        type="response.output_audio_transcript.delta",
        item_id=item_id,
        response_id=response_id,
        delta=text,
    )


def _done(output: list[dict], status: str = "completed", response_id: str = "resp_1"):
    return SimpleNamespace(
        response=SimpleNamespace(
            id=response_id,
            status=status,
            status_details=None,
            output=[SimpleNamespace(**item) for item in output],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )
    )


def _patched():
    return (
        patch.object(OpenAIRealtimeLLMService, "_handle_evt_audio_delta", AsyncMock()),
        patch.object(OpenAIRealtimeLLMService, "_handle_evt_audio_transcript_delta", AsyncMock()),
        patch.object(OpenAIRealtimeLLMService, "_handle_evt_conversation_item_added", AsyncMock()),
        patch.object(OpenAIRealtimeLLMService, "_handle_evt_response_done", AsyncMock()),
    )


def test_the_phase_survives_parsing():
    assert _item_added("item_c", "commentary").item.phase == "commentary"
    assert _item_added("item_f", "final_answer").item.phase == "final_answer"
    # An older model sends none.
    assert _item_added("item_x", None).item.phase is None


@pytest.mark.asyncio
async def test_commentary_before_a_tool_call_is_never_heard_and_the_answer_is():
    service = _service(speak_commentary=False)
    audio_p, words_p, added_p, done_p = _patched()
    with audio_p as audio, words_p as words, added_p, done_p:
        await service._handle_evt_conversation_item_added(_item_added("item_c", "commentary"))
        await service._handle_evt_audio_delta(_audio("item_c"))
        await service._handle_evt_audio_transcript_delta(
            _words("item_c", "Okay, thanks for confirming that; let me think about what to do next.")
        )
        audio.assert_not_awaited()
        words.assert_not_awaited()
        # The response did something else too: the tool call.
        await service._handle_evt_response_done(
            _done(
                [
                    {"id": "item_c", "type": "message", "phase": "commentary"},
                    {"id": "item_t", "type": "function_call", "phase": None},
                ]
            )
        )
        audio.assert_not_awaited()
        words.assert_not_awaited()

        # The next response is the answer, and it plays as it always did.
        await service._handle_evt_conversation_item_added(_item_added("item_f", "final_answer"))
        await service._handle_evt_audio_delta(_audio("item_f", "resp_2"))
        await service._handle_evt_audio_transcript_delta(
            _words("item_f", "And when is the event?", "resp_2")
        )
        assert audio.await_count == 1
        assert words.await_count == 1
    assert service._held_commentary == {}
    assert service._commentary_items == set()


@pytest.mark.asyncio
async def test_commentary_in_front_of_the_answer_in_one_response_is_cut_and_the_answer_plays():
    service = _service(speak_commentary=False)
    audio_p, words_p, added_p, done_p = _patched()
    with audio_p as audio, words_p as words, added_p, done_p:
        # Copy run 6002: "Alright, thanks for calling in. Just a quick question
        # so we can get you to the right path." and then "Hi -- go ahead."
        await service._handle_evt_conversation_item_added(_item_added("item_c", "commentary"))
        await service._handle_evt_audio_delta(_audio("item_c"))
        await service._handle_evt_audio_transcript_delta(_words("item_c", "Alright, thanks for calling in."))
        await service._handle_evt_conversation_item_added(_item_added("item_f", "final_answer"))
        await service._handle_evt_audio_delta(_audio("item_f"))
        await service._handle_evt_audio_transcript_delta(_words("item_f", "Hi — go ahead."))
        await service._handle_evt_response_done(
            _done(
                [
                    {"id": "item_c", "type": "message", "phase": "commentary"},
                    {"id": "item_f", "type": "message", "phase": "final_answer"},
                ]
            )
        )
        assert [call.args[0].item_id for call in audio.await_args_list] == ["item_f"]
        assert [call.args[0].delta for call in words.await_args_list] == ["Hi — go ahead."]


@pytest.mark.asyncio
async def test_a_reply_that_was_only_commentary_is_played_when_it_ends():
    service = _service(speak_commentary=False)
    audio_p, words_p, added_p, done_p = _patched()
    with audio_p as audio, words_p as words, added_p, done_p:
        await service._handle_evt_conversation_item_added(_item_added("item_c", "commentary"))
        await service._handle_evt_audio_delta(_audio("item_c"))
        await service._handle_evt_audio_transcript_delta(_words("item_c", "Hi — go ahead."))
        audio.assert_not_awaited()
        # Nothing else came: no tool, no final answer. It was the whole reply.
        await service._handle_evt_response_done(
            _done([{"id": "item_c", "type": "message", "phase": "commentary"}])
        )
        assert audio.await_count == 1
        assert [call.args[0].delta for call in words.await_args_list] == ["Hi — go ahead."]


@pytest.mark.asyncio
async def test_a_cancelled_reply_plays_none_of_what_was_held():
    service = _service(speak_commentary=False)
    audio_p, words_p, added_p, done_p = _patched()
    with audio_p as audio, words_p, added_p, done_p:
        await service._handle_evt_conversation_item_added(_item_added("item_c", "commentary"))
        await service._handle_evt_audio_delta(_audio("item_c"))
        await service._handle_evt_response_done(
            _done([{"id": "item_c", "type": "message", "phase": "commentary"}], status="cancelled")
        )
        audio.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_line_that_has_not_asked_plays_commentary_as_the_provider_does():
    service = _service(speak_commentary=True)
    audio_p, words_p, added_p, done_p = _patched()
    with audio_p as audio, words_p as words, added_p, done_p:
        await service._handle_evt_conversation_item_added(_item_added("item_c", "commentary"))
        await service._handle_evt_audio_delta(_audio("item_c"))
        await service._handle_evt_audio_transcript_delta(_words("item_c", "One moment."))
        assert audio.await_count == 1
        assert words.await_count == 1
    assert service._held_commentary == {}
    # And the default is to play it.
    assert DograhOpenAIRealtimeLLMService(api_key="test")._speak_commentary is True
