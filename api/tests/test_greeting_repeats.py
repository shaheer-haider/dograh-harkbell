"""A later reply never says the call's greeting again (production run 4603)."""

from unittest.mock import AsyncMock, patch

import pytest
from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.minimax.llm import MiniMaxLLMService

from api.services.pipecat.cut_off import mark_cut_off
from api.services.pipecat.greeting_repeats import (
    GreetingRepeatGuard,
    greeting_from_context,
    sentence_key,
)
from api.services.pipecat.minimax_llm import HarkbellMiniMaxLLMService, _content_chunk
from api.services.pipecat.spoken_reasoning import SpokenReasoningFilter

# Run 4603: the greeting, and the reply to "Hello?" said over it.
GREETING = "Hi, this is Emma at Mirror Labs. Thanks for calling. How can I help you?"
REPLY_4603 = "Hi, this is Emma at Mirror Labs. Thanks for calling. How can I help you?"
NO_QUESTION = "Hi, this is Emma at Mirror Labs. Thanks for calling."
EMAIL = "You can email orders at mirror labs dot com."


def say(reply: str, greeting: str, step: int = 5) -> str:
    reasoning = SpokenReasoningFilter(repeats=GreetingRepeatGuard(greeting))
    said = "".join(reasoning.feed(reply[i : i + step]) for i in range(0, len(reply), step))
    return said + reasoning.finish()


@pytest.mark.parametrize("step", [1, 5, 1000])
def test_run_4603s_reply_says_only_the_question(step):
    assert say(REPLY_4603, GREETING, step) == "How can I help you?"


def test_a_re_greeting_with_more_to_say_keeps_the_rest():
    reply = "Hi, this is Emma at Mirror Labs. Sorry, the line dropped. What's the order number?"
    assert say(reply, GREETING) == "Sorry, the line dropped. What's the order number?"


def test_a_re_greeting_with_nothing_left_says_the_greetings_question():
    assert say("Hi, this is Emma at Mirror Labs.", GREETING) == "How can I help you?"
    # Normalised: case, punctuation, dashes and spacing.
    assert say("hi — this is Emma at Mirror Labs!", GREETING) == "How can I help you?"


def test_a_greeting_with_no_question_leaves_a_repeat_with_nothing_to_say():
    assert say(NO_QUESTION, NO_QUESTION) == ""
    assert say(f"{NO_QUESTION} What can I do for you?", NO_QUESTION) == "What can I do for you?"


def test_a_line_said_twice_that_is_not_the_greeting_is_said_twice():
    greeting = f"{GREETING} {EMAIL}"
    assert say(EMAIL, greeting) == EMAIL
    assert say(f"Sure. {EMAIL}", GREETING) == f"Sure. {EMAIL}"


def test_a_greeting_sentence_said_on_its_own_is_not_a_re_greeting():
    # A goodbye that thanks them for calling still says so.
    assert say("Glad I could help. Thanks for calling.", GREETING) == (
        "Glad I could help. Thanks for calling."
    )


def test_sentence_keys_ignore_case_punctuation_dashes_and_space():
    assert sentence_key("Hi, this is Emma —  at Mirror Labs.") == sentence_key(
        "hi this is emma at mirror labs"
    )


def test_the_greeting_is_read_from_the_context_before_the_callers_words():
    context = LLMContext(
        messages=[
            {"role": "assistant", "tool_calls": [{"id": "c", "type": "function",
             "function": {"name": "begin", "arguments": "{}"}}]},
            {"role": "tool", "content": '{"status": "done"}', "tool_call_id": "c"},
            {"role": "assistant", "content": mark_cut_off("Hi, this is Emma at Mirror Labs.")},
            {"role": "user", "content": "Hello?"},
            {"role": "assistant", "content": "Yes, I'm here."},
        ]
    )
    assert greeting_from_context(context) == "Hi, this is Emma at Mirror Labs."
    # A caller who spoke first heard no greeting to repeat.
    assert greeting_from_context(
        LLMContext(messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}])
    ) is None


# ── through the service ───────────────────────────────────────────────────


def _service() -> HarkbellMiniMaxLLMService:
    with patch.object(HarkbellMiniMaxLLMService, "create_client"):
        service = HarkbellMiniMaxLLMService(api_key="test-key")
    service.push_frame = AsyncMock()
    return service


async def _reply(service, context, text: str) -> str:
    async def completions(self, context):
        async def stream():
            for i in range(0, len(text), 6):
                yield _content_chunk(text[i : i + 6])

        return stream()

    service.push_frame.reset_mock()
    with patch.object(MiniMaxLLMService, "get_chat_completions", completions):
        await service._process_context(context)
    return "".join(
        call.args[0].text
        for call in service.push_frame.await_args_list
        if isinstance(call.args[0], LLMTextFrame)
    )


@pytest.mark.asyncio
async def test_a_call_says_its_greeting_once():
    service = _service()
    context = LLMContext()
    # The opening: nobody has spoken yet, so it is said whole and remembered.
    assert await _reply(service, context, GREETING) == GREETING
    # Remembered as sent, even though only part of it reached the context.
    context.add_message({"role": "assistant", "content": mark_cut_off("Hi, this is Emma at")})
    context.add_message({"role": "user", "content": "Hello?"})
    assert await _reply(service, context, REPLY_4603) == "How can I help you?"


@pytest.mark.asyncio
async def test_a_text_chat_turn_reads_the_greeting_from_the_context():
    # A text chat builds its service for one turn, so it remembers nothing.
    context = LLMContext(
        messages=[
            {"role": "assistant", "content": GREETING},
            {"role": "user", "content": "Hello?"},
        ]
    )
    assert await _reply(_service(), context, REPLY_4603) == "How can I help you?"


@pytest.mark.asyncio
async def test_without_a_greeting_nothing_is_dropped():
    context = LLMContext(messages=[{"role": "user", "content": "Hi there"}])
    assert await _reply(_service(), context, REPLY_4603) == REPLY_4603
