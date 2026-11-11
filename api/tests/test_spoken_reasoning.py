"""The model's own reasoning never reaches what a caller hears (Harkbell T-662)."""

from unittest.mock import AsyncMock, patch

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.minimax.llm import MiniMaxLLMService

from api.services.pipecat.minimax_llm import (
    HarkbellMiniMaxLLMService,
    _content_chunk,
    _tool_call_chunk,
)
from api.services.pipecat.spoken_reasoning import (
    SpokenReasoningFilter,
    is_reasoning,
    opens_with_self_talk,
)

# Staging run 3831, round 181c: the whole of a reply to "tomorrow at seven thirty".
RUN_3831 = (
    "callback_asked is false, so I won't ask about a number. Let me give Priya "
    "the email route. Wait — the caller is Priya, not Dev. Let me re-read. The "
    'caller said "I\'m Priya with Vivid Seats" — no, they said "I\'m Dev with…'
)

SPOKEN = [
    "Wait, did you say seven or eleven?",
    "Let me read that back: 7, 8, 6…",
    "Let me read that back: 7, 8, 6, 1. Is that right?",
    "Sure — what's the order number?",
    "You can email kimon_ops@mirrorlabs.com and they'll sort it out.",
    "The address is orders@mirror-labs.com.",
    "I'm putting you through to the team now — please hold.",
    "I'm putting you through to the team now. Please hold.",
    "Let me check the times for Thursday.",
    "I need to grab your order number first. What is it?",
    "I should mention we close at nine tonight.",
    "Is the number on the caller ID the best one to reach you?",
    "Tomorrow at seven thirty works.",
    "Your order VG-55120 is on its way.",
    "It's at www.mirrorlabs.com/help_center if you'd like to look.",
    "Hmm, let me see. One moment.",
    "Okay, got it.\nAnything else?",
]


def say(text: str, step: int = 4) -> str:
    """What reaches speech when ``text`` streams in ``step``-character pieces."""
    reasoning = SpokenReasoningFilter()
    said = "".join(reasoning.feed(text[i : i + step]) for i in range(0, len(text), step))
    return said + reasoning.finish()


@pytest.mark.parametrize("step", [1, 4, 1000])
def test_the_run_3831_reply_says_nothing(step):
    assert say(RUN_3831, step) == ""


@pytest.mark.parametrize("step", [1, 4, 1000])
@pytest.mark.parametrize("text", SPOKEN)
def test_spoken_replies_pass_untouched(text, step):
    assert say(text, step) == text


@pytest.mark.parametrize(
    "sentence",
    [
        "callback_asked is false.",
        "I'll set order_number_confirmed to true.",
        "So route_call needs the marketplace.",
        "Wait — the caller is Priya, not Dev.",
        "The user wants a refund.",
        "Let me re-read.",
        "Let me reread what they said.",
        "Let me reconsider.",
    ],
)
def test_each_signal_alone_makes_a_sentence_reasoning(sentence):
    assert is_reasoning(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "Is the number on the caller ID the best one?",
        "kimon_ops@mirrorlabs.com",
        "Wait, did you say seven or eleven?",
        "Let me double-check that for you.",
        "Hi, this is Emma — [cut off by the caller]",
    ],
)
def test_ordinary_speech_is_not_reasoning(sentence):
    assert not is_reasoning(sentence)


def test_self_talk_openers_are_held_not_decided():
    assert opens_with_self_talk("Let me give Priya the email route.")
    assert opens_with_self_talk("I need to check something.")
    assert opens_with_self_talk("Okay, I should ask for the number.")
    assert not opens_with_self_talk("Sure — what's the order number?")


def test_a_self_talk_sentence_goes_with_the_reasoning_that_follows_it():
    text = "Let me give Priya the email route. callback_asked is false."
    assert say(text) == ""


def test_a_spoken_answer_after_a_reasoning_paragraph_is_kept():
    answer = "Sorry — could you give me the order number again?"
    assert say(f"{RUN_3831}\n\n{answer}") == answer


def test_what_was_said_before_the_reasoning_is_kept():
    assert say(f"Tomorrow at seven thirty works. {RUN_3831}") == "Tomorrow at seven thirty works. "


def test_reasoning_runs_to_the_end_of_its_paragraph():
    text = "Okay. route_call needs the order. Got it, thanks!"
    assert say(text) == "Okay. "


# ── through the service ───────────────────────────────────────────────────


def _service() -> HarkbellMiniMaxLLMService:
    with patch.object(HarkbellMiniMaxLLMService, "create_client"):
        service = HarkbellMiniMaxLLMService(api_key="test-key")
    service.push_frame = AsyncMock()
    return service


def _context(*names: str) -> LLMContext:
    return LLMContext(
        messages=[],
        tools=ToolsSchema(
            standard_tools=[
                FunctionSchema(name=n, description=n, properties={}, required=[])
                for n in names
            ]
        ),
    )


async def _process(service, chunks):
    async def completions(self, context):
        async def stream():
            for chunk in chunks:
                yield chunk

        return stream()

    with patch.object(MiniMaxLLMService, "get_chat_completions", completions):
        await service._process_context(_context("route_call"))
    return "".join(
        call.args[0].text
        for call in service.push_frame.await_args_list
        if isinstance(call.args[0], LLMTextFrame)
    )


def _pieces(text: str, size: int = 7):
    return [_content_chunk(text[i : i + size]) for i in range(0, len(text), size)]


@pytest.mark.asyncio
async def test_a_reply_that_is_all_reasoning_says_nothing_and_its_call_still_runs():
    service = _service()
    service.register_function("route_call", AsyncMock())
    service._run_or_defer_function_calls = AsyncMock()
    args = {"order_number": "55120933"}

    said = await _process(service, [*_pieces(RUN_3831), _tool_call_chunk(0, "route_call", args)])

    assert said == ""
    calls = service._run_or_defer_function_calls.await_args.args[0]
    assert [(c.function_name, c.arguments) for c in calls] == [("route_call", args)]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", SPOKEN)
async def test_a_spoken_reply_reaches_speech_whole(text):
    assert await _process(_service(), _pieces(text)) == text


@pytest.mark.asyncio
async def test_each_response_is_judged_on_its_own():
    service = _service()
    assert await _process(service, _pieces(RUN_3831)) == ""
    service.push_frame.reset_mock()
    assert await _process(service, _pieces("Sure — what's the order number?")) == (
        "Sure — what's the order number?"
    )
