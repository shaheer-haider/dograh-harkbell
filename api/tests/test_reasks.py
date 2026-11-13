"""Three re-asks in a row hand the call on, counted by the engine (production runs 4793, 4794)."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import LLMContextFrame, LLMTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
)
from pipecat.services.minimax.llm import MiniMaxLLMService

from api.services.pipecat.cut_off import mark_cut_off
from api.services.pipecat.minimax_llm import (
    HarkbellMiniMaxLLMService,
    _content_chunk,
    _tool_call_chunk,
)
from api.services.pipecat.reasks import ReAskGuard, is_reask, reask_count
from api.services.pipecat.worker_runner import run_pipeline_worker

GREETING = "Hi, this is Emma at Mirror Labs. How can I help you?"

# Run 4794: the caller said "No. That is wrong." three times.
RUN_4794 = [
    "Sorry about that — what were you calling about?",
    "I'm sorry — could you tell me what you need?",
    "I'm sorry — could you say that again?",
]

REASKS = [
    *RUN_4794,
    # Run 4793
    "Sorry, I didn't catch that — could you say the order number again, digit by digit?",
    "No problem — could you say the order number again, slowly?",
    "Sorry, you broke up there — which marketplace is the order on?",
    "Sorry, you broke up there — could you say again what the customer is avoiding?",
    "Sorry, I still didn't catch that — could you say it one more time?",
    "Sorry, you broke up there — could you say your name again?",
    "Sorry, I'm having trouble hearing you. Is it tickets that haven't arrived, the wrong tickets, or sending them again?",
    "Could you repeat that for me?",
    "Say that again?",
    "Sorry, I didn\u2019t catch that \u2014 could you say it again?",
    mark_cut_off("Sorry, you broke up there — could you"),
]

NOT_REASKS = [
    "Let me read that back: 7, 8, 6 — is that right?",
    "And when is the event — the date and the time?",
    "Would you like me to repeat the address?",
    "Sure — what's the order number?",
    "Thanks. And your name?",
    "Could you tell me your name?",
    "Sorry, you're",
    "I'll read it one more time: 7, 8, 6, 1.",
    "You didn't get the tickets? Let me take a note.",
    "Sorry about the wait — the team will email you today.",
    GREETING,
]


@pytest.mark.parametrize("line", REASKS)
def test_re_asks_are_recognised(line):
    assert is_reask(line)


@pytest.mark.parametrize("line", NOT_REASKS)
def test_other_replies_are_not_re_asks(line):
    assert not is_reask(line)


# ── the run, as the conversation holds it ─────────────────────────────────


def _http_result(tool_call_id: str, result: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps({"status": "success", "status_code": 200, "data": {"result": result}}),
    }


def _call(tool_call_id: str, name: str, arguments: dict | None = None) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments or {})},
            }
        ],
    }


def test_progress_starts_the_count_again():
    messages = [
        {"role": "assistant", "content": GREETING},
        {"role": "user", "content": "No. That is wrong."},
        {"role": "assistant", "content": RUN_4794[0]},
        {"role": "user", "content": "It's order 78612."},
        {"role": "assistant", "content": "Let me read that back: 7, 8, 6, 1, 2 — is that right?"},
        {"role": "user", "content": "No. That is wrong."},
        {"role": "assistant", "content": RUN_4794[1]},
    ]
    state = reask_count(messages)
    assert state.count == 1
    assert state.caller_said == "No. That is wrong."


def test_a_tool_result_and_the_reply_it_produced_count_once():
    messages = [
        {"role": "assistant", "content": RUN_4794[0]},
        {"role": "user", "content": "the customer"},
        _call("r1", "route_call"),
        _http_result("r1", "heard_unclear"),
        {"role": "assistant", "content": "Sorry, you broke up there — could you say again what the customer is avoiding?"},
    ]
    assert reask_count(messages).count == 2
    # Before its reply is made, the result is the second, awaiting it.
    state = reask_count(messages[:-1])
    assert (state.count, state.awaiting_reply) == (2, True)


def test_a_hand_off_starts_the_count_again_and_its_reply_is_not_a_re_ask():
    messages = [
        *[{"role": "assistant", "content": line} for line in RUN_4794[:2]],
        _call("c1", "could_not_understand", {"times": 3}),
        _http_result("c1", "hand_off"),
        {"role": "assistant", "content": "I'm sorry, I'm having trouble understanding — let me put you through to the team. Please hold."},
    ]
    assert reask_count(messages).count == 0


def test_the_reply_to_a_hand_off_is_not_judged_while_it_is_spoken():
    messages = [
        *[{"role": "assistant", "content": line} for line in RUN_4794[:2]],
        _call("c1", "could_not_understand", {"times": 3}),
        _http_result("c1", "hand_off"),
    ]
    guard = ReAskGuard(reask_count(messages))
    for piece in ("I'm sorry, I'm having trouble ", "understanding — let me put you through."):
        assert guard.pass_on(piece)
    assert not guard.is_reask and not guard.holding


# ── through the service ───────────────────────────────────────────────────


def _tools(*names: str) -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[FunctionSchema(name=n, description=n, properties={}, required=[]) for n in names]
    )


def _service(*registered: str) -> HarkbellMiniMaxLLMService:
    with patch.object(HarkbellMiniMaxLLMService, "create_client"):
        service = HarkbellMiniMaxLLMService(api_key="test-key")
    service.push_frame = AsyncMock()
    service._run_or_defer_function_calls = AsyncMock()
    for name in registered:
        service.register_function(name, AsyncMock())
    return service


async def _reply(service, context, *chunks):
    """What reaches speech, and the calls handed to Pipecat, for one reply."""

    async def completions(self, context):
        async def stream():
            for chunk in chunks:
                yield chunk

        return stream()

    service.push_frame.reset_mock()
    service._run_or_defer_function_calls.reset_mock()
    with patch.object(MiniMaxLLMService, "get_chat_completions", completions):
        await service._process_context(context)
    said = "".join(
        call.args[0].text
        for call in service.push_frame.await_args_list
        if isinstance(call.args[0], LLMTextFrame)
    )
    calls = (
        [(c.function_name, c.arguments) for c in service._run_or_defer_function_calls.await_args.args[0]]
        if service._run_or_defer_function_calls.await_args
        else []
    )
    return said, calls


def _words(text: str, size: int = 7):
    return [_content_chunk(text[i : i + size]) for i in range(0, len(text), size)]


def _context(messages, *tools) -> LLMContext:
    tools = tools or ("could_not_understand", "route_call", "transfer_call")
    return LLMContext(messages=list(messages), tools=_tools(*tools))


def _run_4794_context(replies: int) -> list[dict]:
    messages = [{"role": "assistant", "content": GREETING}]
    for line in RUN_4794[:replies]:
        messages += [{"role": "user", "content": "No. That is wrong."}, {"role": "assistant", "content": line}]
    return messages + [{"role": "user", "content": "No. That is wrong."}]


@pytest.mark.asyncio
async def test_run_4794s_third_re_ask_is_replaced_by_the_hand_off():
    service = _service("could_not_understand", "transfer_call")
    # The first two are said and counted.
    for replies in (0, 1):
        said, calls = await _reply(service, _context(_run_4794_context(replies)), *_words(RUN_4794[replies]))
        assert said == RUN_4794[replies] and calls == []
    # The third is not said; the engine calls the tool in its place.
    said, calls = await _reply(service, _context(_run_4794_context(2)), *_words(RUN_4794[2]))
    assert said == ""
    assert calls == [
        ("could_not_understand", {"about": "what_they_need", "times": 3, "heard": "No. That is wrong."})
    ]


@pytest.mark.asyncio
async def test_the_models_own_hand_off_is_left_alone():
    service = _service("could_not_understand", "transfer_call")
    own = {"about": "what_they_need", "times": 3, "heard": "No."}
    said, calls = await _reply(
        service,
        _context(_run_4794_context(2)),
        *_words(RUN_4794[2]),
        _tool_call_chunk(0, "could_not_understand", own),
    )
    assert said == RUN_4794[2]
    assert calls == [("could_not_understand", own)]


@pytest.mark.asyncio
async def test_a_step_without_the_tool_never_counts():
    service = _service("transfer_call")
    said, calls = await _reply(
        service, _context(_run_4794_context(2), "transfer_call"), *_words(RUN_4794[2])
    )
    assert said == RUN_4794[2] and calls == []


@pytest.mark.asyncio
async def test_run_4793_hands_off_at_the_third_in_a_row_not_before():
    service = _service("could_not_understand", "route_call", "transfer_call")
    messages = [{"role": "assistant", "content": GREETING}, {"role": "user", "content": "My tickets never came."}]

    async def say(line, caller):
        said, calls = await _reply(service, _context(messages), *_words(line))
        messages.extend([{"role": "assistant", "content": said}] if said else [])
        if caller:
            messages.append({"role": "user", "content": caller})
        return said, calls

    # Two order-number re-asks, then a read-back that lands: no hand-off.
    assert (await say("Sorry, I didn't catch that — could you say the order number again, digit by digit?", "seven eight six"))[1] == []
    assert (await say("No problem — could you say the order number again, slowly?", "seven eight six one two"))[1] == []
    assert (await say("Let me read that back: 7, 8, 6, 1, 2 — is that right?", "Yes."))[1] == []
    # A new run: the marketplace, then route_call heard it unclear.
    assert (await say("Sorry, you broke up there — which marketplace is the order on?", "Stub hub. The customer is avoiding"))[1] == []
    messages += [_call("r1", "route_call"), _http_result("r1", "heard_unclear")]
    assert (await say("Sorry, you broke up there — could you say again what the customer is avoiding?", "they keep"))[1] == []
    # The third in a row is not said.
    said, calls = await say("Sorry, I still didn't catch that — could you say it one more time?", None)
    assert said == ""
    assert calls == [("could_not_understand", {"about": "what_they_need", "times": 3, "heard": "they keep"})]


@pytest.mark.asyncio
async def test_an_order_number_run_hands_off_about_the_order_number():
    service = _service("could_not_understand", "transfer_call")
    line = "Sorry, I didn't catch that — could you say the order number again, digit by digit?"
    messages = [{"role": "assistant", "content": GREETING}]
    for _ in range(2):
        messages += [{"role": "user", "content": "seven"}, {"role": "assistant", "content": line}]
    messages.append({"role": "user", "content": "seven eight"})
    said, calls = await _reply(service, _context(messages), *_words(line))
    assert said == ""
    assert calls[0][1]["about"] == "order_number"


@pytest.mark.asyncio
async def test_a_short_apology_already_said_may_stand_but_not_the_ask():
    service = _service("could_not_understand", "transfer_call")
    said, calls = await _reply(
        service, _context(_run_4794_context(2)), *_words("Sorry about that. What were you calling about?")
    )
    assert said == "Sorry about that. "
    assert calls and calls[0][0] == "could_not_understand"


# ── the whole loop, shaped as a text chat ─────────────────────────────────


@pytest.mark.asyncio
async def test_the_model_is_run_on_the_hand_offs_result():
    hand_off = {
        "status": "success",
        "status_code": 200,
        "data": {"result": "hand_off", "times": 3, "say": "I'm sorry, I'm having trouble understanding — let me put you through to the team. Please hold."},
    }
    seen = []

    async def could_not_understand(params):
        seen.append(dict(params.arguments))
        await params.result_callback(hand_off)

    steps = [
        _words(RUN_4794[2]),
        _words("I'm sorry, I'm having trouble understanding — let me put you through to the team. Please hold."),
    ]
    requests = []

    async def completions(self, context):
        requests.append([dict(m) for m in context.messages if isinstance(m, dict)])
        chunks = steps.pop(0)

        async def stream():
            for chunk in chunks:
                yield chunk

        return stream()

    with patch.object(HarkbellMiniMaxLLMService, "create_client"):
        llm = HarkbellMiniMaxLLMService(api_key="test-key")
    llm.register_function("could_not_understand", could_not_understand)
    llm.register_function("transfer_call", AsyncMock())
    context = _context(_run_4794_context(2))
    _user, assistant = LLMContextAggregatorPair(context, assistant_params=LLMAssistantAggregatorParams())
    worker = PipelineWorker(Pipeline([llm, assistant]), enable_rtvi=False)
    runner = asyncio.create_task(run_pipeline_worker(worker))
    try:
        with patch.object(MiniMaxLLMService, "get_chat_completions", completions):
            await asyncio.sleep(0.2)
            await llm.queue_frame(LLMContextFrame(context))
            async with asyncio.timeout(5):
                while len(requests) < 2:
                    await asyncio.sleep(0.02)
            await asyncio.sleep(0.2)
    finally:
        await worker.cancel()
        await asyncio.wait_for(runner, 5)

    assert seen == [{"about": "what_they_need", "times": 3, "heard": "No. That is wrong."}]
    # The model's next request held the call and its real result, as if it
    # had made the call itself, and nothing of the third re-ask.
    second = requests[1]
    call = next(m for m in second if m.get("tool_calls"))
    assert call["tool_calls"][0]["function"]["name"] == "could_not_understand"
    result = next(m for m in second if m.get("role") == "tool")
    assert json.loads(result["content"])["data"]["result"] == "hand_off"
    assert all(RUN_4794[2] not in str(m.get("content")) for m in second)


@pytest.mark.asyncio
async def test_a_text_chat_turn_counts_from_its_checkpoint():
    # A text chat builds a new model service every turn and restores the
    # conversation from a JSON checkpoint, so the count can only come from
    # the messages -- including a tool call and its HTTP result.
    from api.services.workflow.text_chat_runner import (
        _deserialize_text_chat_checkpoint_messages,
        _serialize_text_chat_checkpoint_messages,
    )

    messages = [
        {"role": "assistant", "content": GREETING},
        {"role": "user", "content": "the customer"},
        _call("r1", "route_call"),
        _http_result("r1", "heard_unclear"),
        {"role": "assistant", "content": "Sorry, you broke up there — could you say again what the customer is avoiding?"},
        {"role": "user", "content": "No. That is wrong."},
        {"role": "assistant", "content": RUN_4794[1]},
        {"role": "user", "content": "No. That is wrong."},
    ]
    checkpoint = json.loads(json.dumps(_serialize_text_chat_checkpoint_messages(messages)))
    context = LLMContext()
    context.set_messages(_deserialize_text_chat_checkpoint_messages(checkpoint))
    context.set_tools(_tools("could_not_understand", "route_call", "transfer_call"))

    said, calls = await _reply(
        _service("could_not_understand", "transfer_call"), context, *_words(RUN_4794[2])
    )
    assert said == ""
    assert calls == [
        ("could_not_understand", {"about": "what_they_need", "times": 3, "heard": "No. That is wrong."})
    ]
