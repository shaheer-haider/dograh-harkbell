"""MiniMax with a tool call written as words kept out of the reply.

Now and then MiniMax M3 returns a tool call in ``delta.content`` in its own
grammar (``<function_calls><invoke name="…">``, a bare ``<tool_call>``, or a
bare JSON object ``{"name": "say_goodbye", "arguments": {}}``) instead
of as a structured call. ``tool_markup.py`` keeps that text out of speech, but
the text still became the agent's turn in the conversation, and the model
copies its own previous turns. On a staging text chat (Harkbell round t656b)
one such reply, to "I wanna buy a gift card", was followed by the same markup
on every turn after: "Hello? Can you hear me?" got it, and so did the caller's
goodbye. A caller would have heard nothing for the rest of the call.

This service drops a response's content from the moment it becomes such
markup: all of it when the response opens with one, the rest of it when one
appears after some prose. Nothing dropped is pushed as text, so it never
reaches the assistant's turn in the context, and the next turn has nothing to
copy. Structured tool calls (``delta.tool_calls``) are untouched.

Two things more since production's twin of Mirror Labs answered "Tomorrow at
seven thirty." with ```` ```json\\n{"name": "route_call", …}\\n``` ```` (run
3881, Harkbell T-663):

- An opener split across stream chunks is still an opener. The fence came in
  one chunk and ``{`` in another, so each chunk alone looked like prose, the
  whole call went into the context, and only the speech filter caught the
  JSON — leaving the fences as the reply. A chunk's tail that could still
  become an opener is now held back until the next chunk settles it.
- A call written as JSON is a call the model meant to make. Dropping it kept
  the reply clean but routed nothing: an urgent order never reached the team.
  When the whole response was one complete ``{"name": …, "arguments": {…}}``
  for a tool this response was offered, it is handed to Pipecat as the
  structured call it should have been, and runs. Only then: after prose, or
  beside a second written call, it may be the model thinking aloud — staging
  run 3831 reasoned in words and then called ``route_call`` with a name and an
  order number lifted from the tool's own examples (T-662) — and a made-up
  order reaching the team is worse than a turn the caller repeats. The XML
  grammars are only dropped: their values are all strings, and a guessed type
  is the same trade.
"""

import json
import re
from collections.abc import AsyncIterator
from enum import StrEnum
from uuid import uuid4

from loguru import logger
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.minimax.llm import MiniMaxLLMService

_FENCE = "```"
_OPENERS = (
    "<function_calls",
    "<tool_call",
    "<minimax:tool_call",
    "<invoke",
    '{"name"',
    '{ "name"',
    '{"arguments"',
    '{ "arguments"',
    '{"parameters"',
    '{ "parameters"',
)
_OPENER_ANYWHERE = re.compile(
    r'<(?:function_calls|tool_call|minimax:tool_call|invoke)\b'
    r'|\{\s*"(?:name|arguments|parameters)"\s*:'
    r"|```"
)
# The end of a chunk that may be the start of an opener the next chunk
# completes: "<func", "{", '{ "na', "`", "``". Held back, never dropped.
_PARTIAL_OPENER_AT_END = re.compile(r'(?:<[\w:]*|\{\s*(?:"[a-z_]*"?\s*)?|`{1,2})$')
_FENCE_TAG = re.compile(r"```([\w-]*)")
_WRITTEN_JSON_CALL = re.compile(r'\{\s*"(?:name|arguments|parameters)"\s*:')


class _MarkupState(StrEnum):
    DETECTING = "detecting"
    PASS = "pass"
    DROP = "drop"


def written_json_calls(text: str) -> list[tuple[str, dict]]:
    """Every complete ``{"name": …, "arguments": {…}}`` object in ``text``, in order.

    Fences, prose and anything that does not parse are skipped. ``arguments``
    may be an object or a JSON string holding one, and ``parameters`` is
    accepted for it, because MiniMax has written both.
    """
    calls: list[tuple[str, dict]] = []
    decoder = json.JSONDecoder()
    position = 0
    while True:
        found = _WRITTEN_JSON_CALL.search(text, position)
        if found is None:
            return calls
        try:
            value, end = decoder.raw_decode(text, found.start())
        except json.JSONDecodeError:
            position = found.start() + 1
            continue
        position = end
        if not isinstance(value, dict):
            continue
        name = value.get("name")
        arguments = value.get("arguments", value.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                continue
        if isinstance(name, str) and name and isinstance(arguments, dict):
            if (name, arguments) not in calls:
                calls.append((name, arguments))


def _offered_tool_names(context: LLMContext | None) -> set[str] | None:
    """The tools this response was offered, or None when that cannot be read."""
    if context is None:
        return None
    tools = getattr(context, "tools", None)
    standard = getattr(tools, "standard_tools", None)
    if standard is None:
        return None
    return {tool.name for tool in standard if getattr(tool, "name", None)}


def _content_chunk(text: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="harkbell-held-text",
        choices=[Choice(index=0, delta=ChoiceDelta(content=text), finish_reason=None)],
        created=0,
        model="",
        object="chat.completion.chunk",
    )


def _tool_call_chunk(index: int, name: str, arguments: dict) -> ChatCompletionChunk:
    call = ChoiceDeltaToolCall(
        index=index,
        id=f"call_{uuid4().hex[:24]}",
        type="function",
        function=ChoiceDeltaToolCallFunction(name=name, arguments=json.dumps(arguments)),
    )
    return ChatCompletionChunk(
        id="harkbell-written-call",
        choices=[Choice(index=0, delta=ChoiceDelta(tool_calls=[call]), finish_reason=None)],
        created=0,
        model="",
        object="chat.completion.chunk",
    )


class HarkbellMiniMaxLLMService(MiniMaxLLMService):
    """``MiniMaxLLMService`` that never lets a written tool call become the reply."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reset_markup_state()

    def _reset_markup_state(self):
        self._markup_state = _MarkupState.DETECTING
        self._markup_buffer = ""
        self._pass_tail = ""
        self._dropped_text = ""
        self._saw_structured_call = False
        self._said_something = False

    def _start_dropping(self, dropped: str = ""):
        self._markup_state = _MarkupState.DROP
        self._markup_buffer = ""
        self._pass_tail = ""
        self._dropped_text = dropped
        logger.warning("MiniMax wrote a tool call as text; dropping it from the reply")

    def _pass_through(self, text: str) -> str | None:
        """Text known not to open with markup: cut it at any opener inside it.

        A tail that could still become an opener waits for the next chunk.
        """
        text = self._pass_tail + text
        self._pass_tail = ""
        found = _OPENER_ANYWHERE.search(text)
        if found is not None:
            self._start_dropping(text[found.start() :])
            return text[: found.start()] or None
        partial = _PARTIAL_OPENER_AT_END.search(text)
        if partial is not None and partial.start() < len(text):
            self._pass_tail = text[partial.start() :]
            text = text[: partial.start()]
        return text or None

    async def _filter_tool_markup(self, text: str) -> str | None:
        """One chunk of visible content, with any written tool call taken out."""
        if self._markup_state == _MarkupState.DROP:
            self._dropped_text += text
            return None
        if self._markup_state == _MarkupState.PASS:
            return self._pass_through(text)

        self._markup_buffer += text
        head = self._markup_buffer.lstrip()
        if not head:
            return None
        if head.startswith(_FENCE):
            # A response that opens with a code fence: lose the fence and its
            # language tag, then judge what is inside it as the response.
            tag = _FENCE_TAG.match(head)
            after = head[tag.end() :]
            if not after:
                return None  # "```" or "```js": the tag may not be over yet
            self._markup_buffer = after if after[0].isspace() else head[len(_FENCE) :]
            return await self._filter_tool_markup("")
        if any(head.startswith(opener) for opener in _OPENERS) or _OPENER_ANYWHERE.match(head):
            self._start_dropping(head)
            return None
        if any(opener.startswith(head) for opener in _OPENERS) or _PARTIAL_OPENER_AT_END.fullmatch(head):
            # "<", "<func", "<tool_c", "{", '{\n  "na', "`" — could still become an opener.
            return None
        self._markup_state = _MarkupState.PASS
        buffered, self._markup_buffer = self._markup_buffer, ""
        return self._pass_through(buffered)

    def _held_text(self) -> str:
        """What was held back when the response ended, and is ordinary text after all."""
        if self._markup_state == _MarkupState.DETECTING:
            held = self._markup_buffer
        elif self._markup_state == _MarkupState.PASS:
            held = self._pass_tail
        else:
            held = ""
        self._markup_buffer = ""
        self._pass_tail = ""
        # A lone "<" is a word; a backtick, or a fence that never closed over
        # anything, never is.
        stripped = held.strip()
        return "" if not stripped or stripped.startswith("`") else held

    def _written_calls_to_run(self, context: LLMContext | None) -> list[tuple[str, dict]]:
        if self._markup_state != _MarkupState.DROP or self._saw_structured_call:
            return []
        calls = written_json_calls(self._dropped_text)
        if len(calls) != 1 or self._said_something:
            if calls:
                logger.warning(
                    f"MiniMax wrote {[name for name, _ in calls]} as text beside other words; not running"
                )
            return []
        name, arguments = calls[0]
        offered = _offered_tool_names(context)
        if (offered is not None and name not in offered) or not self.has_function(name):
            logger.warning(f"MiniMax wrote a call to {name!r} as text; it is not a tool here")
            return []
        logger.warning(f"MiniMax wrote a call to {name!r} as text; running it as a call")
        return [(name, arguments)]

    async def get_chat_completions(self, context: LLMContext) -> AsyncIterator[ChatCompletionChunk]:
        stream = await super().get_chat_completions(context)
        return self._drop_tool_markup(stream, context)

    async def _drop_tool_markup(
        self, stream: AsyncIterator[ChatCompletionChunk], context: LLMContext | None = None
    ) -> AsyncIterator[ChatCompletionChunk]:
        try:
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if delta.tool_calls:
                        self._saw_structured_call = True
                    if delta.content:
                        delta.content = await self._filter_tool_markup(delta.content)
                        if delta.content and delta.content.strip():
                            self._said_something = True
                yield chunk
            held = self._held_text()
            if held:
                yield _content_chunk(held)
            for index, (name, arguments) in enumerate(self._written_calls_to_run(context)):
                yield _tool_call_chunk(index, name, arguments)
        finally:
            self._markup_buffer = ""
            self._pass_tail = ""
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

    async def _process_context(self, context: LLMContext):
        self._reset_markup_state()
        await super()._process_context(context)
