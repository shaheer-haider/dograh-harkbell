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
"""

import re
from collections.abc import AsyncIterator
from enum import StrEnum

from loguru import logger
from openai.types.chat import ChatCompletionChunk

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.minimax.llm import MiniMaxLLMService

_OPENERS = ("<function_calls", "<tool_call", "<minimax:tool_call", "<invoke", '{"name"', '{ "name"')
_OPENER_ANYWHERE = re.compile(r'<(?:function_calls|tool_call|minimax:tool_call|invoke)\b|\{\s*"name"\s*:')


class _MarkupState(StrEnum):
    DETECTING = "detecting"
    PASS = "pass"
    DROP = "drop"


class HarkbellMiniMaxLLMService(MiniMaxLLMService):
    """``MiniMaxLLMService`` that never lets a written tool call become the reply."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reset_markup_state()

    def _reset_markup_state(self):
        self._markup_state = _MarkupState.DETECTING
        self._markup_buffer = ""

    def _start_dropping(self):
        self._markup_state = _MarkupState.DROP
        self._markup_buffer = ""
        logger.warning("MiniMax wrote a tool call as text; dropping it from the reply")

    def _pass_through(self, text: str) -> str | None:
        """Text known not to open with markup: cut it at any opener inside it."""
        found = _OPENER_ANYWHERE.search(text)
        if found is None:
            return text
        self._start_dropping()
        return text[: found.start()] or None

    async def _filter_tool_markup(self, text: str) -> str | None:
        """One chunk of visible content, with any written tool call taken out."""
        if self._markup_state == _MarkupState.DROP:
            return None
        if self._markup_state == _MarkupState.PASS:
            return self._pass_through(text)

        self._markup_buffer += text
        head = self._markup_buffer.lstrip()
        if not head:
            return None
        if any(head.startswith(opener) for opener in _OPENERS):
            self._start_dropping()
            return None
        if any(opener.startswith(head) for opener in _OPENERS):
            # "<", "<func", "<tool_c" — could still become an opener.
            return None
        self._markup_state = _MarkupState.PASS
        buffered, self._markup_buffer = self._markup_buffer, ""
        return self._pass_through(buffered)

    async def get_chat_completions(self, context: LLMContext) -> AsyncIterator[ChatCompletionChunk]:
        stream = await super().get_chat_completions(context)
        return self._drop_tool_markup(stream)

    async def _drop_tool_markup(
        self, stream: AsyncIterator[ChatCompletionChunk]
    ) -> AsyncIterator[ChatCompletionChunk]:
        completed = False
        try:
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        delta.content = await self._filter_tool_markup(delta.content)
                yield chunk
            completed = True
        finally:
            # A reply that ended while it could still have been an opener
            # ("<" and nothing after) was ordinary text after all.
            if completed and self._markup_state == _MarkupState.DETECTING and self._markup_buffer.strip():
                await self._push_llm_text(self._markup_buffer)
            self._markup_buffer = ""
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

    async def _process_context(self, context: LLMContext):
        self._reset_markup_state()
        await super()._process_context(context)
