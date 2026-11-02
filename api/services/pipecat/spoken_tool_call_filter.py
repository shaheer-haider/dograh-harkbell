"""Keep a tool call the model wrote as text out of the caller's ear.

Some models now and then put a tool call in the *content* channel instead of
the tool-call channel. Nothing downstream of the LLM can tell it from speech,
so the TTS reads it out word for word: in a Harkbell staging call the agent
answered "yes, this number is fine" with

    {"destination": "assistant", "tool": "leave_a_message", "params": {...}}

and the caller heard a receptionist recite JSON. The model had produced a
well-formed call to a real tool, in the wrong channel.

This processor sits directly behind the LLM, ahead of everything that treats
its text as speech -- the TTS in a voice call, the transcript-correction
callback, the assistant context aggregator, and the text-chat capture. Text
passes through untouched until a ``{`` arrives; from there it is held only for
as long as it could still be a tool call (a quoted first key from a short list
of tool-call keys), so ordinary speech that happens to contain a brace is
released at once. A held object that closes and parses as a tool call is
removed from the stream; anything else is released as it was.

What happens to the removed call:

* **Executed** through the LLM service's own dispatcher, when it names a tool
  the current node actually advertises and has a handler for, and the same
  response carried no native tool call. It goes through
  ``_run_or_defer_function_calls`` -- the path a native call takes at the end
  of a completion -- so a lone transition still waits for whatever the
  response *did* say to finish playing, an interrupted or superseded response
  still drops its transitions, and the context gets the same assistant
  tool-call message and tool result a native call leaves. The tool's own
  handler is what judges the arguments, exactly as for a native call.
* **Dropped** otherwise (an unknown or un-advertised tool, a response that also
  made a native call, a service without that dispatcher). Silence is a far
  smaller failure than a recited JSON object, and the caller's next words
  start an ordinary turn.

One more thing follows from removing text. When the model's only content was
the removed call *and* it also made a native transition call, the service has
deferred that transition until the bot stops speaking -- which it never will,
because nothing reached the TTS. The processor releases that deferral itself
at the end of the response, the way a ``BotStoppedSpeakingFrame`` would.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallFromLLM,
    FunctionCallsFromLLMInfoFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# A held "{" is released at once unless its first key is one of these. They
# cover the shapes models write when they put a call in the content channel:
# the engine-edge shape seen in production ({"destination", "tool", "params"}),
# OpenAI's ({"name", "arguments"} or {"type": "function", "function": {...}}),
# and a few near relations. None of them opens a sentence a person would say.
TOOL_CALL_FIRST_KEYS = frozenset(
    {
        "destination",
        "recipient",
        "tool",
        "tool_name",
        "name",
        "function",
        "type",
        "id",
        "params",
        "parameters",
        "arguments",
        "args",
    }
)

_ARGUMENT_KEYS = ("params", "parameters", "arguments", "args", "input")
_FIRST_KEY = re.compile(r'\{\s*"((?:[^"\\]|\\.)*)"')
_MAX_FIRST_KEY_LENGTH = 32


@dataclass(frozen=True)
class SpokenToolCall:
    """A tool call recovered from the text channel."""

    name: str
    arguments: dict[str, Any]
    text: str


def _as_arguments(value: Any) -> dict[str, Any] | None:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if not value.strip():
            return {}
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def tool_call_from_object(obj: Any) -> tuple[str, dict[str, Any]] | None:
    """The (name, arguments) of a parsed JSON object shaped like a tool call."""
    if not isinstance(obj, dict):
        return None

    function = obj.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        arguments = _as_arguments(function.get("arguments", function.get("parameters")))
        return (function["name"], arguments) if arguments is not None else None

    for name_key, needs_arguments in (
        ("tool", False),
        ("tool_name", False),
        ("function", True),
        ("name", True),
    ):
        name = obj.get(name_key)
        if not isinstance(name, str) or not name.strip():
            continue
        present = [key for key in _ARGUMENT_KEYS if key in obj]
        # {"name": "Sophie"} is a person, not a call: a bare name only counts
        # beside an argument object.
        if needs_arguments and not present:
            continue
        arguments = _as_arguments(obj[present[0]]) if present else {}
        if arguments is None:
            continue
        return name.strip(), arguments
    return None


def _scan_object(text: str) -> int | None:
    """Index just past the JSON object opening at text[0], or None if open."""
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


_NEED_MORE = "need_more"
_NOT_A_CALL = "not_a_call"


def _classify_held(held: str) -> tuple[str, int, SpokenToolCall | None]:
    """Judge text that starts with "{".

    Returns (verdict, length, call): ``_NEED_MORE`` with nothing decided yet,
    ``_NOT_A_CALL`` with how many leading characters can be released, or
    ``"call"`` with how many characters the recovered call spans.
    """
    after = held[1:].lstrip()
    if not after:
        return _NEED_MORE, 0, None
    if after[0] != '"':
        return _NOT_A_CALL, 1, None
    match = _FIRST_KEY.match(held)
    if match is None:
        # The opening quote is there but the key has not closed yet.
        if len(after) > _MAX_FIRST_KEY_LENGTH + 2:
            return _NOT_A_CALL, 1, None
        return _NEED_MORE, 0, None
    if match.group(1) not in TOOL_CALL_FIRST_KEYS:
        return _NOT_A_CALL, 1, None

    end = _scan_object(held)
    if end is None:
        return _NEED_MORE, 0, None
    candidate = held[:end]
    try:
        parsed = json.loads(candidate)
    except ValueError:
        return _NOT_A_CALL, 1, None
    call = tool_call_from_object(parsed)
    if call is None:
        return _NOT_A_CALL, end, None
    name, arguments = call
    return "call", end, SpokenToolCall(name=name, arguments=arguments, text=candidate)


def _opens_with_tool_call_key(held: str) -> bool:
    match = _FIRST_KEY.match(held)
    return match is not None and match.group(1) in TOOL_CALL_FIRST_KEYS


def _has_alnum(text: str) -> bool:
    return any(char.isalnum() for char in text)


def _advertised_tool_names(context: Any) -> set[str]:
    tools = getattr(context, "tools", None)
    standard_tools = getattr(tools, "standard_tools", None) or []
    names = set()
    for tool in standard_tools:
        name = getattr(tool, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


class SpokenToolCallFilter(FrameProcessor):
    """Remove tool calls written as text from an LLM's spoken output.

    Args:
        llm: The LLM service directly upstream. Its dispatcher runs a recovered
            call; without one, every recovered call is dropped.
        context_provider: Returns the context the LLM is answering from. A call
            is executed only for a tool that context advertises, because the
            service keeps earlier nodes' handlers registered after a
            transition and a call to one of those must not run.
    """

    def __init__(
        self,
        *,
        llm: Any = None,
        context_provider: Callable[[], Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._llm = llm
        self._context_provider = context_provider
        self._held = ""
        self._template: LLMTextFrame | None = None
        self._calls: list[tuple[SpokenToolCall, int | None]] = []
        self._spoke = False
        self._stripped = False
        self._native_call_seen = False

    @property
    def holding(self) -> str:
        return self._held

    def _reset(self) -> None:
        self._held = ""
        self._template = None
        self._calls = []
        self._spoke = False
        self._stripped = False

    def _text_frame(self, text: str) -> LLMTextFrame:
        frame = LLMTextFrame(text)
        template = self._template
        if template is not None:
            frame.skip_tts = template.skip_tts
            frame.includes_inter_frame_spaces = template.includes_inter_frame_spaces
            frame.append_to_context = template.append_to_context
        return frame

    async def _emit(self, text: str) -> None:
        if not text:
            return
        if _has_alnum(text):
            self._spoke = True
        await self.push_frame(self._text_frame(text), FrameDirection.DOWNSTREAM)

    def _note_call(self, call: SpokenToolCall) -> None:
        self._stripped = True
        generation = getattr(self._llm, "_tool_response_generation", None)
        self._calls.append((call, generation))
        logger.warning(
            f"{self}: removed a tool call written as speech "
            f"({call.name}, {len(call.text)} chars) before it reached the caller"
        )

    async def _drain(self, *, final: bool) -> None:
        """Release or remove as much of the held text as can be decided."""
        while self._held:
            brace = self._held.find("{")
            if brace < 0:
                text, self._held = self._held, ""
                await self._emit(text)
                return
            if brace > 0:
                text, self._held = self._held[:brace], self._held[brace:]
                await self._emit(text)
                continue

            verdict, length, call = _classify_held(self._held)
            if verdict == _NEED_MORE:
                if not final:
                    return
                if not _opens_with_tool_call_key(self._held):
                    # A response ending on "{" or '{"' was never a call.
                    text, self._held = self._held, ""
                    await self._emit(text)
                    return
                # The response ended inside an object whose first key is a
                # tool-call key. Half a JSON object is not speech either.
                self._stripped = True
                logger.warning(
                    f"{self}: removed an unfinished tool call written as speech "
                    f"({len(self._held)} chars)"
                )
                self._held = ""
                return
            if verdict == _NOT_A_CALL:
                text, self._held = self._held[:length], self._held[length:]
                await self._emit(text)
                continue
            assert call is not None
            self._note_call(call)
            self._held = self._held[length:]

    async def _settle_response(self) -> None:
        """At the end of a response, run or drop what was removed from it."""
        calls, self._calls = self._calls, []
        native_call_seen, self._native_call_seen = self._native_call_seen, False
        stripped, spoke = self._stripped, self._spoke
        self._stripped = False
        self._spoke = False
        if not stripped:
            return

        llm = self._llm
        if not spoke and native_call_seen:
            # The service counted the removed text as speech and is holding a
            # lone transition for a BotStoppedSpeakingFrame no TTS will send.
            release = getattr(llm, "_run_pending_node_transition_function_calls", None)
            if release is not None and getattr(
                llm, "_pending_node_transition_function_calls", None
            ):
                logger.info(f"{self}: releasing a transition deferred for removed text")
                await release()

        if not calls:
            return
        if native_call_seen:
            for call, _ in calls:
                logger.warning(
                    f"{self}: dropped {call.name} written as speech; the same "
                    "response already called a tool"
                )
            return

        dispatch = getattr(llm, "_run_or_defer_function_calls", None)
        functions = getattr(llm, "_functions", None)
        context = self._context_provider() if self._context_provider else None
        advertised = _advertised_tool_names(context) if context is not None else set()

        runnable: list[FunctionCallFromLLM] = []
        generation = None
        for call, call_generation in calls:
            if (
                dispatch is None
                or not isinstance(functions, dict)
                or call.name not in functions
                or call.name not in advertised
            ):
                logger.warning(
                    f"{self}: dropped {call.name} written as speech; it is not a "
                    "tool this node offers"
                )
                continue
            generation = call_generation
            runnable.append(
                FunctionCallFromLLM(
                    function_name=call.name,
                    tool_call_id=f"call_spoken_{uuid.uuid4().hex[:24]}",
                    arguments=call.arguments,
                    context=context,
                )
            )

        if not runnable:
            return
        logger.info(
            f"{self}: running {[c.function_name for c in runnable]} written as "
            "speech through the tool dispatcher"
        )
        await dispatch(
            runnable,
            text_generated=spoke,
            response_generation=generation,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction is not FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            if not self._held and "{" not in frame.text:
                if _has_alnum(frame.text):
                    self._spoke = True
                await self.push_frame(frame, direction)
                return
            self._template = frame
            self._held += frame.text
            await self._drain(final=False)
            return

        if isinstance(frame, FunctionCallsFromLLMInfoFrame):
            if frame.function_calls:
                self._native_call_seen = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            self._reset()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            await self._drain(final=True)
            await self._settle_response()
            self._template = None
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (InterruptionFrame, CancelFrame, EndFrame)):
            # Whatever the interrupted response was about to say or call is
            # stale; the service drops its own transitions the same way.
            self._reset()
            self._native_call_seen = False

        await self.push_frame(frame, direction)
