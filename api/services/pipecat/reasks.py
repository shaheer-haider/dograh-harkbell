"""Three failures to understand in a row hand the call on, counted by the engine.

Mirror Labs' rule: after the agent fails to understand the caller three times,
the call goes to the team. The count used to live in the model -- the prompt
told it to call ``could_not_understand`` (``about``, ``times``, ``heard``)
instead of asking again, and the tool answers ``ask_again`` until ``times`` is
3 and then ``hand_off`` -- and MiniMax very often just asks again in its own
words without calling it, so the count never moves:

- Run 4793 (production copy, workflow 105): "Sorry, I didn't catch that --
  could you say the order number again, digit by digit?", "Sorry, you broke
  up there -- which marketplace is the order on?", "Sorry, I still didn't
  catch that -- could you say it one more time?", "Sorry, you broke up there
  -- could you say your name again?" ... and ``could_not_understand`` only
  once, at ``times=2``. Never handed on.
- Run 4794: the caller said "No. That is wrong." three times and heard
  "Sorry about that -- what were you calling about?", "I'm sorry -- could you
  tell me what you need?", "I'm sorry -- could you say that again?". No tool
  call at all.

So the engine counts, on a step that offers ``could_not_understand``. A reply
that asks the caller to repeat themselves, or says it could not hear or
understand them, is a re-ask (``is_reask``; every pattern is here, with tests
on the lines above). A ``could_not_understand`` that answered ``ask_again``
and a ``route_call`` that answered ``heard_unclear`` count as one re-ask
together with the reply they produced. Any other reply is progress and starts
the count again. The count is read back from the conversation itself
(``reask_count``), so a text chat, which builds a new model service every
turn, counts the same as a call. When the model is about to say the third
re-ask in a row, the reply is held back before it reaches speech and the
engine calls ``could_not_understand`` with ``times`` 3 in its place
(``minimax_llm.py``), through the ordinary tool path, so the model then
follows the tool's own hand-off instructions.
"""

import json
import re
from dataclasses import dataclass, field

from api.services.pipecat.cut_off import CUT_OFF_MARKER

COULD_NOT_UNDERSTAND = "could_not_understand"
THRESHOLD = 3
_HAND_OFF_TOOLS = {COULD_NOT_UNDERSTAND, "transfer_call"}

_I = re.IGNORECASE
_APOLOGY = r"(?:sorry|apologi[sz]e)"
# Each pattern is matched against the reply as spoken so far.
REASK_PATTERNS = [
    # "I didn't catch that", "I still didn't get your name"
    re.compile(
        r"\bI\s+(?:still\s+|just\s+|really\s+)?(?:didn'?t|did\s+not|couldn'?t|could\s+not|can'?t|cannot)\s+"
        # "get" only of words: "I can't get anyone from the team on the line
        # right now" is the line's own out-of-hours sentence, and "I can't get
        # you a refund" a plain answer; neither asks the caller to repeat.
        r"(?:quite\s+)?(?:catch|hear|make\s+out|understand|"
        r"get(?!\s+(?:you\b|anyone|someone|somebody|them\b|hold|through|into|to\b|a\b|an\b)))\b",
        _I,
    ),
    # "could you say the order number again", "can you repeat that"
    re.compile(
        r"\b(?:could|can|would|will)\s+you\s+(?:please\s+)?(?:say|repeat|give\s+me|tell\s+me)\b[^.?!]{0,60}\bagain\b",
        _I,
    ),
    re.compile(r"\b(?:could|can|would)\s+you\s+(?:please\s+)?repeat\b", _I),
    # "Say that again?"
    re.compile(r"(?:^|[.!?—–-]\s*)(?:please\s+)?say\s+(?:that|it)\s+again\b", _I),
    # "could you say it one more time?" -- asked of the caller, not offered
    re.compile(r"\byou\b[^.?!]{0,60}\b(?:one\s+more\s+time|once\s+more)\b[^.!]*\?", _I),
    re.compile(r"\byou\s+(?:broke\s+up|(?:are|'re)\s+breaking\s+up|cut\s+out)\b", _I),
    re.compile(r"\btrouble\s+(?:hearing|understanding)\b", _I),
    # After an apology, asking afresh what they want is a re-ask too.
    re.compile(
        rf"\b{_APOLOGY}\b[^?]*\b(?:what\s+(?:were|are)\s+you\s+calling\s+about|"
        r"(?:tell\s+me\s+)?what\s+you\s+need)\b",
        _I,
    ),
]
_ORDER_NUMBER = re.compile(r"\border\s+(?:number|no\.?)\b", _I)
_CUT_OFF_SUFFIX = re.compile(rf"\s*—\s*{re.escape(CUT_OFF_MARKER)}\s*$")


def is_reask(text: str) -> bool:
    """Whether a reply asks the caller to repeat, or says it could not follow them."""
    # The model writes "didn’t" with a curly apostrophe as often as a
    # straight one, and the patterns spell it straight.
    text = _CUT_OFF_SUFFIX.sub("", text or "").replace("\u2019", "'")
    return any(pattern.search(text) for pattern in REASK_PATTERNS)


def mentions_order_number(text: str) -> bool:
    return bool(_ORDER_NUMBER.search(text or ""))


@dataclass
class ReAskCount:
    """The conversation's run of consecutive re-asks, as it stands."""

    count: int = 0
    # The last re-ask is a tool result whose reply is the one being made now.
    awaiting_reply: bool = False
    texts: list[str] = field(default_factory=list)
    caller_said: str = ""
    # The call was just handed on and the reply being made now is the one
    # that says so -- "I'm having trouble understanding, let me put you
    # through" -- which is not a re-ask whatever its words.
    handing_off: bool = False


def _text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _tool_result(content) -> str | None:
    """``result`` of a tool's answer, whether bare or in the HTTP tool's envelope."""
    try:
        value = json.loads(content) if isinstance(content, str) else content
    except (TypeError, ValueError):
        return None
    for candidate in (value, value.get("data") if isinstance(value, dict) else None):
        if isinstance(candidate, dict) and isinstance(candidate.get("result"), str):
            return candidate["result"]
    return None


def reask_count(messages) -> ReAskCount:
    """Count the consecutive re-asks at the end of a conversation."""
    from api.services.workflow.pipecat_engine_callbacks import IDLE_PROMPTS

    state = ReAskCount()
    names: dict[str, str] = {}
    handed_off = False

    def reset():
        state.count = 0
        state.awaiting_reply = False
        state.texts = []

    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            words = _text(message).strip()
            if words and words not in IDLE_PROMPTS:
                state.caller_said = words
        elif role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                names[call.get("id", "")] = function.get("name", "")
                if function.get("name") == "transfer_call":
                    reset()
                    handed_off = True
            text = _CUT_OFF_SUFFIX.sub("", _text(message)).strip()
            if not text:
                continue
            if handed_off:
                # The reply that hands the call on is not a re-ask, whatever
                # it says ("I'm having trouble understanding -- let me put you
                # through").
                handed_off = False
            elif state.awaiting_reply:
                # Counted already, with the tool result that produced it.
                state.awaiting_reply = False
                state.texts[-1] = text
            elif is_reask(text):
                state.count += 1
                state.texts.append(text)
            else:
                reset()
        elif role == "tool":
            name = names.get(message.get("tool_call_id", ""), "")
            result = _tool_result(message.get("content"))
            if (name == COULD_NOT_UNDERSTAND and result == "ask_again") or (
                name == "route_call" and result == "heard_unclear"
            ):
                state.count += 1
                state.awaiting_reply = True
                state.texts.append("")
            elif name == COULD_NOT_UNDERSTAND and result == "hand_off":
                reset()
                handed_off = True
    state.handing_off = handed_off
    return state


class ReAskGuard:
    """One reply, judged as it is spoken against the run of re-asks before it."""

    def __init__(self, before: ReAskCount):
        self.before = before
        self.spoken = ""
        self.is_reask = False
        self.holding = False
        self.held: list[str] = []

    @property
    def number(self) -> int:
        """Which re-ask in a row this reply is, when it is one."""
        return self.before.count if self.before.awaiting_reply else self.before.count + 1

    def pass_on(self, piece: str) -> bool:
        """Whether a piece of the reply may be spoken now."""
        if self.holding:
            self.held.append(piece)
            return False
        self.spoken += piece
        if self.before.handing_off:
            return True
        if piece.strip() and not self.is_reask and is_reask(self.spoken):
            self.is_reask = True
        if self.is_reask and self.number >= THRESHOLD:
            # The third: not a word of the ask is said.
            self.holding = True
            self.held.append(piece)
            return False
        return True

    def hand_off_arguments(self) -> dict:
        texts = [*self.before.texts, self.spoken]
        about = (
            "order_number"
            if any(mentions_order_number(text) for text in texts)
            else "what_they_need"
        )
        return {"about": about, "times": THRESHOLD, "heard": self.before.caller_said}


def hands_off_itself(tool_names: set[str]) -> bool:
    """Whether the model's own reply already hands the call on."""
    return bool(tool_names & _HAND_OFF_TOOLS)
