"""A number read back is the number the transcriber heard.

Harkbell. Copy run 7103 (2026-11-21, AssemblyAI, DeepSeek): the caller read
an order number and the transcriber wrote "order 786. 6099." -- seven digits.
The agent read back "seven, eight, six, zero, nine, nine": six, one of the
two sixes gone. The caller said yes to it, and route_call filed 786099. The
read-back exists so the caller can catch a mishearing; a digit the model
drops on its own is one the transcriber never got wrong, and the caller's
"yes" to a quick string of digits does not catch it.

So a reply's read-back is held to the digits the caller was transcribed
saying. A read-back that is a part of what they said -- the corrected half of
"four one seven, sorry, four one eight", or a number put together from two
turns -- is the model doing its job and stays. One that differs from a
number they said by a digit or two, and is no part of anything they said,
is the model's slip: the run is said as the transcriber wrote it instead.
Anything further from it is left alone -- it is not a slip of this kind, and
a guess at it would be worse than the model's.
"""

import re

from loguru import logger

_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_REPEATS = {"double": 2, "triple": 3}
# A word, a run of digits, or anything else. Letters glued to digits ("5th",
# "441B") are one token and are not digits.
_TOKEN = re.compile(r"[A-Za-z0-9']+")
# What may stand between two digits of one number: spaces, commas, full stops,
# dashes, and "and" ("zero, and four").
_JOINER = re.compile(r"^(?:[\s,.;:\-–—]|and\b)*$", re.IGNORECASE)
# The shortest run that is a read-back rather than a count or a time.
MIN_DIGITS = 4
# How far a read-back may be from a number the caller said and still be a slip.
MAX_SLIP = 2
# The shortest number said that a read-back is corrected to: a four-digit one
# is too close to the last four of a caller ID read back from the card.
MIN_HEARD_DIGITS = 5
# How many of the caller's turns before the latest are read for the number.
EARLIER_TURNS = 3


def _runs(text: str) -> list[tuple[str, int, int]]:
    """Each run of digits in ``text``: its digits, and where it starts and ends."""
    runs: list[tuple[str, int, int]] = []
    digits = ""
    start = end = 0
    repeat = 1

    def close():
        nonlocal digits, repeat
        if digits:
            runs.append((digits, start, end))
        digits, repeat = "", 1

    for match in _TOKEN.finditer(text or ""):
        token = match.group().lower()
        open_run = bool(digits) or repeat > 1
        joined = open_run and _JOINER.match(text[end : match.start()]) is not None
        if open_run and not joined:
            close()
        if token == "and" and digits:
            # "zero, and four": the run goes on if a digit follows.
            continue
        if token in _REPEATS:
            if not digits:
                start = match.start()
            repeat = _REPEATS[token]
            end = match.end()
            continue
        if token.isdigit():
            value = token
        elif token in _DIGIT_WORDS and (token != "oh" or digits):
            # "oh" is a zero only inside a number: "Oh, the order is" is a word.
            value = _DIGIT_WORDS[token]
        else:
            close()
            continue
        if not digits and repeat == 1:
            start = match.start()
        digits += value * repeat if len(value) == 1 else value
        repeat = 1
        end = match.end()
    close()
    return runs


def _distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


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


def callers_recent_words(messages) -> tuple[list[str], list[str]]:
    """The caller's turns since the agent last spoke, and a few before them."""
    latest: list[str] = []
    earlier: list[str] = []
    spoke = False
    for message in reversed(list(messages or [])):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant" and _text(message).strip():
            spoke = True
        elif role == "user":
            text = _text(message)
            if not spoke:
                latest.insert(0, text)
            elif len(earlier) < EARLIER_TURNS:
                earlier.insert(0, text)
            else:
                break
    return latest, earlier


class ReadBackGuard:
    """One reply, sentence by sentence, with its read-backs held to the transcript."""

    def __init__(self, latest: list[str], earlier: list[str] | None = None):
        self._heard = [digits for text in latest for digits, _, _ in _runs(text)]
        if len(self._heard) > 1:
            # Copy run 7246: "Order 644. For one 7208." is two runs either side
            # of a misheard word, and the read-back dropped the "one" from
            # the whole: 6447208 for 64417208. Each part on its own was too
            # far from it to be the slip it was.
            self._heard.append("".join(self._heard))
        said = [digits for text in [*(earlier or []), *latest] for digits, _, _ in _runs(text)]
        # Everything the caller said, in order, as one string: a number put
        # together from several turns is a part of it.
        self._everything = "".join(said)
        self._numbers = [*said]
        self.corrected: list[tuple[str, str]] = []

    @classmethod
    def from_context(cls, context) -> "ReadBackGuard | None":
        latest, earlier = callers_recent_words(getattr(context, "messages", None))
        guard = cls(latest, earlier)
        return guard if any(len(digits) >= MIN_DIGITS for digits in guard._heard) else None

    def _explained(self, read: str) -> bool:
        return read in self._everything or any(read in number for number in self._numbers)

    def _instead(self, read: str) -> str | None:
        best = None
        for heard in self._heard:
            if len(heard) < MIN_HEARD_DIGITS:
                continue
            distance = _distance(read, heard)
            if 0 < distance <= MAX_SLIP and distance * 3 <= len(heard):
                if best is None or distance < best[0]:
                    best = (distance, heard)
        return best[1] if best else None

    def say(self, sentence: str) -> str:
        """The sentence, with a read-back the model slipped on said as heard."""
        if not sentence.strip():
            return sentence
        out = sentence
        for read, start, end in reversed(_runs(sentence)):
            if len(read) < MIN_DIGITS or self._explained(read):
                continue
            heard = self._instead(read)
            if heard is None:
                continue
            spoken = ", ".join(_NAMES[int(d)] for d in heard)
            original = sentence[start:end]
            if original[:1].isupper():
                spoken = spoken[:1].upper() + spoken[1:]
            out = out[:start] + spoken + out[end:]
            self.corrected.append((read, heard))
            logger.warning(
                f"Read-back {read} is not what the caller was heard saying; saying {heard} instead"
            )
        return out
