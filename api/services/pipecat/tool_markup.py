"""Tool-call markup a language model wrote as words, kept out of speech.

Pipecat's XMLFunctionTagFilter removes only the ``<function=name>...</function>``
form. MiniMax M3 now and then returns a tool call as text in its own grammar
instead of as a structured call: a whole
``<function_calls><invoke name="take_message"><parameter ...>`` block, or a
lone ``<tool_call>`` opener. That text reaches the TTS service as the agent's
reply, and the caller hears the tag names, the parameter names and the
values, their own phone number among them. Two staging text chats caught it
(a ``take_message`` call twice over in one reply, and a bare ``<tool_call>``),
on the same model every phone call runs on.

These patterns ride on the existing filter. They remove every closed block of
those grammars with what is inside it, every ``<parameter>`` element, and any
stray tag of those names, so that a block split across two TTS aggregations
loses its tags and parameter values in each half rather than being read out.
Prose around the markup is kept.
"""

from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter

_TOOL_TAGS = r"function_calls|tool_call|minimax:tool_call|invoke"

TOOL_MARKUP_PATTERNS = [
    rf"<({_TOOL_TAGS})\b[^>]*>.*?</\1\s*>",
    r"<parameter\b[^>]*>.*?</parameter\s*>",
    rf"</?(?:{_TOOL_TAGS}|parameter)\b[^>]*>",
    # A turn that "declines to speak" in an agent framework's own token:
    # staging round t563a's agent answered "Thanks, bye" by saying NO_REPLY
    # aloud. The underscore and tag forms are never prose, so they go
    # anywhere; "[no reply]" only when it is the whole of what would be said.
    r"<no_reply\s*/?>",
    r"\[\s*no_reply\s*\]",
    r"\bNO_REPLY\b",
    r"^\s*\[no reply\]\s*$",
]


def tool_markup_filter() -> XMLFunctionTagFilter:
    """The TTS text filter every speech service is built with."""
    return XMLFunctionTagFilter(custom_patterns=TOOL_MARKUP_PATTERNS)
