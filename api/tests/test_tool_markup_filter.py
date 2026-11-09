import asyncio

import pytest

from api.services.pipecat.tool_markup import tool_markup_filter

# What a staging text chat's agent produced for a client running late
# (round t651a), but for the number: one take_message call, as words, twice.
T651A = """<function_calls>
<invoke name="invoke name="take_message">
<parameter name="customer_name">Owen</parameter>
<parameter name="reason">Running late for 2pm appointment with Juno today</parameter>
<parameter name="customer_phone">+15035550000</parameter>
</invoke>
</function_calls>
<function_calls>
<invoke name="invoke name="take_message">
<parameter name="customer_name">Owen</parameter>
</invoke>
</function_calls>"""


def run(text: str) -> str:
    return asyncio.run(tool_markup_filter().filter(text))


def test_a_reply_that_was_only_a_tool_call_says_nothing():
    assert run(T651A) == ""


def test_a_lone_opener_says_nothing():
    assert run("<tool_call>") == ""


def test_prose_around_the_markup_is_kept():
    assert run(f"Sure, one moment. {T651A} Done.") == "Sure, one moment. Done."
    assert run("Let me check. <tool_call>") == "Let me check."


@pytest.mark.parametrize(
    "half,spoken",
    [
        # A block cut in two by sentence aggregation: neither half reads a tag
        # or a parameter value it can see whole.
        ('<function_calls>\n<invoke name="take_message">\n<parameter name="customer_name">Owen</parameter>', ""),
        ("</invoke>\n</function_calls>", ""),
    ],
)
def test_each_half_of_a_split_block_loses_its_markup(half, spoken):
    assert run(half) == spoken


def test_the_upstream_function_form_still_goes():
    assert run('Okay <function=end_call>{"x": 1}</function> bye') == "Okay bye"


@pytest.mark.parametrize(
    "text",
    ["We're open eleven to seven.", "Anything under 2 inches is the minimum.", "[calmly] Got it, Owen."],
)
def test_ordinary_speech_is_untouched(text):
    assert run(text) == text
