"""A number read back is the number the transcriber heard (read_backs.py).

Copy run 7103 (2026-11-21): the transcriber wrote "order 786. 6099." and the
agent read back "seven, eight, six, zero, nine, nine"; the caller said yes and
786099 was filed.
"""

from types import SimpleNamespace

from api.services.pipecat.read_backs import ReadBackGuard, _runs, callers_recent_words
from api.services.pipecat.spoken_reasoning import SpokenReasoningFilter


def _context(*turns):
    return SimpleNamespace(messages=[{"role": role, "content": text} for role, text in turns])


def test_digits_are_read_the_way_transcribers_write_them():
    assert [d for d, _, _ in _runs("Vivid Seat Support, order 786. 6099.")] == ["7866099"]
    assert [d for d, _, _ in _runs("Order. 6.\n4, 4. 1, 7. 208.")] == ["64417208"]
    assert [d for d, _, _ in _runs("5, 5, 1, 2, 0, and\n4.")] == ["551204"]
    assert [d for d, _, _ in _runs("Order 1004. Live. 811.")] == ["1004", "811"]
    assert [d for d, _, _ in _runs("one double zero four, oh five")] == ["100405"]
    # A date and a word are not digits; "oh" is a word until a number has begun.
    assert _runs("December 5th.") == []
    assert [d for d, _, _ in _runs("Oh, it's one")] == ["1"]


def test_a_digit_the_model_dropped_is_said_as_the_transcriber_heard_it():
    guard = ReadBackGuard(["Vivid Seat Support, order 786. 6099."])
    said = guard.say("Seven, eight, six, zero, nine, nine — did I get that right?")
    assert said == "Seven, eight, six, six, zero, nine, nine — did I get that right?"
    assert guard.corrected == [("786099", "7866099")]


def test_a_read_back_of_what_was_said_is_left_alone():
    guard = ReadBackGuard(["Order. 6.\n4, 4. 1, 7. 208."])
    sentence = "Let me read that back — six, four, four, one, seven, two, zero, eight. Is that right?"
    assert guard.say(sentence) == sentence
    guard = ReadBackGuard(["5, 5, 1, 2, 0, and\n4."])
    sentence = "Five, five, one, two, zero, and four — did I get that right?"
    assert guard.say(sentence) == sentence


def test_a_number_put_together_from_two_turns_is_left_alone():
    # The first digits in one turn, the rest in the next.
    guard = ReadBackGuard(["4, 4, 1, 7, 2, 0, 8."], ["Order six."])
    sentence = "Six, four, four, one, seven, two, zero, eight — is that right?"
    assert guard.say(sentence) == sentence


def test_the_corrected_half_of_a_number_is_left_alone():
    guard = ReadBackGuard(["4, 1, 7, sorry, 4, 1, 8, 2, 2."])
    sentence = "Four, one, eight, two, two — is that right?"
    assert guard.say(sentence) == sentence


def test_a_read_back_far_from_anything_said_is_left_alone():
    guard = ReadBackGuard(["Order 64417208."])
    sentence = "Five, five, one, two, zero, eight, four — is that right?"
    assert guard.say(sentence) == sentence
    assert guard.corrected == []


def test_the_last_four_of_a_caller_id_is_not_turned_into_a_short_number_said():
    guard = ReadBackGuard(["It's 3612."])
    sentence = "Is the number ending three, six, one, one the best one to reach you?"
    assert guard.say(sentence) == sentence


def test_the_guard_reads_the_turns_since_the_agent_last_spoke():
    context = _context(
        ("assistant", "Hi, this is Emma. How can I help?"),
        ("user", "Order six."),
        ("assistant", "Mm-hmm, go on."),
        ("user", "4, 4, 1, 7, 2, 0, 8."),
    )
    latest, earlier = callers_recent_words(context.messages)
    assert latest == ["4, 4, 1, 7, 2, 0, 8."]
    assert earlier == ["Order six."]
    assert ReadBackGuard.from_context(_context(("user", "Hello, I'm calling about an order."))) is None


def test_the_reply_filter_corrects_a_streamed_read_back():
    guard = ReadBackGuard.from_context(_context(("user", "Vivid Seat Support, order 786. 6099.")))
    reply = SpokenReasoningFilter(readbacks=guard)
    said = ""
    for piece in ["Seven, eight, ", "six, zero, nine, ", "nine — did I get ", "that right?"]:
        said += reply.feed(piece)
    said += reply.finish()
    assert said == "Seven, eight, six, six, zero, nine, nine — did I get that right?"


def test_a_digit_dropped_from_a_number_split_by_a_misheard_word_is_put_back():
    # Copy run 7246: "four" written as "For", and "one" dropped from the read-back.
    guard = ReadBackGuard(["Order 644. For one 7208."])
    said = guard.say("Let me read that back to you: six, four, four, seven, two, zero, eight. Is that right?")
    assert said == "Let me read that back to you: six, four, four, one, seven, two, zero, eight. Is that right?"
