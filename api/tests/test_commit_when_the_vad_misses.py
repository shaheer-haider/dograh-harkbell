"""A caller the voice detector never hears is still answered.

Copy runs 6978 and 6980 (2026-11-20), the operator on a mobile line from
Pakistan: Scribe heard every word ("Hello? Hello?", "Hello. Uh, I'm calling
about an order…"), and those words stopped the greeting, but Silero scored the
voice 0.4–0.66 against its 0.7 bar and never reported the caller speaking, so
no commit was sent. On 6980 the words sat uncommitted for 19 s; on 6978 the
first half of a sentence went through on a short VAD blip and everything the
caller said after it was never answered. Boosting the audio did not change the
VAD's verdict, so the transcriber's own words have to be able to end a turn.

These run the same pipeline pieces as test_commit_through_an_audio_gap.py, with
speech the energy VAD scores below its threshold.
"""

import numpy as np
import pytest

from pipecat.frames.frames import InputAudioRawFrame
from pipecat.tests.utils import SleepFrame

from api.tests.test_commit_through_an_audio_gap import (
    FRAME_MS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    WORD_BASE,
    _call,
    _paced,
    _silence,
    _speech,
)


def _quiet_speech(word: int, seconds: float) -> list[InputAudioRawFrame]:
    """Speech the transcriber hears and the VAD scores as no voice."""
    rng = np.random.default_rng(100 + word)
    frames = []
    for _ in range(int(seconds * 1000 / FRAME_MS)):
        samples = rng.integers(-1200, 1200, FRAME_SAMPLES, dtype=np.int16)
        samples[0] = WORD_BASE + word
        frames.append(
            InputAudioRawFrame(audio=samples.tobytes(), sample_rate=SAMPLE_RATE, num_channels=1)
        )
    return frames


def _heard(result: dict) -> str:
    return " ".join(text for _, _, text in result["turns"])


@pytest.mark.asyncio
async def test_a_sentence_the_vad_never_hears_is_answered_soon_after_the_words_stop():
    words = ["Hello.", "I'm", "calling", "about", "an", "order."]
    frames = [*_paced(_silence(0.4))]
    for i in range(len(words)):
        frames += _paced(_quiet_speech(i, 0.3))
    frames += _paced(_silence(6.0))
    result = await _call(frames, words)
    assert result["turns"], f"the turn never ended: {result}"
    assert all(w in _heard(result) for w in words), f"words were lost: {result}"
    assert result["turns"][-1][0] <= 2.0, f"the turn ended late: {result}"


@pytest.mark.asyncio
async def test_two_words_the_vad_never_hears_are_answered():
    # Run 6980: "Hello?" twice, quietly, and nothing else.
    words = ["Hello?", "Hello?"]
    frames = [
        *_paced(_silence(0.4)),
        *_paced(_quiet_speech(0, 0.4)),
        *_paced(_silence(0.6)),
        *_paced(_quiet_speech(1, 0.4)),
        *_paced(_silence(6.0)),
    ]
    result = await _call(frames, words)
    assert result["turns"], f"the turn never ended: {result}"
    assert _heard(result).count("Hello?") == 2, f"words were lost: {result}"
    assert result["turns"][-1][0] <= 2.0, f"the turn ended late: {result}"


@pytest.mark.asyncio
async def test_words_after_a_short_vad_blip_are_answered_too():
    # Run 6978: the VAD caught the start of the sentence and nothing after it.
    words = ["Hello,", "about", "an", "order,", "the", "customer", "who", "bought", "it."]
    frames = [*_paced(_silence(0.4)), *_paced(_speech(0, 0.4))]
    for i in range(1, len(words)):
        frames += _paced(_quiet_speech(i, 0.3))
    frames += _paced(_silence(6.0))
    result = await _call(frames, words)
    assert result["turns"], f"the turn never ended: {result}"
    assert all(w in _heard(result) for w in words), f"words were lost: {result}"
    assert result["turns"][-1][0] <= 2.5, f"the turn ended late: {result}"


@pytest.mark.asyncio
async def test_a_quiet_number_read_slowly_is_not_cut_between_digits():
    # Digits a second apart, the way a caller reads them off a screen: a gap
    # inside the reading must not end the turn on the first digits.
    words = ["four", "one", "seven", "two", "nine", "three", "eight", "five"]
    frames = [*_paced(_silence(0.4))]
    for i in range(len(words)):
        frames += _paced(_quiet_speech(i, 0.3)) + _paced(_silence(0.7))
    frames += _paced(_silence(6.0))
    result = await _call(frames, words)
    assert result["turns"], f"the turn never ended: {result}"
    assert len(result["turns"]) == 1, f"the number was split: {result}"
    assert all(w in _heard(result) for w in words), f"words were lost: {result}"


@pytest.mark.asyncio
async def test_a_caller_the_vad_hears_is_committed_once():
    # The ordinary path is unchanged: one commit for one sentence.
    words = ["Hi,", "Vivid", "Seats", "support."]
    frames = [*_paced(_silence(0.4))]
    for i in range(len(words)):
        frames += _paced(_speech(i, 0.3))
    frames += _paced(_silence(6.0))
    result = await _call(frames, words)
    assert len(result["turns"]) == 1, f"{result}"
    assert [c for c in result["commits"] if c] == [" ".join(words)], f"{result}"


@pytest.mark.asyncio
async def test_a_pause_after_a_dash_keeps_the_sentence_in_one_turn():
    # Copy run 6986: "Sorry, the order ID I said- / Is wrong" became two turns
    # under a flat one-second wait, and each was answered.
    words = ["Sorry,", "the", "order", "I", "said-", "is", "wrong."]
    frames = [*_paced(_silence(0.4))]
    for i in range(5):
        frames += _paced(_quiet_speech(i, 0.3))
    frames += _paced(_silence(1.4))
    for i in range(5, 7):
        frames += _paced(_quiet_speech(i, 0.3))
    frames += _paced(_silence(6.0))
    result = await _call(frames, words)
    assert len(result["turns"]) == 1, f"the pause split the sentence: {result}"
    assert all(w in _heard(result) for w in words), f"words were lost: {result}"


@pytest.mark.asyncio
async def test_a_finished_short_answer_is_committed_quickly():
    # Copy run 6986: "Yeah." waited the full second like any other word.
    words = ["Yeah."]
    frames = [*_paced(_silence(0.4)), *_paced(_quiet_speech(0, 0.4)), *_paced(_silence(6.0))]
    result = await _call(frames, words)
    assert result["turns"], f"the turn never ended: {result}"
    assert "Yeah." in _heard(result)
    assert result["turns"][-1][0] <= 1.5, f"the turn ended late: {result}"


def test_the_wait_reads_how_the_words_end():
    from api.services.pipecat.elevenlabs_manual_stt import words_quiet_seconds as wait

    assert wait("Yeah.") < 1.0
    assert wait("Is this Domino's Pizza?") < 1.0
    assert wait("Sorry, the order ID I said-") >= 2.0
    assert wait("Uh, I'm calling about an") >= 2.0
    assert wait("Hello, um") >= 2.0
    assert wait("Five, five, one, two") >= 2.0
    assert wait("Order 64417208.") < 1.0
