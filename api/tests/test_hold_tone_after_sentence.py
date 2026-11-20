"""The transfer's hold tone waits for the transfer sentence.

Copy runs 6990 and 6992 (2026-11-20): the tone started the moment the dial
went out, so the caller heard "I'm putting you through to the team now —
please hold." with the tone already playing over it.
"""

import asyncio

import pytest

from api.services.workflow.pipecat_engine_custom_tools import wait_for_sentence_then


class FakeSpeech:
    def __init__(self):
        self._played = asyncio.get_event_loop().create_future()

    @property
    def done(self) -> bool:
        return self._played.done()

    async def wait(self) -> bool:
        return await asyncio.shield(self._played)

    def play_out(self):
        self._played.set_result(True)


@pytest.mark.asyncio
async def test_the_tone_starts_only_once_the_sentence_has_played():
    speech, stop = FakeSpeech(), asyncio.Event()
    waiter = asyncio.create_task(wait_for_sentence_then(speech, stop, 5.0))
    await asyncio.sleep(0.2)
    assert not waiter.done(), "the tone would have started over the sentence"
    speech.play_out()
    await asyncio.wait_for(waiter, 1.0)


@pytest.mark.asyncio
async def test_a_transfer_that_ends_first_stops_the_wait():
    speech, stop = FakeSpeech(), asyncio.Event()
    waiter = asyncio.create_task(wait_for_sentence_then(speech, stop, 5.0))
    await asyncio.sleep(0.1)
    stop.set()
    await asyncio.wait_for(waiter, 1.0)


@pytest.mark.asyncio
async def test_a_sentence_that_never_ends_holds_the_tone_only_to_the_limit():
    speech, stop = FakeSpeech(), asyncio.Event()
    await asyncio.wait_for(wait_for_sentence_then(speech, stop, 0.3), 1.0)


@pytest.mark.asyncio
async def test_no_sentence_means_no_wait():
    await asyncio.wait_for(wait_for_sentence_then(None, asyncio.Event(), 5.0), 0.1)
