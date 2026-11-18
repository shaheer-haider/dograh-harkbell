"""Harkbell: everything the provider sends on a gpt-live-transcribe session parses.

The fixture is a real session (2026-11-18): gpt-realtime-2.1 with
``gpt-live-transcribe``, a language and keywords, one spoken caller turn and
one reply. On ``-harkbell.28`` its ``session.updated`` could not be parsed -
the transcription block echoes ``languages`` and ``keywords`` - and that
killed the receive loop of every realtime call.
"""

import json
from pathlib import Path

# Importing the service is what makes the transcription block tolerant.
import api.services.pipecat.realtime.openai_realtime  # noqa: F401
from pipecat.services.openai.realtime import events

FIXTURE = (
    Path(__file__).parent / "fixtures" / "openai_realtime_live_transcribe_session.json"
)


def _events() -> list[dict]:
    return json.loads(FIXTURE.read_text())


def test_the_fixture_is_the_session_that_broke():
    updated = next(e for e in _events() if e["type"] == "session.updated")
    transcription = updated["session"]["audio"]["input"]["transcription"]
    assert transcription["model"] == "gpt-live-transcribe"
    assert transcription["languages"] == ["en"]
    assert "keywords" in transcription


def test_every_event_of_a_live_transcribe_session_parses():
    for event in _events():
        parsed = events.parse_server_event(json.dumps(event))
        assert parsed.type == event["type"]


def test_the_transcription_block_keeps_what_the_engine_reads():
    updated = next(e for e in _events() if e["type"] == "session.updated")
    parsed = events.parse_server_event(json.dumps(updated))
    transcription = parsed.session.audio.input.transcription
    assert transcription.model == "gpt-live-transcribe"


def test_the_callers_words_arrive():
    done = next(
        e
        for e in _events()
        if e["type"] == "conversation.item.input_audio_transcription.completed"
    )
    parsed = events.parse_server_event(json.dumps(done))
    assert "StubHub" in parsed.transcript
