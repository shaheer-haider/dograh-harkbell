"""Harkbell: the session tuning an OpenAI realtime configuration may carry.

Unset fields send nothing, so a configuration from before ``-harkbell.24``
opens the session it always opened.
"""

from types import SimpleNamespace

from api.services.configuration.registry import OpenAIRealtimeLLMConfiguration
from api.services.pipecat.service_factory import create_realtime_llm_service


def _session(**settings) -> dict:
    config = OpenAIRealtimeLLMConfiguration(
        model="gpt-realtime-2.1", voice="marin", api_key="test", **settings
    )
    service = create_realtime_llm_service(SimpleNamespace(realtime=config), None)
    return service._settings.session_properties.model_dump(exclude_none=True)


def test_an_untuned_configuration_sends_no_tuning():
    session = _session()
    assert "reasoning" not in session
    assert "turn_detection" not in session["audio"]["input"]
    assert "noise_reduction" not in session["audio"]["input"]
    assert session["audio"]["output"] == {"voice": "marin"}


def test_semantic_turns_noise_reduction_reasoning_and_speed():
    session = _session(
        language="en",
        reasoning_effort="low",
        turn_detection="semantic_vad",
        vad_eagerness="medium",
        noise_reduction="near_field",
        speed=1.05,
    )
    assert session["reasoning"] == {"effort": "low"}
    assert session["audio"]["input"]["turn_detection"] == {
        "type": "semantic_vad",
        "eagerness": "medium",
    }
    assert session["audio"]["input"]["noise_reduction"] == {"type": "near_field"}
    assert session["audio"]["input"]["transcription"]["language"] == "en"
    assert session["audio"]["output"] == {"voice": "marin", "speed": 1.05}


def test_a_silence_length_alone_means_server_vad():
    turn = _session(vad_silence_ms=700)["audio"]["input"]["turn_detection"]
    assert turn["type"] == "server_vad"
    assert turn["silence_duration_ms"] == 700
    assert turn["threshold"] == 0.5


def test_an_untuned_configuration_keeps_the_default_transcriber():
    transcription = _session()["audio"]["input"]["transcription"]
    assert transcription == {"model": "gpt-realtime-whisper"}


def _service(**settings):
    config = OpenAIRealtimeLLMConfiguration(
        model="gpt-realtime-2.1", voice="marin", api_key="test", **settings
    )
    return create_realtime_llm_service(SimpleNamespace(realtime=config), None)


def test_the_transcript_model_and_its_keywords_reach_the_session():
    service = _service(
        language="en",
        transcription_model="gpt-live-transcribe",
        transcription_keywords=["Mirror Labs", " StubHub ", "", "re-transfer"],
    )
    update = {
        "type": "session.update",
        "session": service._settings.session_properties.model_dump(exclude_none=True),
    }
    sent = service._with_transcription_keywords(update)
    assert sent["session"]["audio"]["input"]["transcription"] == {
        "model": "gpt-live-transcribe",
        "language": "en",
        "keywords": ["Mirror Labs", "StubHub", "re-transfer"],
    }
    # Nothing else on the wire is touched.
    audio = {"type": "input_audio_buffer.append", "audio": "AAAA"}
    assert service._with_transcription_keywords(audio) == {
        "type": "input_audio_buffer.append",
        "audio": "AAAA",
    }


def test_no_keywords_leaves_a_session_update_alone():
    service = _service(transcription_model="gpt-live-transcribe")
    update = {
        "type": "session.update",
        "session": service._settings.session_properties.model_dump(exclude_none=True),
    }
    sent = service._with_transcription_keywords(update)
    assert "keywords" not in sent["session"]["audio"]["input"]["transcription"]


def test_a_transcript_model_without_keywords_sends_none():
    transcription = _session(transcription_model="gpt-live-transcribe")["audio"][
        "input"
    ]["transcription"]
    assert transcription == {"model": "gpt-live-transcribe"}


def test_prefix_padding_rides_server_vad():
    turn = _session(vad_prefix_padding_ms=500, vad_silence_ms=700)["audio"]["input"][
        "turn_detection"
    ]
    assert turn == {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 500,
        "silence_duration_ms": 700,
    }


def test_keywords_stay_off_a_transcriber_that_refuses_them():
    service = _service(transcription_keywords=["Mirror Labs"])
    update = {
        "type": "session.update",
        "session": service._settings.session_properties.model_dump(exclude_none=True),
    }
    sent = service._with_transcription_keywords(update)
    assert sent["session"]["audio"]["input"]["transcription"] == {
        "model": "gpt-realtime-whisper"
    }
