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
