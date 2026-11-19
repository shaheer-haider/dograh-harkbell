from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pipecat.services.elevenlabs.stt import CommitStrategy
from pipecat.transcriptions.language import Language

from api.services.configuration.options import (
    ELEVENLABS_STT_LANGUAGES,
    ELEVENLABS_STT_MODELS,
)
from api.services.configuration.registry import (
    ElevenlabsSTTConfiguration,
    ServiceProviders,
)
from api.services.pipecat.audio_config import AudioConfig
from api.services.pipecat.service_factory import (
    create_stt_service,
    create_tts_service,
    stt_uses_external_turns,
)


def _audio_config() -> AudioConfig:
    return AudioConfig(
        transport_in_sample_rate=16000,
        transport_out_sample_rate=16000,
    )


def _elevenlabs_config(
    language: str = "en",
    base_url: str = "https://api.elevenlabs.io",
) -> SimpleNamespace:
    return SimpleNamespace(
        stt=SimpleNamespace(
            provider=ServiceProviders.ELEVENLABS.value,
            api_key="test-key",
            model="scribe_v2_realtime",
            language=language,
            base_url=base_url,
        )
    )


def _elevenlabs_tts_config(base_url: str) -> SimpleNamespace:
    return SimpleNamespace(
        tts=SimpleNamespace(
            provider=ServiceProviders.ELEVENLABS.value,
            api_key="test-key",
            model="eleven_flash_v2_5",
            voice="test-voice",
            speed=1.0,
            base_url=base_url,
        )
    )


def test_elevenlabs_stt_configuration_exposes_defaults_and_languages():
    config = ElevenlabsSTTConfiguration(api_key="test-key")
    language_schema = ElevenlabsSTTConfiguration.model_json_schema()["properties"][
        "language"
    ]

    assert config.provider == ServiceProviders.ELEVENLABS
    assert config.model == "scribe_v2_realtime"
    assert config.language == "en"
    assert config.base_url == "https://api.elevenlabs.io"
    assert ELEVENLABS_STT_MODELS == ("scribe_v2_realtime",)
    assert "auto" in ELEVENLABS_STT_LANGUAGES
    assert "es" in ELEVENLABS_STT_LANGUAGES
    assert language_schema["examples"] == list(ELEVENLABS_STT_LANGUAGES)


def test_elevenlabs_stt_uses_realtime_service_with_language_mapping():
    user_config = _elevenlabs_config(language="es")

    assert not stt_uses_external_turns(user_config)

    with patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTService"
    ) as stt_service:
        create_stt_service(user_config, _audio_config())

    stt_service.assert_called_once()
    kwargs = stt_service.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["base_url"] == "api.elevenlabs.io"
    assert kwargs["commit_strategy"] == CommitStrategy.VAD
    assert kwargs["sample_rate"] == 16000
    assert kwargs["should_interrupt"] is False
    assert kwargs["settings"].model == "scribe_v2_realtime"
    assert kwargs["settings"].language == Language.ES


def test_elevenlabs_stt_auto_language_passes_none():
    user_config = _elevenlabs_config(language="auto")

    with patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTService"
    ) as stt_service:
        create_stt_service(user_config, _audio_config())

    kwargs = stt_service.call_args.kwargs
    assert kwargs["settings"].language is None


def test_elevenlabs_stt_extracts_hostname_from_residency_base_url():
    user_config = _elevenlabs_config(base_url="https://api.eu.residency.elevenlabs.io")

    with patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTService"
    ) as stt_service:
        create_stt_service(user_config, _audio_config())

    kwargs = stt_service.call_args.kwargs
    assert kwargs["base_url"] == "api.eu.residency.elevenlabs.io"


def test_elevenlabs_stt_custom_language_passes_through():
    user_config = _elevenlabs_config(language="custom-lang")

    with patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTService"
    ) as stt_service:
        create_stt_service(user_config, _audio_config())

    kwargs = stt_service.call_args.kwargs
    assert kwargs["settings"].language == "custom-lang"


def test_elevenlabs_stt_bare_hostname_base_url_is_preserved():
    user_config = _elevenlabs_config(base_url="api.elevenlabs.io")

    with patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTService"
    ) as stt_service:
        create_stt_service(user_config, _audio_config())

    kwargs = stt_service.call_args.kwargs
    assert kwargs["base_url"] == "api.elevenlabs.io"


def test_elevenlabs_stt_preserves_non_default_port_in_base_url():
    user_config = _elevenlabs_config(base_url="https://localhost:8443")

    with patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTService"
    ) as stt_service:
        create_stt_service(user_config, _audio_config())

    kwargs = stt_service.call_args.kwargs
    assert kwargs["base_url"] == "localhost:8443"


def test_elevenlabs_stt_preserves_proxy_path_prefix_in_base_url():
    user_config = _elevenlabs_config(base_url="https://proxy.example.com/elevenlabs")

    with patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTService"
    ) as stt_service:
        create_stt_service(user_config, _audio_config())

    kwargs = stt_service.call_args.kwargs
    assert kwargs["base_url"] == "proxy.example.com/elevenlabs"


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        (
            "https://api.eu.residency.elevenlabs.io/elevenlabs/",
            "wss://api.eu.residency.elevenlabs.io/elevenlabs",
        ),
        ("http://localhost:8000/", "ws://localhost:8000"),
    ],
)
def test_elevenlabs_tts_uses_normalized_websocket_url(base_url, expected_url):
    user_config = _elevenlabs_tts_config(base_url)

    with patch(
        "api.services.pipecat.service_factory.ElevenLabsTTSService"
    ) as tts_service:
        create_tts_service(user_config, _audio_config())

    assert tts_service.call_args.kwargs["url"] == expected_url


def test_elevenlabs_stt_listed_custom_language_maps_to_pipecat_enum():
    user_config = _elevenlabs_config(language="yue")

    with patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTService"
    ) as stt_service:
        create_stt_service(user_config, _audio_config())

    kwargs = stt_service.call_args.kwargs
    assert kwargs["settings"].language == Language.YUE


def test_elevenlabs_stt_sends_no_pause_or_word_list_unless_configured():
    user_config = _elevenlabs_config()

    with patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTSettings"
    ) as settings, patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTService"
    ):
        create_stt_service(
            user_config, _audio_config(), keyterms=["SeatGeek", "Vivid Seats"]
        )

    assert set(settings.call_args.kwargs) == {"model", "language"}


def test_elevenlabs_stt_takes_the_commit_pause_and_the_dictionary_when_asked():
    # Mirror Labs copy, 2026-11-19: the provider's own pause made each reply
    # about 1.3 s slower, and with no word list "SeatGeek" was "Seed Geek".
    user_config = _elevenlabs_config()
    user_config.stt.vad_silence_threshold_secs = 0.8
    user_config.stt.keyterms_from_dictionary = True
    terms = [f"term {index}" for index in range(140)] + ["x" * 80]

    with patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTSettings"
    ) as settings, patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTService"
    ):
        create_stt_service(user_config, _audio_config(), keyterms=terms)

    kwargs = settings.call_args.kwargs
    assert kwargs["vad_silence_threshold_secs"] == 0.8
    assert len(kwargs["keyterms"]) == 100
    assert all(len(term) <= 50 for term in kwargs["keyterms"])


def test_elevenlabs_stt_configuration_bounds_the_commit_pause():
    assert ElevenlabsSTTConfiguration(api_key="k").vad_silence_threshold_secs is None
    assert (
        ElevenlabsSTTConfiguration(
            api_key="k", vad_silence_threshold_secs=0.8
        ).vad_silence_threshold_secs
        == 0.8
    )
    with pytest.raises(ValueError):
        ElevenlabsSTTConfiguration(api_key="k", vad_silence_threshold_secs=0.1)


def test_elevenlabs_stt_commits_on_the_providers_timer_unless_told_otherwise():
    assert ElevenlabsSTTConfiguration(api_key="k").commit_strategy == "vad"
    assert (
        ElevenlabsSTTConfiguration(api_key="k", commit_strategy="manual").commit_strategy
        == "manual"
    )
    with pytest.raises(ValueError):
        ElevenlabsSTTConfiguration(api_key="k", commit_strategy="sometimes")


def test_elevenlabs_stt_on_manual_commits_uses_the_engines_vad_and_drops_the_pause():
    # Mirror Labs copy, 2026-11-20: the provider's 0.7 s timer was 1.0 s of
    # every reply; the engine's VAD hears the caller stop in 0.2 s.
    user_config = _elevenlabs_config()
    user_config.stt.commit_strategy = "manual"
    user_config.stt.vad_silence_threshold_secs = 0.7

    with patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTSettings"
    ) as settings, patch(
        "api.services.pipecat.service_factory.HarkbellElevenLabsRealtimeSTTService"
    ) as manual, patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTService"
    ) as timer:
        create_stt_service(user_config, _audio_config())

    manual.assert_called_once()
    timer.assert_not_called()
    assert "vad_silence_threshold_secs" not in settings.call_args.kwargs
    assert "commit_strategy" not in manual.call_args.kwargs


def test_the_manual_service_commits_manually():
    from api.services.pipecat.elevenlabs_manual_stt import (
        HarkbellElevenLabsRealtimeSTTService,
    )
    from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTSettings

    service = HarkbellElevenLabsRealtimeSTTService(
        api_key="k",
        settings=ElevenLabsRealtimeSTTSettings(model="scribe_v3_live"),
        sample_rate=16000,
    )
    assert service._commit_strategy == CommitStrategy.MANUAL


async def _deferring_service(monkeypatch):
    """The manual service with its commits recorded and its base frame handling stubbed."""
    import asyncio as _asyncio
    from unittest.mock import AsyncMock

    from pipecat.clocks.system_clock import SystemClock
    from pipecat.processors.frame_processor import FrameProcessorSetup
    from pipecat.services.elevenlabs.stt import (
        ElevenLabsRealtimeSTTService,
        ElevenLabsRealtimeSTTSettings,
    )
    from pipecat.utils.asyncio.task_manager import TaskManager

    from api.services.pipecat import elevenlabs_manual_stt
    from api.services.pipecat.elevenlabs_manual_stt import (
        HarkbellElevenLabsRealtimeSTTService,
    )

    passed: list = []

    async def base_process_frame(self, frame, direction):
        passed.append(type(frame).__name__)

    # The parent of ElevenLabs' own service: what still runs on a deferred stop.
    monkeypatch.setattr(ElevenLabsRealtimeSTTService.__mro__[1], "process_frame", base_process_frame)
    monkeypatch.setattr(elevenlabs_manual_stt, "VERDICT_WAIT_SECONDS", 0.15)
    monkeypatch.setattr(elevenlabs_manual_stt, "HOLD_LIMIT_SECONDS", 0.3)
    service = HarkbellElevenLabsRealtimeSTTService(
        api_key="k",
        settings=ElevenLabsRealtimeSTTSettings(model="scribe_v3_live"),
        sample_rate=16000,
    )
    service._connect = AsyncMock()
    task_manager = TaskManager()
    await service.setup(
        FrameProcessorSetup(clock=SystemClock(), task_manager=task_manager, pipeline_worker=None)
    )
    commits: list[str] = []

    async def send_commit():
        commits.append("commit")

    service._send_commit = send_commit
    service.defer_commits = True
    return service, commits, passed


@pytest.mark.asyncio
async def test_a_deferred_stop_waits_for_the_turn_analyzer(monkeypatch):
    import asyncio as _asyncio

    from pipecat.frames.frames import VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
    from pipecat.processors.frame_processor import FrameDirection

    from api.services.pipecat.transcript_commit import TranscriptCommitRequestFrame

    service, commits, passed = await _deferring_service(monkeypatch)
    down, up = FrameDirection.DOWNSTREAM, FrameDirection.UPSTREAM
    await service.process_frame(VADUserStartedSpeakingFrame(), down)
    await service.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2), down)
    # The stop still flows on; no commit until the analyzer says so.
    assert passed[-1] == "VADUserStoppedSpeakingFrame"
    assert commits == []
    await service.process_frame(TranscriptCommitRequestFrame(), up)
    assert commits == ["commit"]
    # Asked again with nothing new heard: nothing to commit.
    await service.process_frame(TranscriptCommitRequestFrame(), up)
    assert commits == ["commit"]
    await _asyncio.sleep(0.2)
    assert commits == ["commit"]
    await service.cleanup()


@pytest.mark.asyncio
async def test_a_held_pause_is_not_committed_and_the_caller_speaking_again_keeps_it_open(monkeypatch):
    import asyncio as _asyncio

    from pipecat.frames.frames import VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
    from pipecat.processors.frame_processor import FrameDirection

    from api.services.pipecat.transcript_commit import (
        TranscriptCommitHoldFrame,
        TranscriptCommitRequestFrame,
    )

    service, commits, passed = await _deferring_service(monkeypatch)
    down, up = FrameDirection.DOWNSTREAM, FrameDirection.UPSTREAM
    await service.process_frame(VADUserStartedSpeakingFrame(), down)
    await service.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2), down)
    await service.process_frame(TranscriptCommitHoldFrame(), up)
    await _asyncio.sleep(0.2)
    assert commits == []
    await service.process_frame(VADUserStartedSpeakingFrame(), down)
    await _asyncio.sleep(0.2)
    assert commits == []
    await service.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2), down)
    await service.process_frame(TranscriptCommitRequestFrame(), up)
    assert commits == ["commit"]
    await service.cleanup()


@pytest.mark.asyncio
async def test_a_stop_with_no_verdict_and_a_hold_never_released_are_committed_anyway(monkeypatch):
    import asyncio as _asyncio

    from pipecat.frames.frames import VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
    from pipecat.processors.frame_processor import FrameDirection

    from api.services.pipecat.transcript_commit import TranscriptCommitHoldFrame

    service, commits, passed = await _deferring_service(monkeypatch)
    down, up = FrameDirection.DOWNSTREAM, FrameDirection.UPSTREAM
    await service.process_frame(VADUserStartedSpeakingFrame(), down)
    await service.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2), down)
    await _asyncio.sleep(0.25)
    assert commits == ["commit"]
    await service.process_frame(VADUserStartedSpeakingFrame(), down)
    await service.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2), down)
    await service.process_frame(TranscriptCommitHoldFrame(), up)
    await _asyncio.sleep(0.2)
    assert commits == ["commit"]
    await _asyncio.sleep(0.2)
    assert commits == ["commit", "commit"]
    await service.cleanup()
