"""The ElevenLabs manual-commit line through a gap in the caller's audio.

Copy run 6660 (2026-11-19): the engine's event loop stopped for 1.2 s while an
HTTP request built an S3 client, so no audio was processed for that long. The
VAD controller forced a speech stop, the audio held back during the stall then
arrived at once, and "Hi, Vivid Seats support." was never answered: the caller
sat in silence for 30 s until they said something else.

These tests run the real pipeline pieces the engine runs — the input
transport, the Harkbell ElevenLabs service with its deferred commits, the user
aggregator with its VAD controller, the min-words turn start and the
number-hold stop strategy — against a fake ElevenLabs socket, and check that
every word the caller said reaches the end of their turn.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from loguru import logger
from websockets.protocol import State

from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn, SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    StartFrame,
    SystemFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTSettings
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from api.services.pipecat.elevenlabs_manual_stt import HarkbellElevenLabsRealtimeSTTService
from api.services.pipecat.number_hold_turn_stop import (
    NumberHoldTurnAnalyzerUserTurnStopStrategy,
)

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
WORD_BASE = 20000


class EnergyVADAnalyzer(VADAnalyzer):
    """The real VAD state machine with energy for confidence.

    Analysed on the executor thread, as Silero is.
    """

    def num_frames_required(self) -> int:
        return FRAME_SAMPLES

    def voice_confidence(self, buffer: bytes) -> float:
        samples = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(samples**2))) / 32768.0
        return 1.0 if rms > 0.05 else 0.0


class InlineEnergyVADAnalyzer(EnergyVADAnalyzer):
    """Analysed inline: frames released by a stall reach VAD before the turn
    analyzer sees them, the ordering that left run 6660's turn open."""

    async def analyze_audio(self, buffer: bytes) -> VADState:
        return self._run_analyzer(buffer)


class ScriptedSmartTurn(BaseSmartTurn):
    """The turn model's verdicts in order; the last one repeats."""

    def __init__(self, verdicts: list[bool], **kwargs):
        super().__init__(**kwargs)
        self._verdicts = list(verdicts)

    def _predict_endpoint(self, audio_array: np.ndarray) -> dict[str, Any]:
        complete = self._verdicts.pop(0) if len(self._verdicts) > 1 else self._verdicts[0]
        return {"prediction": 1 if complete else 0, "probability": 1.0 if complete else 0.0}


class ReadyInputTransport(BaseInputTransport):
    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)


@dataclass
class _StallFrame(SystemFrame):
    """Marker: block the whole event loop, as building the S3 client did."""

    seconds: float = 1.2


class EventLoopStaller(FrameProcessor):
    """Blocks the event loop on a stall marker; every other frame passes."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, _StallFrame):
            logger.warning("test: blocking the event loop")
            time.sleep(frame.seconds)
            return
        await self.push_frame(frame, direction)


class FakeElevenLabsSocket:
    """ElevenLabs' realtime socket: partials per new word, a final per commit.

    A speech frame carries the index of the word it belongs to in its first
    sample, so the fake can say which words the audio since the last commit
    held. Replies arrive after a network delay, as the real ones do.
    """

    def __init__(self, service: HarkbellElevenLabsRealtimeSTTService, words: list[str]):
        self.state = State.OPEN
        self._service = service
        self._words = words
        self._pending: list[int] = []
        self.commits: list[str] = []

    async def send(self, message: str):
        data = json.loads(message)
        if data.get("commit"):
            text = " ".join(self._words[i] for i in self._pending)
            self._pending = []
            self.commits.append(text)
            self._reply(
                {"message_type": "committed_transcript", "text": text}
                if text
                else {"message_type": "insufficient_audio_activity"},
                0.3,
            )
            return
        audio = data.get("audio_base_64") or ""
        if not audio:
            return
        import base64

        samples = np.frombuffer(base64.b64decode(audio), dtype=np.int16)
        if samples.size and samples[0] >= WORD_BASE:
            word = int(samples[0]) - WORD_BASE
            if not self._pending or self._pending[-1] != word:
                self._pending.append(word)
                text = " ".join(self._words[i] for i in self._pending)
                self._reply({"message_type": "partial_transcript", "text": text}, 0.15)

    def _reply(self, data: dict, delay: float):
        async def later():
            await asyncio.sleep(delay)
            await self._service._process_response(data)

        self._service.create_task(later(), "fake_elevenlabs_reply")

    async def close(self):
        self.state = State.CLOSED


def _speech(word: int, seconds: float) -> list[InputAudioRawFrame]:
    rng = np.random.default_rng(word)
    frames = []
    for _ in range(int(seconds * 1000 / FRAME_MS)):
        samples = rng.integers(-9000, 9000, FRAME_SAMPLES, dtype=np.int16)
        samples[0] = WORD_BASE + word
        frames.append(
            InputAudioRawFrame(audio=samples.tobytes(), sample_rate=SAMPLE_RATE, num_channels=1)
        )
    return frames


def _silence(seconds: float) -> list[InputAudioRawFrame]:
    return [
        InputAudioRawFrame(audio=bytes(FRAME_SAMPLES * 2), sample_rate=SAMPLE_RATE, num_channels=1)
        for _ in range(int(seconds * 1000 / FRAME_MS))
    ]


def _paced(frames: list[InputAudioRawFrame]) -> list:
    """Frames at the speed a caller speaks them: 20 ms apart."""
    out: list = []
    for frame in frames:
        out += [frame, SleepFrame(FRAME_MS / 1000)]
    return out


async def _call(
    frames_to_send: list,
    words: list[str],
    verdicts: list[bool] | None = None,
    inline_vad: bool = False,
) -> dict:
    stt = HarkbellElevenLabsRealtimeSTTService(
        api_key="k",
        settings=ElevenLabsRealtimeSTTSettings(model="scribe_v3_live"),
        sample_rate=SAMPLE_RATE,
    )
    socket = FakeElevenLabsSocket(stt, words)

    async def connect():
        stt._websocket = socket
        stt._connected_event.set()

    async def disconnect():
        await socket.close()

    stt._connect = connect
    stt._disconnect = disconnect
    stt.defer_commits = True

    aggregator = LLMUserAggregator(
        LLMContext(),
        params=LLMUserAggregatorParams(
            vad_analyzer=(InlineEnergyVADAnalyzer if inline_vad else EnergyVADAnalyzer)(
                params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.2, min_volume=0.0)
            ),
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=3)],
                stop=[
                    NumberHoldTurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=ScriptedSmartTurn(
                            verdicts or [True], params=SmartTurnParams(stop_secs=2.0)
                        )
                    )
                ],
            ),
            user_turn_stop_timeout=5.0,
        ),
    )
    # (seconds since the last word was spoken, who ended the turn, its text)
    turns: list[tuple[float, str | None, str]] = []
    last_word: list[float] = []

    class LastWordClock(FrameProcessor):
        """Notes when the last speech frame enters the pipeline."""

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame) and frame.audio and frame.audio[1] != 0:
                last_word[:] = [time.monotonic()]
            await self.push_frame(frame, direction)

    @aggregator.event_handler("on_user_turn_stopped")
    async def _on_stopped(agg, strategy, message):
        since = time.monotonic() - last_word[0] if last_word else 0.0
        turns.append((round(since, 2), type(strategy).__name__ if strategy else None, message.content))

    transport = ReadyInputTransport(TransportParams(audio_in_enabled=True))

    await run_test(
        Pipeline([transport, EventLoopStaller(), LastWordClock(), stt, aggregator]),
        frames_to_send=frames_to_send,
        expected_down_frames=None,
    )
    return {"turns": turns, "commits": socket.commits}


WORDS = ["Hi,", "Vivid", "Seats", "support."]
STRATEGY = "NumberHoldTurnAnalyzerUserTurnStopStrategy"


def _assert_answered(result: dict, *, within: float):
    turns = result["turns"]
    assert turns, f"the turn never ended: {result}"
    heard = " ".join(text for _, _, text in turns)
    assert all(word in heard for word in WORDS), f"words were lost: {result}"
    # Ended by the strategy, not by the aggregator's safety timeout, and
    # soon after the caller's last word.
    assert all(by == STRATEGY for _, by, _ in turns), f"a turn ended on the timeout: {result}"
    assert turns[-1][0] <= within, f"the turn ended late: {result}"


@pytest.mark.asyncio
@pytest.mark.parametrize("inline_vad", [False, True], ids=["threaded-vad", "inline-vad"])
async def test_a_gap_with_no_audio_mid_sentence_keeps_the_sentence_in_one_turn(inline_vad):
    # The caller says two words, the audio stops for 1.3 s (no frames at all:
    # the VAD controller forces a stop after 1 s), then the rest arrives.
    frames = [
        *_paced(_silence(0.4)),
        *_paced(_speech(0, 0.3)),
        *_paced(_speech(1, 0.4)),
        SleepFrame(1.3),
        *_paced(_speech(2, 0.3)),
        *_paced(_speech(3, 0.4)),
        *_paced(_silence(6.0)),
    ]
    result = await _call(frames, WORDS, inline_vad=inline_vad)
    _assert_answered(result, within=1.5)
    assert len(result["turns"]) == 1, f"the gap split the sentence: {result}"


@pytest.mark.asyncio
@pytest.mark.parametrize("inline_vad", [False, True], ids=["threaded-vad", "inline-vad"])
@pytest.mark.parametrize("verdicts", [[True], [True, False]], ids=["finished", "unfinished-after"])
async def test_a_stalled_event_loop_mid_sentence_still_answers_every_word(verdicts, inline_vad):
    # Run 6660: the loop stops for 1.2 s while the caller is speaking, and the
    # audio that arrived meanwhile is then delivered at once.
    frames = [
        *_paced(_silence(0.4)),
        *_paced(_speech(0, 0.3)),
        *_paced(_speech(1, 0.2)),
        _StallFrame(seconds=1.2),
        *_speech(1, 0.2),
        *_speech(2, 0.3),
        *_speech(3, 0.4),
        *_silence(0.3),
        *_paced(_silence(6.0)),
    ]
    result = await _call(frames, WORDS, verdicts, inline_vad=inline_vad)
    # The analyzer's 2 s silence window and the backstop's margin, with room.
    _assert_answered(result, within=3.0)
