"""Harkbell: ElevenLabs realtime transcription committed by the engine's own voice detector.

With the provider's own commit timer (``CommitStrategy.VAD``) a transcript is
final only after ``vad_silence_threshold_secs`` of silence, and the turn
analyzer waits for that final before it lets the agent answer. On the Mirror
Labs copy that wait was 1.9 s at the provider's default pause and 1.0 s at
0.7 s, against 0.56 s on AssemblyAI (2026-11-19/20, runs 6503-6545).

``CommitStrategy.MANUAL`` sends the commit when the engine's Silero VAD
reports the caller stopped (``VADUserStoppedSpeakingFrame``, 0.2 s), and the
committed transcript comes back finalized, so the turn ends as soon as the
turn analyzer agrees. Pipecat's service already does that; this subclass adds
what an early commit needs around it:

- ``commit_throttled`` is the provider refusing a commit that came too soon
  after the last one. The base service reports it as an error and the audio
  is then never committed, so the turn waits for a final that never comes.
  Here the commit is sent again a moment later, a few times at most.
- ``insufficient_audio_activity`` is a commit with no speech in it (the VAD
  fired on a noise). There is nothing to transcribe, so it is logged, not
  raised as an error.
"""

import asyncio
import json

from loguru import logger
from websockets.protocol import State

from pipecat.services.elevenlabs.stt import CommitStrategy, ElevenLabsRealtimeSTTService

# How long to wait before sending a throttled commit again, and how many times.
COMMIT_RETRY_SECONDS = 0.25
COMMIT_RETRY_LIMIT = 3


class HarkbellElevenLabsRealtimeSTTService(ElevenLabsRealtimeSTTService):
    """ElevenLabs realtime STT on manual commits, retrying a throttled commit."""

    def __init__(self, **kwargs):
        kwargs["commit_strategy"] = CommitStrategy.MANUAL
        super().__init__(**kwargs)
        self._commit_retry_task: asyncio.Task | None = None
        self._commit_retries = 0

    async def _send_commit(self) -> None:
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        try:
            await self._websocket.send(
                json.dumps(
                    {
                        "message_type": "input_audio_chunk",
                        "audio_base_64": "",
                        "commit": True,
                        "sample_rate": self.sample_rate,
                    }
                )
            )
        except Exception as e:
            logger.warning(f"{self}: could not resend the commit: {e}")

    async def _retry_commit(self) -> None:
        try:
            await asyncio.sleep(COMMIT_RETRY_SECONDS)
        except asyncio.CancelledError:
            return
        self._commit_retry_task = None
        await self._send_commit()

    async def _cancel_commit_retry(self) -> None:
        if self._commit_retry_task:
            await self.cancel_task(self._commit_retry_task)
            self._commit_retry_task = None

    async def _process_response(self, data: dict):
        message_type = data.get("message_type")
        if message_type == "commit_throttled":
            if self._commit_retries < COMMIT_RETRY_LIMIT and self._commit_retry_task is None:
                self._commit_retries += 1
                logger.info(
                    f"{self}: commit throttled, sending it again "
                    f"({self._commit_retries}/{COMMIT_RETRY_LIMIT})"
                )
                self._commit_retry_task = self.create_task(
                    self._retry_commit(), f"{self}::retry_commit"
                )
            else:
                logger.warning(f"{self}: commit throttled, not retrying: {data}")
            return
        if message_type == "insufficient_audio_activity":
            logger.debug(f"{self}: a commit with no speech in it: {data}")
            return
        if message_type in ("committed_transcript", "committed_transcript_with_timestamps"):
            self._commit_retries = 0
            await self._cancel_commit_retry()
        await super()._process_response(data)

    async def _disconnect(self):
        await self._cancel_commit_retry()
        await super()._disconnect()
