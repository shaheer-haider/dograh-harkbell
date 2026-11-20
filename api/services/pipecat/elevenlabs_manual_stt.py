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

Since ``-harkbell.38`` the commit at a VAD stop also waits for the turn
analyzer's word (``transcript_commit.py``), when ``defer_commits`` is on: a
pause the analyzer judges unfinished is not committed, so the transcriber
hears the sentence whole. A stop the analyzer does not answer within
``VERDICT_WAIT_SECONDS`` is committed anyway, and a hold that is never
released is committed after ``HOLD_LIMIT_SECONDS`` of silence.

Since ``-harkbell.41`` the transcriber's own words can end a turn too. On copy
runs 6978 and 6980 (a mobile line from Pakistan) Scribe heard every word and
those words stopped the greeting, but Silero scored the voice 0.4–0.66 against
its 0.7 bar and never reported the caller speaking, so no commit was ever sent:
"Hello? Hello?" waited 19 s, and a sentence said after a short VAD blip was
never answered. A partial transcript with new words while the VAD is not
reporting speech now arms a timer, and once the words have stopped changing for
``WORDS_QUIET_SECONDS`` (``WORDS_NUMBER_QUIET_SECONDS`` when they end on a
part-read number) the turn is committed. A caller the VAD hears never reaches
it: partials during VAD speech arm nothing, and every commit cancels it.
"""

import asyncio
import json

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    Frame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.elevenlabs.stt import CommitStrategy, ElevenLabsRealtimeSTTService

from api.services.pipecat.number_hold_turn_stop import unfinished_number
from api.services.pipecat.transcript_commit import (
    TranscriptCommitHoldFrame,
    TranscriptCommitRequestFrame,
)

# How long to wait before sending a throttled commit again, and how many times.
COMMIT_RETRY_SECONDS = 0.25
COMMIT_RETRY_LIMIT = 3
# How long a VAD stop waits for the turn analyzer's word before it is committed
# anyway. The analyzer answers in tens of milliseconds; this is for a stop it
# never sees (a muted turn, a frame dropped on the way).
VERDICT_WAIT_SECONDS = 0.5
# The longest silence a held commit waits through. The analyzer ends an
# unfinished turn after its own 2 s and a part-read number after 1.5 s, so a
# hold still standing after this was never going to be released.
HOLD_LIMIT_SECONDS = 4.0
# How long the transcriber's words must stop changing, with the VAD silent,
# before they are committed. Scribe sends a partial for every new word while a
# caller speaks, so a second without one is a pause, not a word in progress.
WORDS_QUIET_SECONDS = 1.0
# The same when the words end on a run of digits too short to be a whole order
# number: a caller reading one off a screen pauses between digits.
WORDS_NUMBER_QUIET_SECONDS = 2.0


class HarkbellElevenLabsRealtimeSTTService(ElevenLabsRealtimeSTTService):
    """ElevenLabs realtime STT on manual commits, retrying a throttled commit."""

    def __init__(self, **kwargs):
        kwargs["commit_strategy"] = CommitStrategy.MANUAL
        super().__init__(**kwargs)
        self._commit_retry_task: asyncio.Task | None = None
        self._commit_retries = 0
        # Set by the pipeline when its stop strategy sends commit verdicts.
        self.defer_commits = False
        self._commit_timer: asyncio.Task | None = None
        # Speech has been heard since the last commit.
        self._uncommitted = False
        # The VAD is reporting the caller speaking.
        self._vad_speaking = False
        # The newest partial transcript, and the timer that commits it when the
        # VAD never reports the speech behind it.
        self._last_partial = ""
        self._words_timer: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TranscriptCommitRequestFrame):
            await self._cancel_commit_timer()
            await self._commit_heard()
            return
        if isinstance(frame, TranscriptCommitHoldFrame):
            if self._uncommitted:
                await self._arm_commit_timer(HOLD_LIMIT_SECONDS)
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            # The caller is speaking: whatever waited for a verdict is part of
            # what they are saying now.
            self._uncommitted = True
            self._vad_speaking = True
            await self._cancel_commit_timer()
            await self._cancel_words_timer()
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_speaking = False
        if self.defer_commits and isinstance(frame, VADUserStoppedSpeakingFrame):
            # The ElevenLabs service would commit here. Its parent's handling
            # stays; the commit waits for the turn analyzer.
            await super(ElevenLabsRealtimeSTTService, self).process_frame(frame, direction)
            self._uncommitted = True
            await self._arm_commit_timer(VERDICT_WAIT_SECONDS)
            return
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._uncommitted = False

    async def _commit_heard(self) -> None:
        await self._cancel_words_timer()
        if not self._uncommitted:
            return
        self._uncommitted = False
        await self._send_commit()

    async def _on_words(self, text: str) -> None:
        """New words from the transcriber: commit them if the VAD never does."""
        if not text or text == self._last_partial:
            return
        self._last_partial = text
        if self._vad_speaking:
            return
        self._uncommitted = True
        await self._cancel_words_timer()
        quiet = WORDS_NUMBER_QUIET_SECONDS if unfinished_number(text) else WORDS_QUIET_SECONDS
        self._words_timer = self.create_task(
            self._commit_when_words_stop(quiet), f"{self}::commit_when_words_stop"
        )

    async def _commit_when_words_stop(self, seconds: float) -> None:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        self._words_timer = None
        if self._vad_speaking:
            return
        logger.info(
            f"{self}: committing words the voice detector did not hear, "
            f"{seconds}s after the last one: [{self._last_partial}]"
        )
        await self._commit_heard()

    async def _cancel_words_timer(self) -> None:
        if self._words_timer:
            await self.cancel_task(self._words_timer)
            self._words_timer = None

    async def _arm_commit_timer(self, seconds: float) -> None:
        await self._cancel_commit_timer()
        self._commit_timer = self.create_task(
            self._commit_after(seconds), f"{self}::commit_after"
        )

    async def _commit_after(self, seconds: float) -> None:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        self._commit_timer = None
        logger.debug(f"{self}: committing after {seconds}s with no word from the turn analyzer")
        await self._commit_heard()

    async def _cancel_commit_timer(self) -> None:
        if self._commit_timer:
            await self.cancel_task(self._commit_timer)
            self._commit_timer = None

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
            self._last_partial = ""
            await self._cancel_commit_retry()
            await self._cancel_words_timer()
        await super()._process_response(data)
        if message_type == "partial_transcript":
            await self._on_words(str(data.get("text") or "").strip())

    async def _disconnect(self):
        await self._cancel_commit_retry()
        await self._cancel_commit_timer()
        await self._cancel_words_timer()
        await super()._disconnect()
