"""AssemblyAI speech recognition that does not end the call it was asked to leave.

Pipecat's ``AssemblyAISTTService`` answers the server's ``Termination``
message by pushing a bare ``EndFrame()`` downstream. That message is the
acknowledgment of the ``Terminate`` the service itself sends from
``_disconnect`` -- on every ``stop``, ``cancel`` and settings reconnect -- so
every call that ends sends a second, reason-less ``EndFrame`` down the
pipeline from the middle of it, ahead of the engine's own.

For an ordinary hangup that is harmless: both frames hang up. For a transfer
it hangs up the caller. The engine queues ``EndFrame(reason="transfer_call")``;
the STT, stopping on it, waits for ``Termination`` before passing it on, and
the reason-less frame it pushes from the receive task overtakes it. The Telnyx
serializer reads that one first, finds no transfer reason and hangs the
caller's leg up; the transfer frame arrives at a socket already closing and is
dropped, so the caller is never joined to the conference and the team's phone
is left alone on the line. Both frames reaching the end of the pipeline also
ran ``on_pipeline_finished`` twice -- every artifact uploaded, and the run's
completion processed, twice. Production runs 4791 (``-harkbell.18``), 4803 and
4805 (``-harkbell.20``), 2026-11-13: every transfer since the fleet moved to
AssemblyAI on 2026-11-12 (D-084); the last bridged transfers ran on nova-3.

A termination this service asked for is therefore only acknowledged. One the
server sends unprompted still ends the pipeline as upstream does.
"""

from loguru import logger
from pipecat.services.assemblyai.models import TerminationMessage
from pipecat.services.assemblyai.stt import AssemblyAISTTService


class HarkbellAssemblyAISTTService(AssemblyAISTTService):
    """``AssemblyAISTTService`` whose own ``Terminate`` pushes no ``EndFrame``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._terminating = False

    async def _disconnect(self):
        # Every Terminate this service sends goes out from here, and the
        # Termination that answers it arrives while this waits for it.
        self._terminating = True
        try:
            await super()._disconnect()
        finally:
            self._terminating = False

    async def _handle_termination(self, message: TerminationMessage):
        if not self._terminating:
            await super()._handle_termination(message)
            return
        self._received_termination = True
        self._termination_event.set()
        logger.info(
            f"Session Terminated: Audio Duration={message.audio_duration_seconds}s, "
            f"Session Duration={message.session_duration_seconds}s"
        )
