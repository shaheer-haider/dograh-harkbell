"""A transferred Telnyx call joins the caller to the conference, never hangs them up.

Production, 2026-11-13, runs 4791 (``-harkbell.18``), 4803 and 4805
(``-harkbell.20``): the destination answered, the engine queued
``EndFrame(reason="transfer_call")``, and 0.5 s later the caller's leg was hung
up by ``TelnyxHangupStrategy`` -- no ``[Telnyx Transfer]`` line at all, so the
serializer never saw the transfer reason -- and every teardown ran twice. The
last line before the hangup was AssemblyAI's "Session Terminated": Pipecat's
``AssemblyAISTTService`` pushes a bare ``EndFrame()`` when its own Terminate is
acknowledged, and that frame overtakes the engine's (``assemblyai_stt.py``).

The call here is built the way ``run_pipeline.py`` builds a cascade call --
the websocket transport with the Telnyx serializer, the termination funnel,
the recognizer, the call monitor, the agent bridge and a child agent worker
-- and is ended the way ``transfer_call_handler`` ends it: from inside the
tool, after hold music, with ``end_call_with_reason(TRANSFER_CALL)``.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pipecat.frames.frames import FunctionCallResultProperties, LLMContextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.serializers.call_strategies import HangupStrategy, TransferStrategy
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.services.assemblyai.stt import AssemblyAISTTService, AssemblyAISTTSettings
from pipecat.tests import MockTTSService
from pipecat.tests.mock_llm_service import MockLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.utils.enums import EndTaskReason
from starlette.websockets import WebSocketState
from websockets.protocol import State

from api.services.pipecat.agent_bridge import AgentBridgeProcessor
from api.services.pipecat.assemblyai_stt import HarkbellAssemblyAISTTService
from api.services.pipecat.audio_playback import play_audio_loop
from api.services.pipecat.cut_off import harkbell_context_aggregators
from api.services.pipecat.pipeline_builder import create_agent_worker
from api.services.pipecat.termination_funnel_processor import (
    TerminationFunnelProcessor,
)
from api.services.pipecat.worker_runner import (
    create_worker_runner,
    run_worker_runner,
    wait_for_pipeline_worker_started,
)
from api.services.workflow.pipecat_engine import PipecatEngine


class _Strategies(TransferStrategy, HangupStrategy):
    """Records what the serializer asked of Telnyx."""

    def __init__(self):
        self.transfers = []
        self.hangups = []

    async def execute_transfer(self, context):
        self.transfers.append(dict(context))
        return True

    async def execute_hangup(self, context):
        self.hangups.append(dict(context))
        return True


class _TelnyxSocket:
    """The call's media socket, open until the engine closes it."""

    def __init__(self):
        self.headers = {}
        self.client_state = WebSocketState.CONNECTED
        self.application_state = WebSocketState.CONNECTED
        self._closed = asyncio.Event()

    async def receive(self):
        await self._closed.wait()
        return {"type": "websocket.disconnect"}

    async def send_text(self, data):
        pass

    async def send_bytes(self, data):
        pass

    async def close(self, *args, **kwargs):
        self.application_state = WebSocketState.DISCONNECTED
        self.client_state = WebSocketState.DISCONNECTED
        self._closed.set()


class _AssemblyAISocket:
    """AssemblyAI's streaming socket: a Terminate is answered with Termination."""

    def __init__(self):
        self.state = State.OPEN
        self._messages: asyncio.Queue = asyncio.Queue()

    async def send(self, data):
        if isinstance(data, str) and json.loads(data).get("type") == "Terminate":
            await self._messages.put(
                json.dumps(
                    {
                        "type": "Termination",
                        "audio_duration_seconds": 23,
                        "session_duration_seconds": 23,
                    }
                )
            )

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self._messages.get()
        if message is None:
            raise StopAsyncIteration
        return message

    async def close(self):
        self.state = State.CLOSED
        await self._messages.put(None)


def _recognizer(cls):
    async def connect(self, uri, **kwargs):
        return _AssemblyAISocket()

    stt = cls(
        api_key="key",
        settings=AssemblyAISTTSettings(model="universal-streaming-english"),
        sample_rate=8000,
    )
    return stt, patch.object(cls, "_websocket_connect", connect)


async def _transfer(recognizer_class) -> _Strategies:
    """One call, transferred once the destination answers; what Telnyx was asked."""
    strategies = _Strategies()
    transport = FastAPIWebsocketTransport(
        websocket=_TelnyxSocket(),
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            serializer=TelnyxFrameSerializer(
                stream_id="stream",
                call_control_id="v3:caller-leg",
                api_key="key",
                outbound_encoding="PCMU",
                inbound_encoding="PCMU",
                transfer_strategy=strategies,
                hangup_strategy=strategies,
            ),
            allowed_origins=[],
        ),
    )
    stt, connecting = _recognizer(recognizer_class)
    runner = create_worker_runner()
    llm = MockLLMService(
        mock_steps=[
            MockLLMService.create_function_call_chunks(
                "transfer_call", {}, tool_call_id="call_1"
            )
        ],
        chunk_delay=0.001,
    )
    tts = MockTTSService(mock_audio_duration_ms=40, frame_delay=0)
    context = LLMContext()
    engine = PipecatEngine(llm=llm, context=context, workflow=None, call_context_vars={})
    engine.active_agent.is_child = True

    async def transfer_call(params):
        # As transfer_call_handler: hold music while the destination rings,
        # then its answer ends the call with the transfer reason.
        stop = asyncio.Event()
        hold = asyncio.create_task(
            play_audio_loop(
                stop_event=stop,
                sample_rate=8000,
                queue_frame=transport.output().queue_frame,
            )
        )
        await asyncio.sleep(0.3)
        stop.set()
        await hold
        await params.result_callback(
            {"status": "transfer_success"},
            properties=FunctionCallResultProperties(run_llm=False),
        )
        await engine.end_call_with_reason(
            EndTaskReason.TRANSFER_CALL.value, abort_immediately=False
        )

    llm.register_function("transfer_call", transfer_call)

    user, assistant = harkbell_context_aggregators(
        context,
        user_params=LLMUserAggregatorParams(user_idle_timeout=0),
        assistant_params=LLMAssistantAggregatorParams(),
    )
    monitor = engine.call_monitor
    monitor.bind_user(user, idle_timeout=30)
    funnel = TerminationFunnelProcessor()

    async def dispose(reason, error=None):
        await engine.end_call_with_reason(reason, abort_immediately=True)

    funnel.set_termination_handler(dispose)
    bridge = AgentBridgeProcessor(
        bus=runner.bus,
        worker_name="call-1",
        selected_visit=lambda: engine.selected_visit_id,
        allow_inference=lambda: engine.agent_can_generate(engine.active_agent),
    )
    worker = PipelineWorker(
        Pipeline(
            [
                transport.input(),
                funnel,
                stt,
                user,
                monitor,
                bridge,
                transport.output(),
                assistant,
            ]
        ),
        name="call-1",
        params=PipelineParams(audio_out_sample_rate=8000),
        enable_rtvi=False,
        idle_timeout_secs=None,
    )
    finished = []

    @worker.event_handler("on_pipeline_finished")
    async def on_pipeline_finished(_worker, frame):
        finished.append(frame)

    child = create_agent_worker(
        Pipeline([llm, tts]),
        name=engine.active_agent.visit_id,
        call_worker_name=worker.name,
        audio_config=SimpleNamespace(
            transport_in_sample_rate=8000, transport_out_sample_rate=8000
        ),
    )
    engine.active_agent.worker = child
    engine.call_worker = worker
    engine.set_transport_output(transport.output())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _participant):
        # As event_handlers.py: a disconnect ends the call as a hangup.
        await engine.end_call_with_reason(
            EndTaskReason.USER_HANGUP.value, abort_immediately=True
        )

    with connecting:
        task = asyncio.create_task(run_worker_runner(runner, worker))
        await wait_for_pipeline_worker_started(worker, timeout=10, run_task=task)
        await worker.add_workers(child)
        await wait_for_pipeline_worker_started(child, timeout=3)
        assert await engine.activate_agent(engine.active_agent, timeout=1)
        monitor.activate()
        engine.expect_response()
        await worker.queue_frame(LLMContextFrame(context))
        await asyncio.wait_for(task, 15)

    strategies.finished = finished
    return strategies


@pytest.mark.asyncio
async def test_a_transfer_joins_the_caller_and_never_hangs_them_up():
    telnyx = await _transfer(HarkbellAssemblyAISTTService)
    assert telnyx.transfers == [{"call_control_id": "v3:caller-leg", "api_key": "key"}]
    assert telnyx.hangups == []
    # One end, so one teardown: artifacts and completion run once.
    assert len(telnyx.finished) == 1


@pytest.mark.asyncio
async def test_pipecats_own_recognizer_hangs_the_transferred_caller_up():
    # The production fault, kept so the subclass is never quietly dropped:
    # the recognizer's own EndFrame reaches the serializer first.
    telnyx = await _transfer(AssemblyAISTTService)
    assert telnyx.transfers == []
    assert len(telnyx.hangups) == 1
    assert len(telnyx.finished) == 2


# ── the recognizer on its own ─────────────────────────────────────────────


async def _terminated(*, asked: bool):
    stt, connecting = _recognizer(HarkbellAssemblyAISTTService)
    pushed = []

    async def push_frame(frame, direction=None):
        pushed.append(frame)

    stt.push_frame = push_frame
    with connecting:
        stt._websocket = _AssemblyAISocket()
        stt._connected = True
        if asked:
            receive = asyncio.create_task(stt._receive_messages())
            await stt._disconnect()
            receive.cancel()
        else:
            await stt._handle_message(
                {"type": "Termination", "audio_duration_seconds": 1, "session_duration_seconds": 1}
            )
    return stt, pushed


@pytest.mark.asyncio
async def test_its_own_terminate_is_acknowledged_without_ending_the_pipeline():
    from pipecat.frames.frames import EndFrame

    stt, pushed = await _terminated(asked=True)
    assert stt._received_termination
    assert not any(isinstance(f, EndFrame) for f in pushed)


@pytest.mark.asyncio
async def test_a_termination_the_server_sends_unasked_still_ends_the_pipeline():
    from pipecat.frames.frames import EndFrame

    _stt, pushed = await _terminated(asked=False)
    assert any(isinstance(f, EndFrame) for f in pushed)
