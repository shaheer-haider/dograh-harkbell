"""Harkbell T-679: a Telnyx media socket that never brings "start" is restarted once.

Production runs 4727 and 4735: an answered inbound call whose socket was
accepted but never carried a "start", so the caller heard nothing for the
whole call.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from api.enums import WorkflowRunMode, WorkflowRunState
from api.services.telephony.providers.telnyx import stream_recovery
from api.services.telephony.providers.telnyx.provider import TelnyxProvider

CONNECTED = json.dumps({"event": "connected"})
START = json.dumps(
    {
        "event": "start",
        "stream_id": "stream-1",
        "start": {"call_control_id": "cc-1", "media_format": {"encoding": "PCMU"}},
    }
)


class FakeSocket:
    """A media socket that delivers each message after its delay."""

    def __init__(self, *messages: tuple[float, str], second: bool = False):
        self.state = SimpleNamespace()
        stream_recovery.note_accepted(self)
        if second:
            stream_recovery.mark_second_socket(self)
        self._messages = list(messages)
        self.closed: tuple[int, str] | None = None

    async def receive_text(self) -> str:
        if not self._messages:
            await asyncio.Event().wait()
        delay, text = self._messages.pop(0)
        await asyncio.sleep(delay)
        return text

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)


@pytest.fixture
def call(monkeypatch):
    monkeypatch.setattr(stream_recovery, "START_TIMEOUT_SECS", 0.1)
    provider = TelnyxProvider({"api_key": "key", "connection_id": "conn"})
    provider.restart_streaming = AsyncMock(return_value=True)
    provider.hangup_call = AsyncMock(return_value=True)
    run = SimpleNamespace(
        initial_context={"direction": "inbound", "provider": "telnyx"},
        gathered_context={"call_id": "cc-1"},
    )
    pipeline = AsyncMock()
    allow = AsyncMock(return_value=True)
    failed = AsyncMock()
    module = "api.services.telephony.providers.telnyx.provider"
    with (
        patch(f"{module}.db_client.get_workflow_run", new=AsyncMock(return_value=run)),
        patch(
            f"{module}.get_backend_endpoints",
            new=AsyncMock(return_value=("https://engine", "wss://engine")),
        ),
        patch("api.services.pipecat.run_pipeline.run_pipeline_telephony", new=pipeline),
        patch.object(stream_recovery, "allow_second_socket", new=allow),
        patch("api.services.workflow_run_failure.mark_workflow_run_failed", new=failed),
    ):
        yield SimpleNamespace(
            provider=provider, run=run, pipeline=pipeline, allow=allow, failed=failed
        )


async def _handle(call, socket):
    await call.provider.handle_websocket(socket, 7, 3, 42)


@pytest.mark.asyncio
async def test_a_stream_that_starts_runs_the_pipeline_untouched(call):
    socket = FakeSocket((0, CONNECTED), (0.02, START))
    await _handle(call, socket)
    call.pipeline.assert_awaited_once()
    assert call.pipeline.await_args.kwargs["call_id"] == "cc-1"
    call.provider.restart_streaming.assert_not_awaited()
    call.allow.assert_not_awaited()
    assert socket.closed is None


@pytest.mark.asyncio
async def test_a_socket_with_no_start_is_restarted_once(call):
    socket = FakeSocket((0, CONNECTED))
    await _handle(call, socket)
    call.allow.assert_awaited_once_with(42)
    call.provider.restart_streaming.assert_awaited_once()
    call_control_id, stream_url = call.provider.restart_streaming.await_args.args
    assert call_control_id == "cc-1"
    assert stream_url.startswith("wss://engine") and "/7/3/42" in stream_url
    assert socket.closed == (1000, "Restarting the media stream")
    call.pipeline.assert_not_awaited()
    call.provider.hangup_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_socket_with_nothing_at_all_is_restarted_too(call):
    await _handle(call, FakeSocket())
    call.provider.restart_streaming.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_restarted_streams_socket_runs_the_pipeline(call):
    socket = FakeSocket((0, CONNECTED), (0.02, START), second=True)
    await _handle(call, socket)
    call.pipeline.assert_awaited_once()
    call.provider.restart_streaming.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_second_silent_socket_hangs_up_and_fails_the_run(call):
    socket = FakeSocket((0, CONNECTED), second=True)
    await _handle(call, socket)
    call.provider.restart_streaming.assert_not_awaited()
    call.provider.hangup_call.assert_awaited_once_with("cc-1")
    call.failed.assert_awaited_once()
    run_id, reason = call.failed.await_args.args
    assert run_id == 42
    assert "did not start it when asked again" in reason
    call.pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_stream_that_cannot_be_restarted_is_hung_up(call):
    call.provider.restart_streaming.return_value = False
    await _handle(call, FakeSocket((0, CONNECTED)))
    call.provider.hangup_call.assert_awaited_once_with("cc-1")
    assert "could not be restarted" in call.failed.await_args.args[1]


@pytest.mark.asyncio
async def test_a_late_start_on_the_first_socket_never_starts_a_pipeline(call):
    # The start lands after recovery began: it is never read.
    socket = FakeSocket((0, CONNECTED), (0.25, START))
    await _handle(call, socket)
    await asyncio.sleep(0.3)
    call.provider.restart_streaming.assert_awaited_once()
    call.pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_outbound_call_waits_for_its_answer_however_long(call):
    # Its socket opens while the far end rings; "start" comes on answer.
    call.run.initial_context = {"direction": "outbound", "provider": "telnyx"}
    socket = FakeSocket((0, CONNECTED), (0.4, START))
    await _handle(call, socket)
    call.pipeline.assert_awaited_once()
    call.provider.restart_streaming.assert_not_awaited()


# ── the one second socket ─────────────────────────────────────────────────


class FakeRedis:
    def __init__(self):
        self.values = {}

    async def setex(self, key, ttl, value):
        self.values[key] = value

    async def getdel(self, key):
        return self.values.pop(key, None)


@pytest.mark.asyncio
async def test_exactly_one_second_socket_is_admitted(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(stream_recovery, "_client", AsyncMock(return_value=redis))
    assert not await stream_recovery.take_second_socket(42)
    assert await stream_recovery.allow_second_socket(42)
    assert await stream_recovery.take_second_socket(42)
    assert not await stream_recovery.take_second_socket(42)


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [True, False])
async def test_a_running_telnyx_run_admits_its_second_socket_only_when_allowed(allowed):
    from api.routes.telephony import _handle_telephony_websocket

    run = SimpleNamespace(
        id=42,
        workflow_id=7,
        mode=WorkflowRunMode.TELNYX.value,
        state=WorkflowRunState.RUNNING.value,
        initial_context={"provider": "telnyx", "direction": "inbound"},
        gathered_context={"call_id": "cc-1"},
    )
    provider = SimpleNamespace(PROVIDER_NAME="telnyx", handle_websocket=AsyncMock())
    socket = FakeSocket()
    take = AsyncMock(return_value=allowed)
    with (
        patch("api.routes.telephony.db_client") as db,
        patch(
            "api.routes.telephony.get_telephony_provider_for_run",
            new=AsyncMock(return_value=provider),
        ),
        patch.object(stream_recovery, "take_second_socket", new=take),
    ):
        db.get_workflow_run = AsyncMock(return_value=run)
        db.get_workflow = AsyncMock(return_value=SimpleNamespace(id=7, organization_id=3))
        db.update_workflow_run = AsyncMock()
        await _handle_telephony_websocket(socket, 7, 3, 42)

    take.assert_awaited_once_with(42)
    if allowed:
        provider.handle_websocket.assert_awaited_once()
        assert stream_recovery.is_second_socket(socket)
        # Already running: the state is not written again.
        db.update_workflow_run.assert_not_awaited()
        assert socket.closed is None
    else:
        provider.handle_websocket.assert_not_awaited()
        assert socket.closed == (4409, "Workflow run not available for connection")


@pytest.mark.asyncio
async def test_other_providers_running_runs_are_refused_without_asking():
    from api.routes.telephony import _handle_telephony_websocket

    run = SimpleNamespace(
        id=42,
        workflow_id=7,
        mode=WorkflowRunMode.TWILIO.value,
        state=WorkflowRunState.RUNNING.value,
        initial_context={"provider": "twilio"},
        gathered_context={},
    )
    socket = FakeSocket()
    take = AsyncMock(return_value=True)
    with (
        patch("api.routes.telephony.db_client") as db,
        patch.object(stream_recovery, "take_second_socket", new=take),
    ):
        db.get_workflow_run = AsyncMock(return_value=run)
        db.get_workflow = AsyncMock(return_value=SimpleNamespace(id=7, organization_id=3))
        await _handle_telephony_websocket(socket, 7, 3, 42)
    take.assert_not_awaited()
    assert socket.closed == (4409, "Workflow run not available for connection")
