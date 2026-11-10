"""The caller is hung up when the transfer destination leaves the bridge.

Regression for a Harkbell production call (Mirror Labs copy, run 4426,
2026-11-10): the destination answered, the caller was joined into the
conference, and ``TelnyxConferenceStrategy`` then removed the transfer
context. When the destination hung up eight seconds later, the
transfer-result webhook found no context, took the hangup for a pre-answer
failure, published TRANSFER_FAILED to a pipeline that had already ended, and
never hung the caller up — who sat alone in the conference until they hung
up themselves. The context now outlives the bridge, so the post-answer
``call.hangup`` branch can hang the caller's leg up.
"""

import json
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request

from api.services.telephony.call_transfer_manager import CallTransferManager
from api.services.telephony.providers.telnyx.strategies import (
    BRIDGED_TRANSFER_CONTEXT_TTL_SECONDS,
    TelnyxConferenceStrategy,
)
from api.services.telephony.transfer_event_protocol import TransferContext


class _FakeRedis:
    def __init__(self) -> None:
        self._store: Dict[str, str] = {}
        self.ttls: List[int] = []

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.ttls.append(ttl)
        self._store[key] = value

    async def set(self, key: str, value: str, *, ex=None, nx: bool = False):
        if nx and key in self._store:
            return None
        self._store[key] = value
        return True

    async def get(self, key: str):
        return self._store.get(key)

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self._store.pop(key, None)

    async def publish(self, channel: str, message: str) -> None:
        pass


def _context(conference_id: str | None = "conf-1") -> TransferContext:
    return TransferContext(
        transfer_id="tx-1",
        call_sid="dest-leg",
        target_number="+12025550100",
        tool_uuid="tool-uuid",
        original_call_sid="caller-leg",
        conference_name="transfer-caller-leg",
        initiated_at=0.0,
        workflow_run_id=4426,
        conference_id=conference_id,
    )


def _manager_with(context: TransferContext):
    fake = _FakeRedis()
    manager = CallTransferManager(redis_client=fake)
    return fake, manager, context


@pytest.mark.asyncio
async def test_joining_the_caller_keeps_the_context_for_the_life_of_the_bridge():
    fake, manager, context = _manager_with(_context())
    await manager.store_transfer_context(context)

    strategy = TelnyxConferenceStrategy()
    with (
        patch(
            "api.services.telephony.call_transfer_manager.get_call_transfer_manager",
            new=AsyncMock(return_value=manager),
        ),
        patch.object(strategy, "_join_caller", new=AsyncMock(return_value=True)),
    ):
        joined = await strategy.execute_transfer(
            {"call_control_id": "caller-leg", "api_key": "key"}
        )

    assert joined is True
    kept = await manager.get_transfer_context("tx-1")
    assert kept is not None and kept.conference_id == "conf-1"
    assert await manager.find_transfer_context_for_call("caller-leg") is not None
    # Re-stored for hours, not the five minutes a dial is given.
    assert fake.ttls[-1] == BRIDGED_TRANSFER_CONTEXT_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_failed_join_still_removes_the_context():
    fake, manager, context = _manager_with(_context())
    await manager.store_transfer_context(context)

    strategy = TelnyxConferenceStrategy()
    with (
        patch(
            "api.services.telephony.call_transfer_manager.get_call_transfer_manager",
            new=AsyncMock(return_value=manager),
        ),
        patch.object(strategy, "_join_caller", new=AsyncMock(return_value=False)),
    ):
        joined = await strategy.execute_transfer(
            {"call_control_id": "caller-leg", "api_key": "key"}
        )

    assert joined is False
    assert await manager.get_transfer_context("tx-1") is None


@pytest.mark.asyncio
async def test_destination_hangup_after_the_bridge_hangs_the_caller_up():
    from api.services.telephony.providers.telnyx import routes

    fake, manager, context = _manager_with(_context())
    await manager.store_transfer_context(context)
    request = SimpleNamespace(
        json=AsyncMock(
            return_value={
                "data": {
                    "event_type": "call.hangup",
                    "payload": {
                        "call_control_id": "dest-leg",
                        "hangup_cause": "normal_clearing",
                    },
                }
            }
        )
    )
    hangup = AsyncMock()
    with (
        patch.object(routes, "get_call_transfer_manager", new=AsyncMock(return_value=manager)),
        patch.object(routes, "_hangup_caller_leg", new=hangup),
    ):
        answer = await routes.handle_telnyx_transfer_result("tx-1", request)

    assert answer == {"status": "success"}
    hangup.assert_awaited_once()
    assert hangup.await_args.args[0].original_call_sid == "caller-leg"
    # The bridge is over: nothing is left behind for the next call to find.
    assert await manager.get_transfer_context("tx-1") is None


def _caller_hangup_request(call_control_id: str = "caller-leg"):
    body = json.dumps(
        {
            "data": {
                "event_type": "call.hangup",
                "payload": {
                    "call_control_id": call_control_id,
                    "connection_id": "conn",
                    "hangup_cause": "normal_clearing",
                },
            }
        }
    )

    async def receive():
        return {"type": "http.request", "body": body.encode("utf-8"), "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/telephony/telnyx/events/4437",
            "query_string": b"",
            "headers": [],
        },
        receive,
    )


def _signed_provider():
    return SimpleNamespace(
        api_key="key",
        verify_inbound_signature=AsyncMock(return_value=True),
        parse_status_callback=lambda event: {
            "call_id": "caller-leg",
            "status": "completed",
            "extra": {},
        },
    )


async def _caller_hangs_up_with(context: TransferContext) -> tuple[AsyncMock, CallTransferManager]:
    """Drive the caller-leg events route with a call.hangup while a transfer
    context stands, and return what the hangup strategy was asked to do."""
    from api.services.telephony.providers.telnyx import routes

    _, manager, _ = _manager_with(context)
    await manager.store_transfer_context(context)
    hangup = AsyncMock(return_value=True)
    with (
        patch.object(routes, "db_client") as db_client,
        patch.object(routes.ws_auth, "token_configured", return_value=False),
        patch.object(routes, "get_telephony_provider_for_run", new=AsyncMock(return_value=_signed_provider())),
        patch.object(routes, "_process_status_update", new=AsyncMock()),
        patch.object(routes, "get_call_transfer_manager", new=AsyncMock(return_value=manager)),
        patch.object(routes.TelnyxHangupStrategy, "execute_hangup", new=hangup),
    ):
        db_client.get_workflow_run_by_id = AsyncMock(return_value=SimpleNamespace(workflow_id=7))
        db_client.get_workflow_by_id = AsyncMock(return_value=SimpleNamespace(organization_id=11))
        answer = await routes.handle_telnyx_events(_caller_hangup_request(), workflow_run_id=4437)
    assert answer == {"status": "success"}
    return hangup, manager


@pytest.mark.asyncio
async def test_caller_hanging_up_while_the_destination_rings_hangs_the_destination_up():
    # Runs 4437 and 4438: the caller left 5 s and 23 s into the ring, and the
    # team's phone rang on to Telnyx's 30 s timeout.
    hangup, manager = await _caller_hangs_up_with(_context(conference_id=None))

    hangup.assert_awaited_once()
    assert hangup.await_args.args[0]["call_control_id"] == "dest-leg"
    assert await manager.get_transfer_context("tx-1") is None


@pytest.mark.asyncio
async def test_caller_hanging_up_after_the_bridge_leaves_the_conference_to_end_it():
    # Joined with end_conference_on_exit, so the conference takes the
    # destination down; a second hangup here would race it.
    hangup, manager = await _caller_hangs_up_with(_context(conference_id="conf-1"))

    hangup.assert_not_awaited()
    assert await manager.get_transfer_context("tx-1") is not None
