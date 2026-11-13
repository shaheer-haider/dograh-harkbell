"""A Telnyx media stream that opens and never starts, restarted once.

Harkbell T-679. Production runs 4727 and 4735 (2026-11-12, the copy of Mirror
Labs, inbound): the call was answered, the media WebSocket was accepted ~20 ms
later, Telnyx's ``call.answered`` and ``streaming.started`` webhooks arrived --
and the socket never carried a ``start`` message. The handler waited in
``receive_text()`` for the whole call; the caller heard nothing and hung up.
Both were the same caller redialling about a second after the previous call
ended (2 of 6 such redials, none of ~14 spaced calls or 8 concurrent ones).
Nothing on the engine side reads or redirects a new socket's frames, so the
cause is not known; what is known is that the caller sits in silence.

So an answered inbound call whose socket brings no ``start`` within
``START_TIMEOUT_SECS`` is restarted: the engine asks Telnyx to stop streaming
and start again to the same URL, and the run admits exactly one more socket
for it -- the media endpoint otherwise refuses a run that is no longer
``initialized``. The first socket's handler returns without a pipeline the
moment it gives up on it, so a ``start`` arriving on it late is never read and
only the second socket can ever run one. If the second socket brings no
``start`` either, the call is hung up rather than left silent, and the run is
failed with the reason.

The admission is a Redis key, because the second socket may reach a different
API process from the first; taking it is a GETDEL, so at most one more socket
is ever let in. Outbound calls open their socket while the far end is still
ringing and send ``start`` only on answer, so they are never timed.
"""

import time
from typing import Any

import redis.asyncio as aioredis
from loguru import logger

from api.constants import REDIS_URL

# A normal start arrives 100-350 ms after the socket is accepted.
START_TIMEOUT_SECS = 3.0
# Long enough for Telnyx to dial back, short enough to be gone by the next call.
SECOND_SOCKET_TTL_SECS = 30

_ACCEPTED_AT = "telnyx_media_accepted_at"
_SECOND_SOCKET = "telnyx_media_second_socket"

_redis: aioredis.Redis | None = None


def _key(workflow_run_id: int) -> str:
    return f"telnyx:media:second-socket:{workflow_run_id}"


async def _client() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


async def allow_second_socket(workflow_run_id: int) -> bool:
    """Let one more media socket in for a run whose first never started."""
    try:
        await (await _client()).setex(_key(workflow_run_id), SECOND_SOCKET_TTL_SECS, "1")
        return True
    except Exception as e:  # noqa: BLE001 - recovery is best-effort
        logger.error(f"[run {workflow_run_id}] Could not admit a second media socket: {e}")
        return False


async def take_second_socket(workflow_run_id: int) -> bool:
    """Whether this socket is the one second socket the run was allowed."""
    try:
        return bool(await (await _client()).getdel(_key(workflow_run_id)))
    except Exception as e:  # noqa: BLE001 - an unreadable key admits nothing
        logger.error(f"[run {workflow_run_id}] Could not read the second-socket key: {e}")
        return False


def _state(websocket: Any):
    return getattr(websocket, "state", None)


def note_accepted(websocket: Any) -> None:
    """Remember when the media socket was accepted, to time its first frames."""
    state = _state(websocket)
    if state is not None:
        setattr(state, _ACCEPTED_AT, time.monotonic())


def accepted_at(websocket: Any) -> float | None:
    state = _state(websocket)
    value = getattr(state, _ACCEPTED_AT, None) if state is not None else None
    return value if isinstance(value, (int, float)) else None


def mark_second_socket(websocket: Any) -> None:
    state = _state(websocket)
    if state is not None:
        setattr(state, _SECOND_SOCKET, True)


def is_second_socket(websocket: Any) -> bool:
    state = _state(websocket)
    return state is not None and getattr(state, _SECOND_SOCKET, False) is True
