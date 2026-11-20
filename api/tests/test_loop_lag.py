"""Unit tests for the event-loop lag gauge.

The gauge is the per-pod saturation signal surfaced on /health/active-calls.
It imports only asyncio, so these run anywhere with no external services.
"""

import asyncio

import pytest

from api.services.observability import loop_lag


@pytest.fixture(autouse=True)
def _reset_gauge():
    """Isolate module globals between tests (single shared gauge)."""
    loop_lag._samples.clear()
    task, loop_lag._task = loop_lag._task, None
    if task is not None:
        task.cancel()
    yield
    loop_lag._samples.clear()
    if loop_lag._task is not None:
        loop_lag._task.cancel()
        loop_lag._task = None


def test_stats_empty_is_zero():
    assert loop_lag.stats() == {"p95_ms": 0.0, "max_ms": 0.0, "samples": 0}


def test_percentile_nearest_rank():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert loop_lag._percentile(values, 0) == 1.0
    assert loop_lag._percentile(values, 95) == 100.0  # index clamps to last
    assert loop_lag._percentile([], 95) == 0.0


@pytest.mark.asyncio
async def test_start_is_idempotent_and_holds_reference():
    t1 = loop_lag.start()
    t2 = loop_lag.start()
    assert t1 is t2, "start() must not spawn a second monitor"
    assert loop_lag._task is t1, "module must keep a strong ref (else GC kills it)"
    await asyncio.sleep(0.25)
    assert not t1.done(), "monitor must still be running"


@pytest.mark.asyncio
async def test_idle_loop_reports_low_lag():
    loop_lag.start()
    await asyncio.sleep(0.5)  # several probe intervals of a healthy loop
    assert loop_lag.stats()["p95_ms"] < 5


@pytest.mark.asyncio
async def test_blocked_loop_is_detected():
    loop_lag.start()
    await asyncio.sleep(0.02)  # let the monitor enter its sleep before we starve it
    # Starve the loop well past the 0.1s probe interval; the monitor's in-flight
    # sleep overshoots and records the lag once it finally gets scheduled.
    end = asyncio.get_running_loop().time() + 0.3
    while asyncio.get_running_loop().time() < end:
        pass
    await asyncio.sleep(0.15)
    assert loop_lag.stats()["max_ms"] > 50


@pytest.mark.asyncio
async def test_a_block_long_enough_to_cut_a_callers_audio_is_logged():
    # Harkbell: copy run 6660 lost a sentence to a 1.2 s block nothing logged.
    import time

    from loguru import logger

    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        loop_lag.start()
        await asyncio.sleep(0.15)
        time.sleep(0.6)  # blocks the loop, as a synchronous call on it does
        await asyncio.sleep(0.15)
        await asyncio.sleep(0.15)
    finally:
        logger.remove(sink)
    import re

    blocked = [float(m.group(1)) for w in warnings if (m := re.search(r"blocked for ([0-9.]+)s", w))]
    assert blocked and max(blocked) >= 0.5, warnings
    # A loop that never blocks says nothing.
    warnings.clear()
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        await asyncio.sleep(0.35)
    finally:
        logger.remove(sink)
    assert not any("Event loop blocked" in w for w in warnings)
