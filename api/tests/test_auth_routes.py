import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routes.auth as auth_routes
from api.routes.auth import router
from api.schemas.auth import LoginRequest, SignupRequest
from api.services.auth import depends as auth_depends
from api.services.auth.depends import get_user


def _make_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_stack_mode_hides_email_password_auth_routes(monkeypatch):
    monkeypatch.setattr(auth_depends, "AUTH_PROVIDER", "stack")
    client = TestClient(_make_test_app())

    signup_response = client.post(
        "/auth/signup",
        json={
            "email": "user@example.com",
            "password": "password123",
            "name": "User",
        },
    )
    login_response = client.post(
        "/auth/login",
        json={
            "email": "user@example.com",
            "password": "password123",
        },
    )

    assert signup_response.status_code == 404
    assert signup_response.json() == {"detail": "Not found"}
    assert login_response.status_code == 404
    assert login_response.json() == {"detail": "Not found"}


def test_signup_disabled_returns_403(monkeypatch):
    monkeypatch.setattr(auth_routes, "ENABLE_SIGNUP", False)
    client = TestClient(_make_test_app())

    response = client.post(
        "/auth/signup",
        json={
            "email": "user@example.com",
            "password": "password123",
            "name": "User",
        },
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Signup is disabled"}


def test_stack_mode_keeps_current_user_route_available(monkeypatch):
    monkeypatch.setattr(auth_depends, "AUTH_PROVIDER", "stack")
    app = _make_test_app()
    app.dependency_overrides[get_user] = lambda: SimpleNamespace(
        id=7,
        email="user@example.com",
        selected_organization_id=42,
        provider_id="stack-user-1",
    )
    client = TestClient(app)

    response = client.get("/auth/me")

    assert response.status_code == 200
    assert response.json() == {
        "id": 7,
        "email": "user@example.com",
        "name": None,
        "organization_id": 42,
        "provider_id": "stack-user-1",
    }


# Harkbell: bcrypt takes about 0.2 s and the engine has one event loop, which
# carries every live call's audio. A password checked on the loop is a fifth
# of a second in which no call is heard or answered; Harkbell's API logs in as
# each business's own account, so it happens in the middle of other calls.


async def _longest_gap_while(work) -> float:
    """The longest the event loop went without running, while ``work`` ran."""
    longest = 0.0
    done = False

    async def tick():
        nonlocal longest
        last = time.monotonic()
        while not done:
            await asyncio.sleep(0.005)
            now = time.monotonic()
            longest = max(longest, now - last)
            last = now

    ticker = asyncio.create_task(tick())
    # Let the ticker take its first reading: a route that never suspends
    # before its slow call would otherwise finish before the ticker started,
    # and a blocked loop would measure as no gap at all.
    await asyncio.sleep(0.02)
    try:
        await work()
    finally:
        done = True
        await ticker
    return longest


SLOW_SECS = 0.3


async def test_login_checks_the_password_off_the_event_loop(monkeypatch):
    def slow_verify(password, password_hash):
        time.sleep(SLOW_SECS)
        return False

    async def user_by_email(email):
        return SimpleNamespace(id=1, email=email, password_hash="hash", provider_id="p")

    monkeypatch.setattr(auth_routes, "verify_password", slow_verify)
    monkeypatch.setattr(auth_routes.db_client, "get_user_by_email", user_by_email)

    async def login():
        with pytest.raises(HTTPException) as refused:
            await auth_routes.login(LoginRequest(email="user@example.com", password="password123"))
        assert refused.value.status_code == 401

    assert await _longest_gap_while(login) < SLOW_SECS / 2


async def test_signup_hashes_the_password_off_the_event_loop(monkeypatch):
    class Stop(Exception):
        pass

    def slow_hash(password):
        time.sleep(SLOW_SECS)
        return "hash"

    async def nobody(email):
        return None

    async def create_user(**kwargs):
        assert kwargs["password_hash"] == "hash"
        raise Stop

    monkeypatch.setattr(auth_routes, "ENABLE_SIGNUP", True)
    monkeypatch.setattr(auth_routes, "hash_password", slow_hash)
    monkeypatch.setattr(auth_routes.db_client, "get_user_by_email", nobody)
    monkeypatch.setattr(auth_routes.db_client, "create_user_with_email", create_user)

    async def signup():
        with pytest.raises(Stop):
            await auth_routes.signup(
                SignupRequest(email="user@example.com", password="password123", name="User")
            )

    assert await _longest_gap_while(signup) < SLOW_SECS / 2
