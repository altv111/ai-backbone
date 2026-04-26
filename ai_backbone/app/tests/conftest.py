import httpx
import pytest_asyncio

from app.main import app


@pytest_asyncio.fixture
async def app_instance():
    await app.router._startup()
    try:
        yield app
    finally:
        await app.router._shutdown()


@pytest_asyncio.fixture
async def client(app_instance):
    transport = httpx.ASGITransport(app=app_instance)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client
