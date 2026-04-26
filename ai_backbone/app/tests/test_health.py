import pytest


@pytest.mark.asyncio
async def test_app_starts(client):
    response = await client.get("/v1/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/v1/health")
    body = response.json()
    assert body["status"] == "ok"
    assert "app" in body


@pytest.mark.asyncio
async def test_request_id_header_returned(client):
    response = await client.get("/v1/health")
    assert "x-request-id" in response.headers
