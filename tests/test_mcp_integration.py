"""Интеграция через настоящий MCP ClientSession и Streamable HTTP."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from mcp_server.server import build_http_app, create_mcp
from telegram_bot.mcp_client import MCPClient, MCPUnavailable, parse_tool_result, to_openai_tools

TOKEN = "test-token"
ADMIN_ID = 1001
TOOL_NAMES = {
    "list_services",
    "find_service",
    "get_service_by_id",
    "find_by_category",
    "add_service",
    "recommend_services",
    "calculate",
    "estimate_project",
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@asynccontextmanager
async def _running_server(app) -> AsyncIterator[str]:
    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError("MCP-сервер не запустился")
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        await task


@asynccontextmanager
async def _session(url: str, *, token: str, user_id: int) -> AsyncIterator[ClientSession]:
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Telegram-User-Id": str(user_id),
    }
    timeout = httpx.Timeout(10.0, read=20.0)
    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=False) as http_client:
        async with streamable_http_client(
            url,
            http_client=http_client,
            terminate_on_close=False,
        ) as (
            read_stream,
            write_stream,
            _get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session


def _payload(result) -> dict:
    return parse_tool_result(result)


def test_client_closes_session_without_delete(monkeypatch) -> None:
    seen: dict[str, object] = {}

    @asynccontextmanager
    async def fake_http(*args, **kwargs):
        seen["terminate_on_close"] = kwargs.get("terminate_on_close")
        yield object(), object(), lambda: None

    class FakeSession:
        def __init__(self, read_stream, write_stream) -> None:
            self.read_stream = read_stream

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def initialize(self) -> None:
            return None

    monkeypatch.setattr("telegram_bot.mcp_client.streamable_http_client", fake_http)
    monkeypatch.setattr("telegram_bot.mcp_client.ClientSession", FakeSession)

    async def run() -> None:
        client = MCPClient("http://127.0.0.1:8000/mcp", "token")
        async with client.connect(7179176116):
            return None

    asyncio.run(run())
    assert seen["terminate_on_close"] is False


def test_mcp_lists_and_calls_tools(tmp_path) -> None:
    asyncio.run(_mcp_roundtrip(tmp_path))


async def _mcp_roundtrip(tmp_path) -> None:
    mcp = create_mcp(
        db_path=tmp_path / "services.db",
        admin_ids={ADMIN_ID},
        host="127.0.0.1",
        port=8000,
    )
    app = build_http_app(mcp, auth_token=TOKEN)
    async with _running_server(app) as url:
        async with _session(url, token=TOKEN, user_id=ADMIN_ID) as session:
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            assert names == TOOL_NAMES

            add_tool = next(tool for tool in listed.tools if tool.name == "add_service")
            description = (add_tool.description or "").casefold()
            assert "всегда вызывай" in description
            assert "не отказывай до вызова" in description
            assert "trusted_user_id" not in add_tool.inputSchema.get("properties", {})
            assert "admin_id" not in add_tool.inputSchema.get("properties", {})
            assert "telegram_user_id" not in add_tool.inputSchema.get("properties", {})

            openai_tools = to_openai_tools(list(listed.tools))
            assert len(openai_tools) == 8
            assert openai_tools[0]["type"] == "function"
            assert openai_tools[0]["function"]["parameters"]["type"] == "object"

            catalog = _payload(await session.call_tool("list_services", {}))
            assert catalog["ok"] is True
            assert catalog["count"] == 15

            found = _payload(await session.call_tool("find_service", {"name": "воронке"}))
            assert found["ok"] is True
            assert found["count"] == 1

            created = _payload(
                await session.call_tool(
                    "add_service",
                    {
                        "name": "Учебная услуга администратора",
                        "category": "Аналитика",
                        "description": "Добавлена через настоящий MCP-вызов от администратора.",
                        "price": 32000,
                        "duration_days": 6,
                    },
                )
            )
            assert created["ok"] is True
            assert created["service"]["id"] == 16

            calculated = _payload(await session.call_tool("calculate", {"expression": "32000 + 89000"}))
            assert calculated == {"ok": True, "result": 121000}

        async with _session(url, token=TOKEN, user_id=5) as session:
            denied = _payload(
                await session.call_tool(
                    "add_service",
                    {
                        "name": "Чужая услуга",
                        "category": "Аналитика",
                        "description": "Эту услугу обычный пользователь добавить не может.",
                        "price": 1000,
                        "duration_days": 2,
                    },
                )
            )
            assert denied["ok"] is False
            assert denied["error"]["code"] == "forbidden"

        with pytest.raises(Exception):
            async with _session(url, token="wrong-token", user_id=ADMIN_ID) as session:
                await session.list_tools()


def test_add_service_allows_admin_and_rejects_regular_user(tmp_path) -> None:
    asyncio.run(_admin_and_regular_user(tmp_path))


async def _admin_and_regular_user(tmp_path) -> None:
    mcp = create_mcp(
        db_path=tmp_path / "services.db",
        admin_ids={ADMIN_ID},
        host="127.0.0.1",
        port=8000,
    )
    app = build_http_app(mcp, auth_token=TOKEN)
    payload = {
        "name": "Услуга из цепочки бота",
        "category": "Аналитика",
        "description": "Проверка, что администратор проходит доверенный заголовок.",
        "price": 41000,
        "duration_days": 5,
    }
    async with _running_server(app) as url:
        admin_client = MCPClient(url, TOKEN)
        async with admin_client.connect(ADMIN_ID) as session:
            created = await admin_client.call_json(session, "add_service", payload)
        assert created["ok"] is True
        assert created["service"]["name"] == payload["name"]

        regular_client = MCPClient(url, TOKEN)
        forged = dict(payload)
        forged["name"] = "Услуга обычного пользователя"
        forged["telegram_user_id"] = ADMIN_ID
        async with regular_client.connect(5) as session:
            denied = await regular_client.call_json(session, "add_service", forged)
        assert denied["ok"] is False
        assert denied["error"]["code"] == "forbidden"

        spoofed_client = MCPClient(url, "wrong-token")
        with pytest.raises(MCPUnavailable):
            async with spoofed_client.connect(ADMIN_ID) as session:
                await spoofed_client.call_json(session, "add_service", payload)


def test_tool_result_parser_hides_protocol_errors() -> None:
    from mcp.types import CallToolResult, TextContent

    failed = CallToolResult(
        content=[TextContent(type="text", text="Traceback: secret token sk-test")],
        isError=True,
    )
    payload = parse_tool_result(failed)
    assert payload["ok"] is False
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "sk-test" not in encoded
    assert "Traceback" not in encoded
