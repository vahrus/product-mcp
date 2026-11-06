"""Настройки бота. Секреты читаются только из окружения и файла .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from dotenv import load_dotenv

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    openai_api_key: str
    openai_model: str
    mcp_server_url: str
    mcp_auth_token: str
    admin_ids: frozenset[int]
    max_tool_calls: int = 5
    history_limit: int = 10

    @classmethod
    def load(cls) -> Settings:
        load_dotenv()
        telegram_token = os.getenv("TELEGRAM_API_TOKEN", "").strip()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not telegram_token or not openai_api_key:
            raise ValueError(
                "Задайте TELEGRAM_API_TOKEN и OPENAI_API_KEY в файле .env. "
                "Пример переменных лежит в .env.example."
            )
        mcp_server_url = os.getenv("MCP_SERVER_URL", DEFAULT_MCP_URL).strip() or DEFAULT_MCP_URL
        mcp_auth_token = os.getenv("MCP_AUTH_TOKEN", "").strip()
        validate_mcp_url(mcp_server_url, auth_token=mcp_auth_token)
        return cls(
            telegram_token=telegram_token,
            openai_api_key=openai_api_key,
            openai_model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            mcp_server_url=mcp_server_url,
            mcp_auth_token=mcp_auth_token,
            admin_ids=parse_admin_ids(os.getenv("TELEGRAM_ADMIN_IDS", "")),
        )


def parse_admin_ids(raw: str | None) -> frozenset[int]:
    if raw is None or not raw.strip():
        return frozenset()
    admin_ids: set[int] = set()
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        if not value.isdigit() or not 1 <= len(value) <= 15:
            raise ValueError(
                "TELEGRAM_ADMIN_IDS должен содержать числовые Telegram ID через запятую"
            )
        admin_ids.add(int(value))
    return frozenset(admin_ids)


def validate_mcp_url(url: str, *, auth_token: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("MCP_SERVER_URL должен быть адресом HTTP или HTTPS")
    if parsed.username or parsed.password:
        raise ValueError("MCP_SERVER_URL не должен содержать логин или пароль")
    if parsed.path.rstrip("/") != "/mcp":
        raise ValueError("MCP_SERVER_URL должен указывать на endpoint /mcp")
    host = parsed.hostname.lower()
    if host not in _LOCAL_HOSTS:
        if parsed.scheme != "https":
            raise ValueError("Удалённый MCP-сервер можно подключать только по HTTPS")
        if not auth_token:
            raise ValueError("Для удалённого MCP-сервера задайте MCP_AUTH_TOKEN")
