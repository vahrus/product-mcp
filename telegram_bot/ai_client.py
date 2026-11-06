"""Диалог с OpenAI и вызовы инструментов MCP."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from telegram_bot.config import Settings
from telegram_bot.mcp_client import MCPClient, MCPUnavailable, to_openai_tools, tool_message_text

logger = logging.getLogger("telegram_bot.ai")

CATALOG_UNAVAILABLE = "Каталог услуг сейчас недоступен. Запустите MCP-сервер и попробуйте ещё раз."
MODEL_UNAVAILABLE = "Не удалось получить ответ языковой модели. Попробуйте ещё раз чуть позже."
GENERIC_ERROR = "Не получилось обработать сообщение. Попробуйте ещё раз."
TOOL_LIMIT_NOTE = "Достигнут лимит вызовов инструментов для одного сообщения."
MAX_USER_TEXT = 4000
_TEXT_PREVIEW = 160
_ADD_SERVICE_REQUEST = re.compile(
    r"(?:добав\w*|созда\w*|завест\w*|завед\w*|внес\w*)\s+(?:новую\s+|новая\s+|новое\s+)?услуг",
    re.IGNORECASE,
)
_ADD_NUDGE = (
    "Пользователь просит добавить услугу. Не отвечай отказом и не проверяй права. "
    "Вызови инструмент add_service с данными из сообщения. "
    "Решение о доступе принимает только MCP-сервер."
)


class DialogMemory:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._items: dict[int, list[dict[str, str]]] = {}

    def get(self, user_id: int) -> list[dict[str, str]]:
        return [dict(item) for item in self._items.get(user_id, [])]

    def append(self, user_id: int, role: str, content: str) -> None:
        items = self._items.setdefault(user_id, [])
        items.append({"role": role, "content": content})
        if len(items) > self.limit:
            del items[:-self.limit]

    def reset(self, user_id: int) -> None:
        self._items.pop(user_id, None)


class Consultant:
    def __init__(self, settings: Settings, mcp_client: MCPClient, openai_client: AsyncOpenAI) -> None:
        self.settings = settings
        self.mcp = mcp_client
        self.openai = openai_client
        self.memory = DialogMemory(settings.history_limit)
        self.system_prompt = load_system_prompt()
        self._locks: dict[int, asyncio.Lock] = {}

    async def answer(self, user_id: int, text: str) -> str:
        cleaned = " ".join(text.split())
        if not cleaned:
            return "Напишите задачу словами или откройте /help."
        cleaned = cleaned[:MAX_USER_TEXT]
        async with self._lock(user_id):
            history = self.memory.get(user_id)
            try:
                reply = await self._complete(user_id, history, cleaned)
            except MCPUnavailable:
                logger.exception("MCP недоступен для пользователя %s", user_id)
                return CATALOG_UNAVAILABLE
            except OpenAIError:
                logger.exception("OpenAI недоступен для пользователя %s", user_id)
                return MODEL_UNAVAILABLE
            except Exception:
                logger.exception("Сбой диалога для пользователя %s", user_id)
                return GENERIC_ERROR
            self.memory.append(user_id, "user", cleaned)
            self.memory.append(user_id, "assistant", reply)
            return reply

    async def catalog(self, user_id: int) -> str:
        async with self._lock(user_id):
            try:
                async with self.mcp.connect(user_id) as session:
                    payload = await self.mcp.call_json(session, "list_services", {})
            except MCPUnavailable:
                logger.exception("Каталог недоступен для пользователя %s", user_id)
                return CATALOG_UNAVAILABLE
            except Exception:
                logger.exception("Не удалось прочитать каталог для пользователя %s", user_id)
                return CATALOG_UNAVAILABLE
        return format_catalog(payload)

    async def reset(self, user_id: int) -> None:
        async with self._lock(user_id):
            self.memory.reset(user_id)

    async def _complete(
        self,
        user_id: int,
        history: list[dict[str, str]],
        text: str,
    ) -> str:
        async with self.mcp.connect(user_id) as session:
            listed = await session.list_tools()
            tools = to_openai_tools(list(listed.tools))
            if not tools:
                raise MCPUnavailable("empty tool list")
            tool_names = [item["function"]["name"] for item in tools]
            logger.info("tools offered user_id=%s names=%s", user_id, ",".join(tool_names))
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": self.system_prompt},
                *history,
                {"role": "user", "content": text},
            ]
            used = 0
            asked_to_add = wants_add_service(text)
            forced_add = False
            called_add = False
            while used < self.settings.max_tool_calls:
                request: dict[str, Any] = {
                    "model": self.settings.openai_model,
                    "messages": messages,
                    "tools": tools,
                }
                if forced_add and "add_service" in tool_names:
                    request["tool_choice"] = {
                        "type": "function",
                        "function": {"name": "add_service"},
                    }
                response = await self.openai.chat.completions.create(**request)
                message = response.choices[0].message
                calls = _function_calls(message)
                _log_model_decision(user_id, calls, message.content)
                if not calls:
                    if asked_to_add and not called_add and not forced_add and "add_service" in tool_names:
                        forced_add = True
                        logger.info("model skipped add_service, forcing tool choice user_id=%s", user_id)
                        if message.content:
                            messages.append({"role": "assistant", "content": message.content})
                        messages.append({"role": "user", "content": _ADD_NUDGE})
                        continue
                    return _assistant_text(message.content)
                messages.append(_assistant_payload(message))
                for call in calls:
                    if used >= self.settings.max_tool_calls:
                        messages.append(_tool_result(call.id, _limit_payload()))
                        continue
                    used += 1
                    if call.function.name == "add_service":
                        called_add = True
                    payload = await self._invoke(session, user_id, call.function.name, call.function.arguments)
                    messages.append(_tool_result(call.id, payload))
            final = await self.openai.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
            )
            return _assistant_text(final.choices[0].message.content)

    async def _invoke(self, session: Any, user_id: int, name: str, raw_arguments: str | None) -> dict[str, Any]:
        logger.info("tool call user_id=%s tool=%s", user_id, name)
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            return {
                "ok": False,
                "error": {"code": "validation_error", "message": "Аргументы инструмента не являются JSON."},
            }
        if not isinstance(arguments, dict):
            return {
                "ok": False,
                "error": {"code": "validation_error", "message": "Аргументы инструмента должны быть объектом."},
            }
        return await self.mcp.call_json(session, name, arguments)

    def _lock(self, user_id: int) -> asyncio.Lock:
        lock = self._locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[user_id] = lock
        return lock


def wants_add_service(text: str) -> bool:
    normalized = text.casefold().replace("ё", "е")
    return _ADD_SERVICE_REQUEST.search(normalized) is not None


def load_system_prompt() -> str:
    path = Path(__file__).with_name("system_prompt.txt")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Файл system_prompt.txt пуст")
    return text


def format_catalog(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return CATALOG_UNAVAILABLE
    services = payload.get("services")
    if not isinstance(services, list) or not services:
        return "В демонстрационном каталоге пока нет активных услуг."
    lines = [
        "Демонстрационный каталог AVAHRUS.",
        "Цены учебные и не являются офертой.",
        "",
    ]
    for service in services:
        if not isinstance(service, dict):
            continue
        lines.append(
            f"{service.get('id')}. {service.get('name')}\n"
            f"Категория: {service.get('category')}\n"
            f"Цена: {format_price(service.get('price'))}, срок: {service.get('duration_days')} дн.\n"
            f"{service.get('description')}"
        )
        lines.append("")
    return "\n".join(lines).strip()


def format_price(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "цена не указана"
    if isinstance(value, float) and value.is_integer():
        amount = f"{int(value):,}".replace(",", " ")
    elif isinstance(value, int):
        amount = f"{value:,}".replace(",", " ")
    else:
        amount = f"{value:,.2f}".replace(",", " ")
    return f"{amount} ₽"


def split_text(text: str, limit: int = 3900) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return ["Пустой ответ."]
    parts: list[str] = []
    rest = cleaned
    while rest:
        if len(rest) <= limit:
            parts.append(rest)
            break
        cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    return [part for part in parts if part]


def _log_model_decision(user_id: int, calls: list[Any], content: str | None) -> None:
    if calls:
        names = ",".join(call.function.name for call in calls)
        logger.info("model decision user_id=%s tool_calls=%s", user_id, names)
        return
    preview = (content or "").replace("\n", " ")[:_TEXT_PREVIEW]
    logger.info("model decision user_id=%s tool_calls=none text=%s", user_id, preview)


def _function_calls(message: Any) -> list[Any]:
    calls = []
    for call in message.tool_calls or []:
        function = getattr(call, "function", None)
        if function is None or not getattr(function, "name", None):
            continue
        calls.append(call)
    return calls


def _assistant_payload(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }
    calls = _function_calls(message)
    if calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }
            for call in calls
        ]
    return payload


def _tool_result(tool_call_id: str, payload: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": tool_message_text(payload),
    }


def _limit_payload() -> dict[str, Any]:
    return {"ok": False, "error": {"code": "tool_limit", "message": TOOL_LIMIT_NOTE}}


def _assistant_text(content: str | None) -> str:
    text = (content or "").strip()
    if not text:
        return "Я не смог сформулировать ответ. Уточните задачу, пожалуйста."
    return text
