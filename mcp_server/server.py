"""FastMCP-сервер каталога AVAHRUS. Транспорт: Streamable HTTP, endpoint /mcp.

Идентификатор администратора читается из заголовка X-Telegram-User-Id.
Его выставляет код Telegram-бота. В аргументах инструмента такого поля нет,
поэтому модель не может подменить права.
"""

from __future__ import annotations

import hmac
import logging
import os
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from starlette.applications import Starlette
from starlette.types import ASGIApp, Receive, Scope, Send

from mcp_server.db import ServiceRepository, default_db_path
from mcp_server.tools import (
    add_service as add_service_tool,
)
from mcp_server.tools import (
    calculate as calculate_tool,
)
from mcp_server.tools import (
    estimate_project as estimate_project_tool,
)
from mcp_server.tools import (
    find_by_category as find_by_category_tool,
)
from mcp_server.tools import (
    find_service as find_service_tool,
)
from mcp_server.tools import (
    get_service_by_id as get_service_by_id_tool,
)
from mcp_server.tools import (
    list_services as list_services_tool,
)
from mcp_server.tools import (
    recommend_services as recommend_services_tool,
)

logger = logging.getLogger("mcp_server")

TELEGRAM_USER_HEADER = "x-telegram-user-id"
AUTHENTICATED_STATE_KEY = "avahrus_client_authenticated"
MCP_PATH = "/mcp"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


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


def parse_telegram_user_id(raw: str | None) -> int | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value.isdigit() or not 1 <= len(value) <= 15:
        return None
    return int(value)


def trusted_telegram_user_id(ctx: Context) -> int | None:
    """Берёт Telegram ID только из уже аутентифицированного HTTP-запроса.

    Заголовок сам по себе не доказательство. Его читаем после того, как
    middleware проверил Bearer-токен или локальное подключение.
    """
    try:
        request = ctx.request_context.request
    except (LookupError, ValueError, AttributeError):
        return None
    if request is None or not _request_was_authenticated(request):
        return None
    return parse_telegram_user_id(request.headers.get(TELEGRAM_USER_HEADER))


def _request_was_authenticated(request: Any) -> bool:
    state = request.scope.get("state")
    if isinstance(state, dict):
        return bool(state.get(AUTHENTICATED_STATE_KEY))
    return bool(getattr(state, AUTHENTICATED_STATE_KEY, False))


class TrustedClientMiddleware:
    """Проверяет секрет MCP и пропускает запрос дальше вместе с исходными заголовками."""

    def __init__(self, app: ASGIApp, auth_token: str) -> None:
        self.app = app
        self.auth_token = auth_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path.rstrip("/") != MCP_PATH:
            await _reject(send, 404, "Not found")
            return
        headers = _header_map(scope)
        if not _request_is_allowed(headers, scope, self.auth_token):
            await _reject(send, 401, "Unauthorized")
            return
        # Метка ставится только после проверки клиента. Инструмент не доверяет
        # Telegram ID, если запрос дошёл без этой метки.
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state[AUTHENTICATED_STATE_KEY] = True
        else:
            setattr(state, AUTHENTICATED_STATE_KEY, True)
        await self.app(scope, receive, send)


def create_mcp(
    *,
    db_path: os.PathLike[str] | str | None = None,
    admin_ids: set[int] | frozenset[int] | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastMCP:
    repository = ServiceRepository(default_db_path() if db_path is None else Path(db_path))
    repository.initialize()
    resolved_admins = frozenset(admin_ids or ())
    mcp = FastMCP(
        name="AVAHRUS Services",
        instructions=(
            "Демонстрационный каталог услуг автоматизации бизнеса AVAHRUS. "
            "Цены учебные. Просьбу добавить услугу передавай в add_service: "
            "право доступа проверяет сервер, клиент не отказывает заранее."
        ),
        host=host,
        port=port,
        streamable_http_path=MCP_PATH,
        json_response=True,
        stateless_http=False,
        session_idle_timeout=30,
    )
    _register_tools(mcp, repository, resolved_admins)
    return mcp


def build_http_app(mcp: FastMCP, *, auth_token: str) -> Starlette:
    app = mcp.streamable_http_app()
    app.add_middleware(TrustedClientMiddleware, auth_token=auth_token)
    return app


def _register_tools(
    mcp: FastMCP,
    repository: ServiceRepository,
    admin_ids: frozenset[int],
) -> None:
    @mcp.tool(structured_output=False)
    def list_services(
        category: Annotated[
            str | None,
            Field(description="Категория каталога. Пустое значение возвращает все активные услуги."),
        ] = None,
    ) -> dict[str, Any]:
        """Вернуть демонстрационный каталог активных услуг AVAHRUS. Цены учебные."""
        return list_services_tool(repository, category)

    @mcp.tool(structured_output=False)
    def find_service(
        name: Annotated[str, Field(description="Часть названия услуги.")],
    ) -> dict[str, Any]:
        """Найти активные услуги по частичному совпадению названия."""
        return find_service_tool(repository, name)

    @mcp.tool(structured_output=False)
    def get_service_by_id(
        service_id: Annotated[int, Field(description="Целочисленный ID услуги из каталога.")],
    ) -> dict[str, Any]:
        """Вернуть подробную информацию об услуге по её ID."""
        return get_service_by_id_tool(repository, service_id)

    @mcp.tool(structured_output=False)
    def find_by_category(
        category: Annotated[str, Field(description="Точное название категории услуг.")],
    ) -> dict[str, Any]:
        """Найти активные услуги выбранной категории."""
        return find_by_category_tool(repository, category)

    @mcp.tool(structured_output=False)
    def add_service(
        name: Annotated[str, Field(description="Название новой услуги.")],
        category: Annotated[str, Field(description="Одна из существующих категорий каталога.")],
        description: Annotated[str, Field(description="Учебное описание услуги.")],
        price: Annotated[float, Field(description="Демонстрационная цена в рублях, не меньше 0.")],
        duration_days: Annotated[int, Field(description="Учебный срок в днях, целое число больше 0.")],
        ctx: Context,
    ) -> dict[str, Any]:
        """Добавить услугу в каталог.

        Всегда вызывай этот инструмент, когда пользователь просит добавить услугу.
        Не проверяй права самостоятельно и не отказывай до вызова: доступ
        подтверждает или отклоняет только сервер. Не передавай Telegram ID
        в аргументах: такого параметра нет.
        """
        user_id = trusted_telegram_user_id(ctx)
        result = add_service_tool(
            repository,
            name=name,
            category=category,
            description=description,
            price=price,
            duration_days=duration_days,
            trusted_user_id=user_id,
            admin_ids=admin_ids,
        )
        if not result.get("ok"):
            logger.info("add_service denied for telegram_user_id=%s", user_id)
        return result

    @mcp.tool(structured_output=False)
    def recommend_services(
        business_task: Annotated[str, Field(description="Задача клиента своими словами.")],
        budget: Annotated[
            float | None,
            Field(description="Необязательный верхний бюджет в рублях."),
        ] = None,
    ) -> dict[str, Any]:
        """Подобрать услуги по теме задачи. Бюджет, если он указан, отсекает более дорогие.

        Нерелевантные услуги не добавляются, чтобы набрать количество.
        Если совпадений нет, результат пустой. Это не embeddings.
        Цены демонстрационные.
        """
        return recommend_services_tool(repository, business_task, budget)

    @mcp.tool(structured_output=False)
    def calculate(
        expression: Annotated[
            str,
            Field(description="Арифметическое выражение из чисел, скобок и операций + - * / // % **."),
        ],
    ) -> dict[str, Any]:
        """Безопасный калькулятор базовой арифметики. Не выполняет код и не ищет услуги."""
        return calculate_tool(expression)

    @mcp.tool(structured_output=False)
    def estimate_project(
        service_ids: Annotated[
            list[int],
            Field(description="ID выбранных услуг. Повторы считаются один раз."),
        ],
        discount_percent: Annotated[
            float,
            Field(description="Скидка в процентах от 0 до 100. Это не согласованная коммерческая скидка."),
        ] = 0,
    ) -> dict[str, Any]:
        """Посчитать предварительную сумму выбранных услуг и скидку. Заказ не оформляется."""
        return estimate_project_tool(repository, service_ids, discount_percent)


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    host = os.getenv("MCP_BIND_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = _port(os.getenv("MCP_PORT", "8000"))
    token = os.getenv("MCP_AUTH_TOKEN", "").strip()
    if host not in _LOOPBACK_HOSTS and not token:
        raise SystemExit(
            "Публичный адрес без MCP_AUTH_TOKEN запрещён. "
            "Оставьте MCP_BIND_HOST=127.0.0.1 или задайте токен и опубликуйте сервер только через HTTPS."
        )
    if host not in _LOOPBACK_HOSTS:
        logger.warning(
            "Сервер слушает не только localhost. Завершите TLS на обратном прокси "
            "и не открывайте порт приложения напрямую в интернет."
        )
    try:
        admin_ids = parse_admin_ids(os.getenv("TELEGRAM_ADMIN_IDS", ""))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    mcp = create_mcp(
        db_path=default_db_path(),
        admin_ids=admin_ids,
        host=host,
        port=port,
    )
    app = build_http_app(mcp, auth_token=token)
    ssl_certfile, ssl_keyfile = _ssl_files()
    scheme = "https" if ssl_certfile else "http"
    logger.info(
        "MCP-сервер AVAHRUS слушает %s://%s:%s%s, администраторов в списке: %s",
        scheme,
        host,
        port,
        MCP_PATH,
        len(admin_ids),
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )


def _port(raw: str) -> int:
    try:
        port = int(raw)
    except ValueError:
        raise SystemExit("MCP_PORT должен быть целым числом") from None
    if not 1 <= port <= 65535:
        raise SystemExit("MCP_PORT должен быть в диапазоне 1..65535")
    return port


def _ssl_files() -> tuple[str | None, str | None]:
    cert = os.getenv("MCP_SSL_CERTFILE", "").strip()
    key = os.getenv("MCP_SSL_KEYFILE", "").strip()
    if bool(cert) != bool(key):
        raise SystemExit("Для HTTPS задайте оба файла: MCP_SSL_CERTFILE и MCP_SSL_KEYFILE")
    return (cert or None), (key or None)


def _request_is_allowed(headers: dict[str, str], scope: Scope, auth_token: str) -> bool:
    if auth_token:
        return _bearer_matches(headers.get("authorization", ""), auth_token)
    client = scope.get("client")
    host = client[0] if client else ""
    return host in {"127.0.0.1", "::1"}


def _bearer_matches(header: str, expected: str) -> bool:
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented.strip():
        return False
    return hmac.compare_digest(presented.strip().encode("utf-8"), expected.encode("utf-8"))


def _header_map(scope: Scope) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for key, value in scope.get("headers") or []:
        try:
            mapped[key.decode("latin-1").lower()] = value.decode("latin-1")
        except UnicodeError:
            continue
    return mapped


async def _reject(send: Send, status: int, message: str) -> None:
    body = message.encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


if __name__ == "__main__":
    main()
