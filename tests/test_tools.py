"""Проверки инструментов каталога, калькулятора и предварительного расчёта."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from mcp_server.db import ServiceRepository
from mcp_server.server import create_mcp
from mcp_server.tools import (
    add_service,
    calculate,
    estimate_project,
    find_by_category,
    find_service,
    get_service_by_id,
    list_services,
    recommend_services,
)
from telegram_bot.ai_client import Consultant, DialogMemory, load_system_prompt, wants_add_service
from telegram_bot.config import Settings, validate_mcp_url
from telegram_bot.mcp_client import to_openai_tools


@pytest.fixture
def repo(tmp_path) -> ServiceRepository:
    database = ServiceRepository(tmp_path / "services.db")
    database.initialize()
    return database


def test_find_service_is_case_insensitive_and_safe(repo: ServiceRepository) -> None:
    found = find_service(repo, "ТЕЛЕГРАМ")
    assert found["ok"] is True
    assert found["count"] == 1
    assert found["services"][0]["name"] == "Телеграм-бот для приёма заявок"

    injected = find_service(repo, "%' OR 1=1 --")
    assert injected["ok"] is True
    assert injected["services"] == []


def test_find_by_category_and_id(repo: ServiceRepository) -> None:
    by_category = find_by_category(repo, "аналитика")
    assert by_category["ok"] is True
    assert by_category["count"] == 2

    by_id = get_service_by_id(repo, 1)
    assert by_id["ok"] is True
    assert by_id["service"]["id"] == 1

    missing = get_service_by_id(repo, 9999)
    assert missing["ok"] is False
    assert missing["error"]["code"] == "not_found"

    unknown = find_by_category(repo, "Несуществующая")
    assert unknown["ok"] is False
    assert unknown["error"]["code"] == "validation_error"


def test_add_service_requires_trusted_admin(repo: ServiceRepository) -> None:
    denied = add_service(
        repo,
        name="Закрытая услуга",
        category="Аналитика",
        description="Описание, которое не должно сохраниться без прав администратора.",
        price=1000,
        duration_days=3,
        trusted_user_id=5,
        admin_ids={42},
    )
    assert denied["ok"] is False
    assert denied["error"]["code"] == "forbidden"
    assert len(repo.all_services()) == 15

    created = add_service(
        repo,
        name="Закрытая услуга",
        category="Аналитика",
        description="Описание, которое сохраняет только доверенный администратор.",
        price=1000,
        duration_days=3,
        trusted_user_id=42,
        admin_ids={42},
    )
    assert created["ok"] is True
    assert created["service"]["name"] == "Закрытая услуга"
    assert len(repo.all_services()) == 16


def test_add_service_rejects_invalid_price(repo: ServiceRepository) -> None:
    for price in (-1, float("nan"), float("inf"), "100", True):
        result = add_service(
            repo,
            name="Услуга с плохой ценой",
            category="Аналитика",
            description="Описание достаточно длинное для проверки валидации цены.",
            price=price,  # type: ignore[arg-type]
            duration_days=3,
            trusted_user_id=1,
            admin_ids={1},
        )
        assert result["ok"] is False
        assert result["error"]["code"] == "validation_error"
    assert len(repo.all_services()) == 15


def test_estimate_project_and_discount(repo: ServiceRepository) -> None:
    plain = estimate_project(repo, [1, 2], 0)
    assert plain["ok"] is True
    assert plain["subtotal"] == 164000
    assert plain["discount_amount"] == 0
    assert plain["total"] == 164000
    assert plain["order_created"] is False
    assert plain["price_is_demo"] is True

    discounted = estimate_project(repo, [1, 2, 2], 10)
    assert discounted["ok"] is True
    assert [item["id"] for item in discounted["items"]] == [1, 2]
    assert discounted["subtotal"] == 164000
    assert discounted["discount_amount"] == 16400
    assert discounted["total"] == 147600

    partial = estimate_project(repo, [1, 9999], 0)
    assert partial["ok"] is True
    assert partial["missing_ids"] == [9999]
    assert partial["total"] == 89000

    too_big = estimate_project(repo, [1], 150)
    assert too_big["ok"] is False
    assert too_big["error"]["code"] == "validation_error"


def test_recommend_services_uses_keywords_and_budget(repo: ServiceRepository) -> None:
    result = recommend_services(repo, "Нужен телеграм-бот для заявок")
    assert result["ok"] is True
    assert result["method"] == "keyword_match"
    assert "embeddings" in result["method_note"]
    names = [item["service"]["name"] for item in result["matches"]]
    assert "Телеграм-бот для приёма заявок" in names
    assert result["matches"][0]["matched_terms"]
    assert result["matches"][0]["explanation"]

    limited = recommend_services(repo, "Нужен телеграм-бот для заявок", budget=80000)
    assert limited["ok"] is True
    limited_names = [item["service"]["name"] for item in limited["matches"]]
    assert "Телеграм-бот для приёма заявок" not in limited_names
    assert "Чат-бот записи клиентов" not in limited_names
    assert limited["omitted_over_budget"] >= 1
    assert "не подставляются" in limited["note"]
    assert all(item["service"]["price"] <= 80000 for item in limited["matches"])

    unrelated = recommend_services(repo, "рецепт домашнего борща")
    assert unrelated["ok"] is True
    assert unrelated["matches"] == []
    assert "нет" in unrelated["note"]


def test_recommend_client_replies_stays_on_task_and_budget(repo: ServiceRepository) -> None:
    phrase = "Хочу автоматизировать ответы клиентам, бюджет до 100 000 рублей"
    result = recommend_services(repo, phrase, budget=100_000)
    names = [item["service"]["name"] for item in result["matches"]]

    assert result["ok"] is True
    assert result["budget"] == 100_000
    assert "Телеграм-бот для приёма заявок" in names
    assert "Автоматизация онбординга сотрудников" not in names
    assert "Чат-бот записи клиентов" not in names
    assert "Бот внутренней поддержки" not in names
    assert all(item["service"]["price"] <= 100_000 for item in result["matches"])
    assert result["count"] >= 1

    from_phrase = recommend_services(repo, phrase)
    assert [item["service"]["name"] for item in from_phrase["matches"]] == names
    assert from_phrase["budget"] == 100_000


def test_calculator_accepts_basic_arithmetic() -> None:
    assert calculate("2 + 2 * 3")["result"] == 8
    assert calculate("(10 - 4) / 2")["result"] == 3
    assert calculate("2 ** 3")["result"] == 8
    assert calculate("7 // 2")["result"] == 3
    assert calculate("89_000")["ok"] is False


def _deep_addition(levels: int) -> str:
    expression = "1"
    for _ in range(levels):
        expression = f"({expression}+1)"
    return expression


def test_calculator_rejects_code_and_dangerous_values() -> None:
    rejected = [
        "__import__('os').system('echo hacked')",
        "eval('1+1')",
        "(1).__class__.__bases__",
        "9 ** 99",
        "10 ** 1000",
        "1 / 0",
        "2 +",
        "(" * 30 + "1" + ")" * 30,
        _deep_addition(20),
        "+".join(["1"] * 20),
        "1" * 201,
        "a + 1",
        "",
    ]
    for expression in rejected:
        result = calculate(expression)
        assert result["ok"] is False, expression
        assert result["error"]["code"] == "validation_error"
        assert "result" not in result


def test_catalog_list_marks_demo_prices(repo: ServiceRepository) -> None:
    catalog = list_services(repo)
    assert catalog["ok"] is True
    assert catalog["count"] == 15
    assert catalog["price_is_demo"] is True
    filtered = list_services(repo, "AI-чат-боты")
    assert filtered["count"] == 3


def test_dialog_memory_keeps_last_ten_messages() -> None:
    memory = DialogMemory(limit=10)
    for index in range(12):
        memory.append(1, "user", f"сообщение {index}")
    stored = memory.get(1)
    assert len(stored) == 10
    assert stored[0]["content"] == "сообщение 2"
    memory.append(2, "user", "другой пользователь")
    assert len(memory.get(1)) == 10
    memory.reset(1)
    assert memory.get(1) == []
    assert len(memory.get(2)) == 1


def test_system_prompt_sends_add_service_to_the_server() -> None:
    prompt = load_system_prompt().casefold()
    assert "add_service" in prompt
    assert "не решай сам" in prompt
    assert "не отказывай до вызова" in prompt
    assert "forbidden" in prompt


def test_add_service_description_forbids_model_precheck(tmp_path) -> None:
    mcp = create_mcp(db_path=tmp_path / "services.db", admin_ids={7179176116})
    tools = asyncio.run(mcp.list_tools())
    description = next(tool.description for tool in tools if tool.name == "add_service").casefold()
    assert "всегда вызывай" in description
    assert "не отказывай до вызова" in description
    assert "только администратору" not in description


def test_openai_tools_keep_add_service() -> None:
    listed = [
        SimpleNamespace(name="list_services", description="список", inputSchema={"type": "object"}),
        SimpleNamespace(
            name="add_service",
            description="Всегда вызывай. Не отказывай до вызова.",
            inputSchema={"type": "object", "properties": {"name": {"type": "string"}}},
        ),
    ]
    converted = to_openai_tools(listed)
    assert [item["function"]["name"] for item in converted] == ["list_services", "add_service"]


def test_add_request_detection_does_not_match_catalog_questions() -> None:
    assert wants_add_service("Добавь услугу «Разбор процесса»")
    assert wants_add_service("Хочу создать новую услугу для заявок")
    assert not wants_add_service("Покажи каталог услуг")
    assert not wants_add_service("Кто может добавить скидку?")


def test_text_refusal_still_calls_add_service(caplog) -> None:
    script = [
        _openai_message("Добавление услуги доступно только администраторам."),
        _openai_message(
            None,
            tool_calls=[
                _tool_call(
                    "add_service",
                    {
                        "name": "Разбор процесса",
                        "category": "Аналитика",
                        "description": "Учебная услуга",
                        "price": 40000,
                        "duration_days": 5,
                    },
                )
            ],
        ),
        _openai_message("Услуга передана на сервер. Цена демонстрационная."),
    ]
    openai = _FakeOpenAI(script)
    mcp = _FakeMCP()
    consultant = Consultant(_settings(), mcp, openai)
    consultant.memory.append(5, "assistant", "Добавление доступно только администраторам.")
    caplog.set_level("INFO", logger="telegram_bot.ai")

    reply = asyncio.run(
        consultant.answer(5, "Добавь услугу «Разбор процесса», категория Аналитика, 40000 рублей, 5 дней")
    )

    assert mcp.calls[0][0] == "add_service"
    assert mcp.calls[0][1]["name"] == "Разбор процесса"
    assert "telegram_user_id" not in mcp.calls[0][1]
    offered = openai.completions.requests[0]["tools"]
    assert "add_service" in {item["function"]["name"] for item in offered}
    assert openai.completions.requests[1]["tool_choice"]["function"]["name"] == "add_service"
    assert "tool_choice" not in openai.completions.requests[0]
    assert reply.startswith("Услуга передана")
    assert "model decision user_id=5 tool_calls=none" in caplog.text
    assert "model decision user_id=5 tool_calls=add_service" in caplog.text
    assert "forcing tool choice" in caplog.text


def test_ordinary_question_is_not_forced_into_add_service() -> None:
    openai = _FakeOpenAI([_openai_message("Могу показать каталог.")])
    mcp = _FakeMCP()
    consultant = Consultant(_settings(), mcp, openai)

    reply = asyncio.run(consultant.answer(7179176116, "Покажи, чем вы занимаетесь"))

    assert reply == "Могу показать каталог."
    assert mcp.calls == []
    assert "tool_choice" not in openai.completions.requests[0]
    assert len(openai.completions.requests) == 1


def _settings() -> Settings:
    return Settings(
        telegram_token="telegram-token",
        openai_api_key="openai-key",
        openai_model="gpt-4.1-mini",
        mcp_server_url="http://127.0.0.1:8000/mcp",
        mcp_auth_token="mcp-token",
        admin_ids=frozenset({7179176116}),
    )


class _FakeMCP:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    @asynccontextmanager
    async def connect(self, user_id: int):
        yield _FakeSession()

    async def call_json(self, session, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return {"ok": True, "service": {"id": 16, "name": arguments.get("name")}}


class _FakeSession:
    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(name="list_services", description="список", inputSchema={"type": "object"}),
                SimpleNamespace(
                    name="add_service",
                    description="Всегда вызывай. Не отказывай до вызова.",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]
        )


class _FakeOpenAI:
    def __init__(self, script: list) -> None:
        self.completions = _FakeCompletions(script)
        self.chat = SimpleNamespace(completions=self.completions)


class _FakeCompletions:
    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.script.pop(0)


def _openai_message(content: str | None, tool_calls: list | None = None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(name: str, arguments: dict):
    return SimpleNamespace(
        id="call-add",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def test_remote_mcp_url_requires_https_and_token() -> None:
    validate_mcp_url("http://127.0.0.1:8000/mcp", auth_token="")
    validate_mcp_url("https://mcp.example.com/mcp", auth_token="secret")
    with pytest.raises(ValueError):
        validate_mcp_url("http://mcp.example.com/mcp", auth_token="secret")
    with pytest.raises(ValueError):
        validate_mcp_url("https://mcp.example.com/mcp", auth_token="")
    with pytest.raises(ValueError):
        validate_mcp_url("https://user:pass@mcp.example.com/mcp", auth_token="secret")
