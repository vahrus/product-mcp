"""Реализация инструментов каталога. Регистрация в MCP находится в server.py.

Подбор услуг — объяснимый поиск по словам, основам слов и категориям.
Это не семантический поиск и не embeddings.
Калькулятор разбирает выражение через AST и не вызывает eval или exec.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from mcp_server.db import CATEGORIES, DatabaseError, ServiceRepository, canonical_category

MAX_NAME_LENGTH = 150
MAX_DESCRIPTION_LENGTH = 2000
MAX_TASK_LENGTH = 2000
MAX_QUERY_LENGTH = 200
MAX_PRICE = 100_000_000
MAX_DURATION_DAYS = 3650
MAX_SERVICES_IN_ESTIMATE = 20
MAX_RECOMMENDATIONS = 5
MIN_RECOMMENDATION_SCORE = 4

MAX_EXPRESSION_LENGTH = 200
MAX_AST_DEPTH = 12
MAX_AST_NODES = 80
MAX_ABS_RESULT = 10**12
MAX_EXPONENT = 8
MAX_POWER_BASE = 10**6

_ALLOWED_EXPRESSION_CHARS = set("0123456789+-*/%.() \t")
_TOKEN_RE = re.compile(r"[0-9a-zа-я]{3,}")
_BUDGET_RE = re.compile(
    r"бюджет(?:\s+[^\d\s]+){0,4}\s+(\d{1,3}(?:[ \u00a0]\d{3})+|\d{4,8})",
    re.IGNORECASE,
)
# Общие слова вроде «автоматизировать» намеренно не входят в темы:
# иначе любая услуга из «Автоматизации процессов» кажется подходящей.
_CONCEPTS: dict[str, tuple[str, ...]] = {
    "client_dialog": ("ответ", "отвеч", "переписк", "диалог", "обращен", "консультац"),
    "requests": ("заяв",),
    "employees": ("сотрудник", "онбординг", "персонал"),
    "booking": ("запис", "расписан"),
    "documents": ("договор", "документ", "счет", "коммерческ", "предложен"),
    "sales": ("продаж", "воронк", "сделк", "досье"),
    "analytics": ("аналитик", "дашборд", "отчет", "конверс"),
    "integration": ("интеграц", "crm", "1с", "1c"),
    "knowledge": ("знан", "регламент", "инструкц"),
    "messengers": ("телеграм", "мессенджер"),
    "clients": ("клиент",),
    "bots": ("бот",),
}
_BROAD_CONCEPTS = frozenset({"clients", "bots"})
# Тема в названии описывает саму услугу. Если её нет в задаче, услуга про другое.
_NAME_CONFLICTS = frozenset(
    {
        "employees",
        "booking",
        "documents",
        "sales",
        "analytics",
        "integration",
        "knowledge",
    }
)
_CONCEPT_LABELS = {
    "client_dialog": "ответы и диалог с клиентом",
    "requests": "заявки",
    "employees": "сотрудники",
    "booking": "запись клиентов",
    "documents": "документы",
    "sales": "продажи",
    "analytics": "аналитика",
    "integration": "интеграции",
    "knowledge": "база знаний",
    "messengers": "мессенджеры",
    "clients": "клиенты",
    "bots": "бот",
}
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

KEYWORD_METHOD = "keyword_match"
KEYWORD_METHOD_NOTE = (
    "Подбор выполнен сравнением слов, основ слов и категорий. "
    "Это не семантический поиск и не embeddings."
)


class CalculatorError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def list_services(repo: ServiceRepository, category: str | None = None) -> dict[str, Any]:
    """Активный каталог, опционально отфильтрованный по категории."""
    selected_category, error = _optional_category(category)
    if error is not None:
        return error
    services = [item for item in repo.all_services() if item["active"]]
    if selected_category is not None:
        services = [item for item in services if item["category"] == selected_category]
    return _ok(
        category=selected_category,
        count=len(services),
        services=services,
        price_is_demo=True,
    )


def find_service(repo: ServiceRepository, name: str) -> dict[str, Any]:
    """Поиск активных услуг по части названия. Регистр и буква ё не различаются."""
    query, error = _text(name, minimum=2, maximum=MAX_QUERY_LENGTH, field="Название")
    if error is not None:
        return error
    folded = _normalize(query)
    services = [
        item
        for item in repo.all_services()
        if item["active"] and folded in _normalize(item["name"])
    ]
    return _ok(query=query, count=len(services), services=services, price_is_demo=True)


def get_service_by_id(repo: ServiceRepository, service_id: int) -> dict[str, Any]:
    parsed, error = _service_id(service_id)
    if error is not None:
        return error
    service = repo.get_by_id(parsed)
    if service is None:
        return _fail("not_found", "Услуга с таким номером не найдена")
    return _ok(service=service, price_is_demo=True)


def find_by_category(repo: ServiceRepository, category: str) -> dict[str, Any]:
    selected, error = _required_category(category)
    if error is not None:
        return error
    services = [
        item
        for item in repo.all_services()
        if item["active"] and item["category"] == selected
    ]
    return _ok(category=selected, count=len(services), services=services, price_is_demo=True)


def add_service(
    repo: ServiceRepository,
    *,
    name: str,
    category: str,
    description: str,
    price: float,
    duration_days: int,
    trusted_user_id: int | None,
    admin_ids: set[int] | frozenset[int],
) -> dict[str, Any]:
    """Добавляет услугу, если доверенный идентификатор входит в список администраторов.

    trusted_user_id приходит из HTTP-заголовка, который выставляет код бота.
    Аргументы модели этот идентификатор не содержат.
    """
    if trusted_user_id is None or trusted_user_id not in admin_ids:
        return _fail(
            "forbidden",
            "Недостаточно прав: Telegram-пользователь этого запроса не входит в список администраторов. "
            "Идентификатор взят из доверенного заголовка после аутентификации клиента, а не из текста диалога.",
        )

    clean_name, error = _text(name, minimum=3, maximum=MAX_NAME_LENGTH, field="Название")
    if error is not None:
        return error
    selected_category, error = _required_category(category)
    if error is not None:
        return error
    clean_description, error = _text(
        description,
        minimum=10,
        maximum=MAX_DESCRIPTION_LENGTH,
        field="Описание",
    )
    if error is not None:
        return error
    clean_price, error = _price(price)
    if error is not None:
        return error
    clean_duration, error = _duration(duration_days)
    if error is not None:
        return error

    try:
        created = repo.add_service(
            name=clean_name,
            category=selected_category,
            description=clean_description,
            price=clean_price,
            duration_days=clean_duration,
        )
    except DatabaseError as exc:
        return _fail(exc.code, exc.message)
    return _ok(
        service=created,
        message="Услуга добавлена в демонстрационный каталог. Заказ при этом не оформлен.",
        price_is_demo=True,
    )


def recommend_services(
    repo: ServiceRepository,
    business_task: str,
    budget: float | None = None,
) -> dict[str, Any]:
    task, error = _text(business_task, minimum=3, maximum=MAX_TASK_LENGTH, field="Описание задачи")
    if error is not None:
        return error
    limit, error = _optional_budget(budget)
    if error is not None:
        return error
    text_budget = _budget_in_text(task)
    if text_budget is not None:
        limit = text_budget if limit is None else min(limit, text_budget)

    ranked: list[dict[str, Any]] = []
    omitted_over_budget = 0
    for service in repo.all_services():
        if not service["active"]:
            continue
        score, terms = _explain_match(task, service)
        if score < MIN_RECOMMENDATION_SCORE:
            continue
        if limit is not None and float(service["price"]) > limit:
            omitted_over_budget += 1
            continue
        ranked.append(
            {
                "service": service,
                "score": score,
                "matched_terms": terms,
                "explanation": _explanation(service["category"], terms),
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["service"]["price"], item["service"]["id"]))
    matches = ranked[:MAX_RECOMMENDATIONS]
    if matches:
        note = (
            "Показаны только услуги, в которых нашлась та же задача, что в запросе. "
            "Список не дополняется другими услугами из бюджета."
        )
    elif omitted_over_budget:
        note = (
            "По задаче услуги нашлись, но все они дороже указанного бюджета. "
            "Нерелевантные услуги вместо них не подставляются."
        )
    else:
        note = "Подходящих услуг по этой задаче нет. Уточните задачу другими словами."
    return _ok(
        method=KEYWORD_METHOD,
        method_note=KEYWORD_METHOD_NOTE,
        business_task=task,
        budget=limit,
        count=len(matches),
        matches=matches,
        omitted_over_budget=omitted_over_budget,
        note=note,
        price_is_demo=True,
    )


def calculate(expression: str) -> dict[str, Any]:
    """Считает арифметическое выражение. Код, имена и вызовы функций запрещены."""
    if not isinstance(expression, str):
        return _fail("validation_error", "Выражение должно быть строкой")
    cleaned = expression.strip()
    if not cleaned:
        return _fail("validation_error", "Пустое выражение")
    if len(cleaned) > MAX_EXPRESSION_LENGTH:
        return _fail("validation_error", "Выражение слишком длинное")
    if any(char not in _ALLOWED_EXPRESSION_CHARS for char in cleaned):
        return _fail(
            "validation_error",
            "Разрешены только числа и операции + - * / // % ** и скобки",
        )
    if _paren_depth(cleaned) > MAX_AST_DEPTH:
        return _fail("validation_error", "Слишком сложное выражение")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return _fail("validation_error", "Не удалось разобрать выражение")
    if _ast_size(tree) > MAX_AST_NODES or _ast_depth(tree) > MAX_AST_DEPTH:
        return _fail("validation_error", "Слишком сложное выражение")
    try:
        result = _eval_node(tree.body)
    except CalculatorError as exc:
        return _fail("validation_error", exc.message)
    return _ok(result=_public_number(result))


def estimate_project(
    repo: ServiceRepository,
    service_ids: list[int],
    discount_percent: float = 0,
) -> dict[str, Any]:
    ids, error = _service_ids(service_ids)
    if error is not None:
        return error
    discount, error = _discount(discount_percent)
    if error is not None:
        return error

    items: list[dict[str, Any]] = []
    missing: list[int] = []
    inactive: list[int] = []
    for service_id in ids:
        service = repo.get_by_id(service_id)
        if service is None:
            missing.append(service_id)
            continue
        if not service["active"]:
            inactive.append(service_id)
            continue
        items.append(
            {
                "id": service["id"],
                "name": service["name"],
                "price": service["price"],
                "duration_days": service["duration_days"],
                "currency": "RUB",
            }
        )
    if not items:
        return _fail("not_found", "Не найдено активных услуг с указанными номерами")

    subtotal = sum((Decimal(str(item["price"])) for item in items), Decimal("0"))
    discount_amount = (subtotal * discount / Decimal("100")).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )
    total = (subtotal - discount_amount).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    duration = max(item["duration_days"] for item in items)
    return _ok(
        items=items,
        missing_ids=missing,
        inactive_ids=inactive,
        currency="RUB",
        subtotal=_money(subtotal),
        discount_percent=_money(discount),
        discount_amount=_money(discount_amount),
        total=_money(total),
        duration_days_estimate=duration,
        duration_note="Срок показан по самой долгой из выбранных услуг и не является календарным планом проекта.",
        price_is_demo=True,
        order_created=False,
        note=(
            "Предварительный расчёт по демонстрационным ценам. "
            "Заказ не оформлен и проект не согласован."
        ),
    )


def _explain_match(task: str, service: dict[str, Any]) -> tuple[int, list[str]]:
    """Сравнивает темы запроса с названием и описанием услуги, но не с ярлыком категории.

    Общее слово «автоматизировать» не считается совпадением с «автоматизацией процессов».
    Если в задаче есть конкретная тема, одного слова «клиент» или «бот» недостаточно.
    """
    task_concepts = _concepts_in(task)
    if not task_concepts:
        return 0, []
    name_concepts = _concepts_in(service["name"])
    description_concepts = _concepts_in(service["description"])
    service_concepts = set(name_concepts) | set(description_concepts)
    if "clients" in task_concepts and "employees" in service_concepts and "clients" not in service_concepts:
        return 0, []
    if set(name_concepts) & _NAME_CONFLICTS - set(task_concepts):
        return 0, []
    specific_task = set(task_concepts) - _BROAD_CONCEPTS
    score = 0
    labels: list[str] = []
    shared_specific = False
    for concept in task_concepts:
        in_name = concept in name_concepts
        in_description = concept in description_concepts
        if not in_name and not in_description:
            continue
        if concept not in _BROAD_CONCEPTS:
            shared_specific = True
            score += 6 if in_name else 4
        else:
            score += 2 if in_name else 1
        labels.append(_CONCEPT_LABELS[concept])
    if specific_task and not shared_specific:
        return 0, []
    if score < MIN_RECOMMENDATION_SCORE:
        return 0, []
    return score, labels[:8]


def _concepts_in(text: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for concept, terms in _CONCEPTS.items():
        words = [_display_term(text, term) for term in terms if _term_in(text, term)]
        if words:
            found[concept] = words
    return found


def _budget_in_text(text: str) -> float | None:
    match = _BUDGET_RE.search(_normalize(text))
    if match is None:
        return None
    digits = re.sub(r"\D", "", match.group(1))
    if not digits:
        return None
    value = int(digits)
    if value > MAX_PRICE:
        return None
    return float(value)


def _explanation(category: str, terms: list[str]) -> str:
    if not terms:
        return f"Категория услуги: {category}."
    shown = ", ".join(terms)
    return f"Совпала задача: {shown}. Категория услуги: {category}."


def _display_term(text: str, term: str) -> str:
    for token in _TOKEN_RE.findall(_normalize(text)):
        if token == term or token.startswith(term):
            return token
    return term


def _term_in(text: str, term: str) -> bool:
    normalized = _normalize(text)
    if len(term) <= 3:
        pattern = rf"(?<![0-9a-zа-я]){re.escape(term)}(?![0-9a-zа-я])"
    else:
        pattern = rf"(?<![0-9a-zа-я]){re.escape(term)}"
    return re.search(pattern, normalized) is not None


def _normalize(text: str) -> str:
    return text.casefold().replace("ё", "е")


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        return _number_constant(node.value)
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY:
            raise CalculatorError("Разрешены только числа и базовые арифметические операции")
        value = _UNARY[type(node.op)](_eval_node(node.operand))
        return _check_result(value)
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS and not isinstance(node.op, ast.Pow):
            raise CalculatorError("Разрешены только числа и базовые арифметические операции")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            value = _safe_power(left, right)
        else:
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise CalculatorError("Деление на ноль")
            value = _BINOPS[type(node.op)](left, right)
        return _check_result(value)
    raise CalculatorError("Разрешены только числа и базовые арифметические операции")


def _number_constant(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError("Разрешены только числа и базовые арифметические операции")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("Результат слишком большой")
    return _check_result(value)


def _safe_power(base: int | float, exponent: int | float) -> int | float:
    if isinstance(exponent, float):
        if not exponent.is_integer():
            raise CalculatorError("Степень должна быть целым числом")
        exponent = int(exponent)
    if isinstance(exponent, bool) or not isinstance(exponent, int):
        raise CalculatorError("Степень должна быть целым числом")
    if exponent < 0 or exponent > MAX_EXPONENT:
        raise CalculatorError("Слишком большая степень")
    if abs(base) > MAX_POWER_BASE:
        raise CalculatorError("Слишком большое основание степени")
    if base == 0 and exponent < 0:
        raise CalculatorError("Деление на ноль")
    return base**exponent


def _check_result(value: int | float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError("Результат слишком большой")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("Результат слишком большой")
    if abs(value) > MAX_ABS_RESULT:
        raise CalculatorError("Результат слишком большой")
    return value


def _public_number(value: int | float) -> int | float:
    if isinstance(value, float):
        rounded = round(value, 10)
        if rounded.is_integer():
            return int(rounded)
        return rounded
    return int(value)


def _paren_depth(expression: str) -> int:
    depth = 0
    maximum = 0
    for char in expression:
        if char == "(":
            depth += 1
            maximum = max(maximum, depth)
        elif char == ")":
            depth -= 1
            if depth < 0:
                return MAX_AST_DEPTH + 1
    return maximum


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_ast_depth(child) for child in children)


def _ast_size(node: ast.AST) -> int:
    return 1 + sum(_ast_size(child) for child in ast.iter_child_nodes(node))


def _optional_category(category: str | None) -> tuple[str | None, dict[str, Any] | None]:
    if category is None:
        return None, None
    if not isinstance(category, str) or not category.strip():
        return None, None
    return _required_category(category)


def _required_category(category: str) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(category, str) or not category.strip():
        return "", _fail("validation_error", "Укажите категорию услуги")
    selected = canonical_category(category)
    if selected is None:
        available = ", ".join(CATEGORIES)
        return "", _fail("validation_error", f"Неизвестная категория. Доступны: {available}")
    return selected, None


def _text(value: object, *, minimum: int, maximum: int, field: str) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(value, str):
        return "", _fail("validation_error", f"{field} должно быть строкой")
    cleaned = " ".join(value.split())
    if len(cleaned) < minimum:
        return "", _fail("validation_error", f"{field} слишком короткое")
    if len(cleaned) > maximum:
        return "", _fail("validation_error", f"{field} слишком длинное")
    return cleaned, None


def _price(value: object) -> tuple[float, dict[str, Any] | None]:
    number, error = _real(value, field="Цена")
    if error is not None:
        return 0, error
    if number < 0 or number > MAX_PRICE:
        return 0, _fail("validation_error", "Цена должна быть от 0 до 100000000 рублей")
    return _money(Decimal(str(number))), None


def _optional_budget(value: object) -> tuple[float | None, dict[str, Any] | None]:
    if value is None:
        return None, None
    number, error = _real(value, field="Бюджет")
    if error is not None:
        return None, error
    if number < 0 or number > MAX_PRICE:
        return None, _fail("validation_error", "Бюджет должен быть от 0 до 100000000 рублей")
    return _money(Decimal(str(number))), None


def _duration(value: object) -> tuple[int, dict[str, Any] | None]:
    number, error = _integral(value, field="Срок")
    if error is not None:
        return 0, error
    if number < 1 or number > MAX_DURATION_DAYS:
        return 0, _fail("validation_error", "Срок должен быть от 1 до 3650 дней")
    return number, None


def _discount(value: object) -> tuple[Decimal, dict[str, Any] | None]:
    number, error = _real(value, field="Скидка")
    if error is not None:
        return Decimal("0"), error
    if number < 0 or number > 100:
        return Decimal("0"), _fail("validation_error", "Скидка должна быть от 0 до 100 процентов")
    return Decimal(str(number)), None


def _service_id(value: object) -> tuple[int, dict[str, Any] | None]:
    number, error = _integral(value, field="Номер услуги")
    if error is not None:
        return 0, error
    if number < 1 or number > 1_000_000_000:
        return 0, _fail("validation_error", "Номер услуги должен быть положительным целым числом")
    return number, None


def _service_ids(values: object) -> tuple[list[int], dict[str, Any] | None]:
    if not isinstance(values, list):
        return [], _fail("validation_error", "Передайте список номеров услуг")
    if not values or len(values) > MAX_SERVICES_IN_ESTIMATE:
        return [], _fail(
            "validation_error",
            f"В расчёте может быть от 1 до {MAX_SERVICES_IN_ESTIMATE} услуг",
        )
    parsed: list[int] = []
    for value in values:
        service_id, error = _service_id(value)
        if error is not None:
            return [], error
        if service_id not in parsed:
            parsed.append(service_id)
    return parsed, None


def _real(value: object, *, field: str) -> tuple[float, dict[str, Any] | None]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0, _fail("validation_error", f"{field} должна быть числом")
    if isinstance(value, float) and not math.isfinite(value):
        return 0, _fail("validation_error", f"{field} должна быть обычным числом")
    return float(value), None


def _integral(value: object, *, field: str) -> tuple[int, dict[str, Any] | None]:
    if isinstance(value, bool):
        return 0, _fail("validation_error", f"{field} должно быть целым числом")
    if isinstance(value, int):
        return value, None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value), None
    return 0, _fail("validation_error", f"{field} должно быть целым числом")


def _money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _fail(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}
