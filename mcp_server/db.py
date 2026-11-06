"""SQLite-каталог демонстрационных услуг AVAHRUS."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

CATEGORIES: tuple[str, ...] = (
    "AI-чат-боты",
    "Автоматизация процессов",
    "AI-ассистенты",
    "Работа с документами",
    "Интеграции",
    "Аналитика",
)

SEED_SERVICES: tuple[dict[str, Any], ...] = (
    {
        "name": "Телеграм-бот для приёма заявок",
        "category": "AI-чат-боты",
        "description": (
            "Бот принимает заявки в Telegram, задаёт уточняющие вопросы и передаёт "
            "диалог менеджеру. Подходит для первичной линии клиентских обращений."
        ),
        "price": 89000,
        "duration_days": 14,
    },
    {
        "name": "Чат-бот записи клиентов",
        "category": "AI-чат-боты",
        "description": (
            "Бот показывает свободные окна, записывает клиента и отправляет напоминание. "
            "Сокращает ручную переписку администратора."
        ),
        "price": 75000,
        "duration_days": 10,
    },
    {
        "name": "Бот внутренней поддержки",
        "category": "AI-чат-боты",
        "description": (
            "Сотрудники задают боту вопросы по регламентам и получают ответ со ссылкой "
            "на инструкцию. Сложный вопрос можно передать живому специалисту."
        ),
        "price": 98000,
        "duration_days": 16,
    },
    {
        "name": "Автоматизация обработки заявок",
        "category": "Автоматизация процессов",
        "description": (
            "Заявки из почты и мессенджеров попадают в единый маршрут: регистрация, "
            "назначение ответственного и контроль срока."
        ),
        "price": 120000,
        "duration_days": 21,
    },
    {
        "name": "Маршрут согласования документов",
        "category": "Автоматизация процессов",
        "description": (
            "Договор или счёт проходит заданную цепочку согласования. Инициатор видит "
            "статус без ручных напоминаний коллегам."
        ),
        "price": 110000,
        "duration_days": 18,
    },
    {
        "name": "Автоматизация онбординга сотрудников",
        "category": "Автоматизация процессов",
        "description": (
            "Новому сотруднику автоматически ставятся задачи, заявки на доступы и "
            "учебные материалы по чек-листу первого месяца."
        ),
        "price": 95000,
        "duration_days": 15,
    },
    {
        "name": "AI-ассистент менеджера продаж",
        "category": "AI-ассистенты",
        "description": (
            "Ассистент готовит краткое досье клиента перед звонком и черновик следующего "
            "шага по истории переписки."
        ),
        "price": 130000,
        "duration_days": 20,
    },
    {
        "name": "AI-ассистент руководителя",
        "category": "AI-ассистенты",
        "description": (
            "Ассистент собирает сводку по задачам команды и выделяет просроченные "
            "договорённости к планёрке."
        ),
        "price": 150000,
        "duration_days": 25,
    },
    {
        "name": "Ассистент по базе знаний",
        "category": "AI-ассистенты",
        "description": (
            "Сотрудник спрашивает базу знаний обычным языком и получает ответ с "
            "указанием документа-источника."
        ),
        "price": 105000,
        "duration_days": 14,
    },
    {
        "name": "Разбор входящих договоров",
        "category": "Работа с документами",
        "description": (
            "Сервис извлекает стороны, сроки и сумму из договора и показывает фрагменты, "
            "которые стоит проверить юристу."
        ),
        "price": 115000,
        "duration_days": 18,
    },
    {
        "name": "Черновики коммерческих предложений",
        "category": "Работа с документами",
        "description": (
            "По брифу клиента собирается черновик коммерческого предложения в структуре "
            "компании. Менеджер проверяет текст перед отправкой."
        ),
        "price": 85000,
        "duration_days": 12,
    },
    {
        "name": "Интеграция с CRM",
        "category": "Интеграции",
        "description": (
            "Заявки, статусы и комментарии синхронизируются между чат-ботом и CRM, "
            "чтобы менеджер не переносил данные вручную."
        ),
        "price": 140000,
        "duration_days": 21,
    },
    {
        "name": "Интеграция с 1С и мессенджерами",
        "category": "Интеграции",
        "description": (
            "Статусы счетов и заказов из 1С можно запросить или отправить клиенту в "
            "мессенджер без отдельной ручной выгрузки."
        ),
        "price": 160000,
        "duration_days": 28,
    },
    {
        "name": "Дашборд загрузки команды",
        "category": "Аналитика",
        "description": (
            "Дашборд показывает входящие заявки, текущую загрузку и узкие места по "
            "сотрудникам за выбранный период."
        ),
        "price": 90000,
        "duration_days": 12,
    },
    {
        "name": "Отчёт по воронке продаж",
        "category": "Аналитика",
        "description": (
            "Отчёт собирает этапы воронки, конверсию и зависшие сделки. Цифры берутся "
            "из подключённых источников, а не вводятся вручную."
        ),
        "price": 100000,
        "duration_days": 14,
    },
)


class DatabaseError(Exception):
    """Ошибка каталога с кодом, который можно отдать клиенту."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def default_db_path() -> Path:
    """Путь к рабочей базе. Тесты передают свой путь и этот файл не трогают."""
    configured = os.getenv("DATABASE_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "data" / "services.db"


def canonical_category(value: str) -> str | None:
    normalized = value.strip().casefold()
    for category in CATEGORIES:
        if category.casefold() == normalized:
            return category
    return None


class ServiceRepository:
    """Короткоживущие соединения SQLite. Каталог небольшой, фильтры по русскому тексту считаются в Python."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def connection(self) -> "_Connection":
        return _Connection(self.path)

    def initialize(self) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS services (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    price REAL NOT NULL CHECK (price >= 0),
                    duration_days INTEGER NOT NULL CHECK (duration_days > 0),
                    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_services_category ON services(category)"
            )
            count = conn.execute("SELECT COUNT(*) AS count FROM services").fetchone()["count"]
            if count == 0:
                conn.executemany(
                    """
                    INSERT INTO services (name, category, description, price, duration_days, active)
                    VALUES (:name, :category, :description, :price, :duration_days, 1)
                    """,
                    SEED_SERVICES,
                )

    def all_services(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, name, category, description, price, duration_days, active
                FROM services
                ORDER BY id
                """
            ).fetchall()
        return [_row_to_service(row) for row in rows]

    def get_by_id(self, service_id: int) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT id, name, category, description, price, duration_days, active
                FROM services
                WHERE id = ?
                """,
                (service_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_service(row)

    def add_service(
        self,
        *,
        name: str,
        category: str,
        description: str,
        price: float,
        duration_days: int,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            existing = conn.execute("SELECT name FROM services").fetchall()
            if any(row["name"].casefold() == name.casefold() for row in existing):
                raise DatabaseError("conflict", "Услуга с таким названием уже есть в каталоге")
            cursor = conn.execute(
                """
                INSERT INTO services (name, category, description, price, duration_days, active)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (name, category, description, price, duration_days),
            )
            service_id = cursor.lastrowid
        created = self.get_by_id(int(service_id))
        if created is None:
            raise DatabaseError("internal_error", "Услуга не сохранилась")
        return created


class _Connection:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        self._conn = conn
        return conn

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        conn = self._conn
        if conn is None:
            return
        try:
            if exc_type is None:
                conn.commit()
            else:
                conn.rollback()
        finally:
            conn.close()
            self._conn = None


def _row_to_service(row: sqlite3.Row) -> dict[str, Any]:
    price = float(row["price"])
    return {
        "id": int(row["id"]),
        "name": row["name"],
        "category": row["category"],
        "description": row["description"],
        "price": int(price) if price.is_integer() else round(price, 2),
        "duration_days": int(row["duration_days"]),
        "active": bool(row["active"]),
        "currency": "RUB",
        "price_is_demo": True,
    }
