"""Проверки SQLite-каталога. Используется временный файл, не data/services.db."""

from __future__ import annotations

import sqlite3

import pytest

from mcp_server.db import CATEGORIES, SEED_SERVICES, ServiceRepository


@pytest.fixture
def repo(tmp_path) -> ServiceRepository:
    database = ServiceRepository(tmp_path / "services.db")
    database.initialize()
    return database


def test_database_is_created_and_seeded(repo: ServiceRepository, tmp_path) -> None:
    assert (tmp_path / "services.db").exists()
    services = repo.all_services()
    assert len(services) == 15
    assert len(SEED_SERVICES) == 15
    assert {item["category"] for item in services} == set(CATEGORIES)
    assert all(item["active"] for item in services)
    assert all(item["price"] >= 0 for item in services)
    assert all(item["price_is_demo"] for item in services)

    repo.initialize()
    assert len(repo.all_services()) == 15


def test_find_service_by_partial_name(repo: ServiceRepository) -> None:
    found = [
        item
        for item in repo.all_services()
        if "телеграм" in item["name"].casefold()
    ]
    assert found
    assert found[0]["name"] == "Телеграм-бот для приёма заявок"


def test_find_by_category(repo: ServiceRepository) -> None:
    analytics = [item for item in repo.all_services() if item["category"] == "Аналитика"]
    assert len(analytics) == 2
    assert {item["name"] for item in analytics} == {
        "Дашборд загрузки команды",
        "Отчёт по воронке продаж",
    }


def test_get_service_by_id(repo: ServiceRepository) -> None:
    service = repo.get_by_id(1)
    assert service is not None
    assert service["id"] == 1
    assert service["name"] == SEED_SERVICES[0]["name"]
    assert repo.get_by_id(9999) is None


def test_add_service(repo: ServiceRepository) -> None:
    created = repo.add_service(
        name="Учебный бот для записи на демо",
        category="AI-чат-боты",
        description="Короткое учебное описание новой услуги для проверки каталога.",
        price=50000,
        duration_days=7,
    )
    assert created["id"] == 16
    assert created["price"] == 50000
    assert len(repo.all_services()) == 16


def test_reject_negative_price_in_schema(repo: ServiceRepository) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        with repo.connection() as conn:
            conn.execute(
                """
                INSERT INTO services (name, category, description, price, duration_days)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("Плохая цена", "Аналитика", "Описание", -5, 3),
            )
    assert len(repo.all_services()) == 15
