"""Telegram-бот AVAHRUS на aiogram 3."""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from openai import AsyncOpenAI

from telegram_bot.ai_client import Consultant, split_text
from telegram_bot.config import Settings
from telegram_bot.mcp_client import MCPClient

logger = logging.getLogger("telegram_bot")

START_TEXT = (
    "Здравствуйте! Я AI-консультант AVAHRUS.\n\n"
    "Помогу подобрать демонстрационные услуги по автоматизации бизнеса: "
    "чат-боты, ассистенты, документы, интеграции и аналитика.\n\n"
    "Цены в каталоге учебные и не являются офертой. Заказ через бота не оформляется.\n\n"
    "Команды: /catalog, /help, /reset.\n"
    "Напишите задачу своими словами, например: "
    "«Нужен телеграм-бот для заявок, бюджет до 100000»."
)

HELP_TEXT = (
    "Я могу:\n"
    "• показать каталог услуг командой /catalog;\n"
    "• найти услугу по названию, категории или номеру;\n"
    "• подобрать варианты по описанию задачи — поиск идёт по ключевым словам, не по embeddings;\n"
    "• посчитать предварительную стоимость и скидку;\n"
    "• выполнить арифметический расчёт.\n\n"
    "Команда /reset очищает историю диалога. Бот помнит последние 10 сообщений каждого пользователя.\n\n"
    "Добавление услуги доступно только администратору. Права проверяет MCP-сервер, "
    "а не текст сообщения.\n"
    "Цены демонстрационные. Бот не подтверждает и не оформляет заказ."
)


def build_router(consultant: Consultant) -> Router:
    router = Router()

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer(START_TEXT)

    @router.message(Command("help"))
    async def help_command(message: Message) -> None:
        await message.answer(HELP_TEXT)

    @router.message(Command("catalog"))
    async def catalog(message: Message) -> None:
        user_id = _user_id(message)
        if user_id is None:
            return
        await _send_chat_action(message)
        text = await consultant.catalog(user_id)
        await _answer_parts(message, text)

    @router.message(Command("reset"))
    async def reset(message: Message) -> None:
        user_id = _user_id(message)
        if user_id is None:
            return
        await consultant.reset(user_id)
        await message.answer("История диалога очищена.")

    @router.message(F.text)
    async def dialog(message: Message) -> None:
        user_id = _user_id(message)
        if user_id is None or message.text is None:
            return
        if message.text.startswith("/"):
            await message.answer("Такой команды нет. Список возможностей: /help.")
            return
        logger.info("message from telegram_user_id=%s", user_id)
        await _send_chat_action(message)
        reply = await consultant.answer(user_id, message.text)
        await _answer_parts(message, reply)

    return router


async def run_bot(settings: Settings) -> None:
    mcp_client = MCPClient(settings.mcp_server_url, settings.mcp_auth_token)
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    consultant = Consultant(settings, mcp_client, openai_client)
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(consultant))
    try:
        async with Bot(token=settings.telegram_token) as bot:
            logger.info("Telegram-бот запущен, модель %s", settings.openai_model)
            await dispatcher.start_polling(bot)
    finally:
        await openai_client.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = Settings.load()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    asyncio.run(run_bot(settings))


def _user_id(message: Message) -> int | None:
    if message.from_user is None:
        return None
    return message.from_user.id


async def _send_chat_action(message: Message) -> None:
    try:
        await message.bot.send_chat_action(message.chat.id, "typing")
    except Exception:
        logger.debug("Не удалось отправить статус набора", exc_info=True)


async def _answer_parts(message: Message, text: str) -> None:
    for part in split_text(text):
        await message.answer(part)


if __name__ == "__main__":
    main()
