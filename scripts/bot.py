"""Telegram interface for the community knowledge-base assistant."""

import asyncio
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from query_kb import (
    MODEL_NAME,
    answer_question,
    load_index,
)


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

INDEX_DIR = os.getenv("KB_INDEX_DIR", "kb_index")


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if update.message is None:
        return

    await update.message.reply_text(
        "سلام! من ربات دانش گروه هستم. "
        "می‌تونی درباره اقامت، بیمه، بانک، خونه "
        "و موضوعات مشابه ازم سؤال بپرسی."
    )


def should_answer(message, bot_username: str) -> bool:
    if message.chat.type == "private":
        return True

    is_mentioned = (
        f"@{bot_username}" in message.text
    )

    replied_message = message.reply_to_message
    is_reply_to_bot = (
        replied_message
        and replied_message.from_user
        and replied_message.from_user.username
        == bot_username
    )

    return bool(is_mentioned or is_reply_to_bot)


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.message
    if message is None or not message.text:
        return

    bot_username = context.bot.username
    if not bot_username:
        return

    if not should_answer(message, bot_username):
        return

    question = message.text.replace(
        f"@{bot_username}",
        "",
    ).strip()

    if not question:
        return

    await context.bot.send_chat_action(
        chat_id=message.chat_id,
        action="typing",
    )

    bot_data = context.application.bot_data

    try:
        reply, _ = await asyncio.to_thread(
            answer_question,
            bot_data["openai_client"],
            bot_data["embedding_model"],
            bot_data["index"],
            bot_data["entries"],
            question,
        )
    except Exception:
        log.exception("Could not answer the question")
        reply = (
            "یه خطایی پیش اومد. "
            "لطفاً دوباره امتحان کن."
        )

    await message.reply_text(reply)


def main():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not bot_token or not openai_key:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN and OPENAI_API_KEY "
            "must be configured."
        )

    log.info(
        "Loading embedding model and index from %s",
        INDEX_DIR,
    )

    embedding_model = SentenceTransformer(MODEL_NAME)
    index, entries = load_index(INDEX_DIR)
    openai_client = OpenAI(api_key=openai_key)

    application = (
        Application.builder()
        .token(bot_token)
        .build()
    )

    application.bot_data.update(
        {
            "embedding_model": embedding_model,
            "index": index,
            "entries": entries,
            "openai_client": openai_client,
        }
    )

    application.add_handler(
        CommandHandler(
            "start",
            start_command,
        )
    )
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    log.info("Bot ready with %d entries", len(entries))
    application.run_polling()


if __name__ == "__main__":
    main()
