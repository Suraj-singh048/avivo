import logging
import sys

from telegram.ext import ApplicationBuilder, CommandHandler

from config import TELEGRAM_TOKEN
from bot.handlers import ask_handler, image_handler, help_handler, summarize_handler
from rag.ingest import ingest

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN is not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    logger.info("Checking knowledge base...")
    ingest(force=False)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("ask",       ask_handler))
    app.add_handler(CommandHandler("image",     image_handler))
    app.add_handler(CommandHandler("help",      help_handler))
    app.add_handler(CommandHandler("start",     help_handler))
    app.add_handler(CommandHandler("summarize", summarize_handler))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
