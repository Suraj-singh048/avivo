import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from graph.nodes import get_user_history
from graph.pipeline import get_graph

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "*RAG Bot* — powered by Gemini 2.5 Pro\n\n"
    "Commands:\n"
    "• `/ask <question>` — Ask anything from the knowledge base\n"
    "• `/image` — (Not Supported: implements Option A Mini-RAG)\n"
    "• `/summarize` — Summarize your last conversation\n"
    "• `/help` — Show this message\n\n"
    "_Knowledge base: company policies, tech FAQ, recipes, product guide._"
)

async def help_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)

async def image_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "This bot implements **Option A: Mini-RAG**. The image feature is not supported in this track.",
        parse_mode=ParseMode.MARKDOWN
    )


async def ask_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(ctx.args).strip()
    if not query:
        await update.message.reply_text(
            "Please provide a question.\nUsage: `/ask What is the return policy?`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    user_id = update.effective_user.id
    thinking_msg = await update.message.reply_text("Searching knowledge base...")

    try:
        graph = get_graph()
        result = graph.invoke(
            {
                "query": query,
                "user_id": user_id,
                "history": get_user_history(user_id),
                "retrieved_chunks": [],
                "answer": "",
                "sources": [],
                "cache_hit": False,
            }
        )

        answer = result["answer"]
        sources = result["sources"]
        cache_hit = result["cache_hit"]

        source_line = ""
        if sources:
            formatted = ", ".join(f"`{s}`" for s in sources)
            source_line = f"\n\n*Sources:* {formatted}"

        cache_note = " _(cached)_" if cache_hit else ""
        reply = f"{answer}{source_line}{cache_note}"

        await thinking_msg.delete()
        await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.exception("Error in ask_handler")
        await thinking_msg.delete()
        await update.message.reply_text(
            f"Something went wrong: `{type(e).__name__}`\nPlease try again.",
            parse_mode=ParseMode.MARKDOWN,
        )


async def summarize_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    history = get_user_history(user_id)

    if not history:
        await update.message.reply_text(
            "No conversation history yet. Ask something with `/ask` first.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    lines = []
    for msg in history:
        role = "You" if msg["role"] == "user" else "Bot"
        content = msg["content"][:120] + ("..." if len(msg["content"]) > 120 else "")
        lines.append(f"*{role}:* {content}")

    reply = "*Recent conversation:*\n\n" + "\n\n".join(lines)
    await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
