"""Central configuration for the Avivo RAG Telegram Bot."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
DB_PATH = str(DATA_DIR / "rag.db")
CACHE_PATH = str(CACHE_DIR / "query_cache.json")

# ── API Keys ─────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Model Config ──────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-pro"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# ── RAG Config ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 300          # characters per chunk
CHUNK_OVERLAP = 50        # character overlap between chunks
TOP_K = 3                 # number of chunks to retrieve
MAX_HISTORY = 3           # messages per user to keep
MAX_CACHE_ENTRIES = 100   # max query → answer cache entries
