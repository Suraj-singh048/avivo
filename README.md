# Avivo RAG Telegram Bot

A lightweight Telegram bot that answers questions from a local knowledge base using a **Mini-RAG pipeline** powered by **Gemini 2.5 Pro** and **LangGraph**.

---

## System Design

```
User (Telegram)
      │ /ask <query>
      ▼
┌─────────────────────────────────────┐
│        python-telegram-bot v22       │
│  (async: /ask  /help  /summarize)   │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│         LangGraph StateGraph         │
│                                      │
│  [cache] ──hit──► END                │
│     │ miss                           │
│  [retrieve] ─► sqlite-vec KNN        │
│     │                                │
│  [generate] ─► Gemini 2.5 Pro        │
│     │          (google-genai SDK)     │
│  [format]  ─► history + cache save   │
└─────────────────────────────────────┘
                │
                ▼
        Reply to user
```

---

## Tech Stack

| Component      | Library                    | Version    |
|---------------|----------------------------|------------|
| LLM            | `google-genai` (Gemini 2.5 Pro) | ≥1.68.0 |
| Orchestration  | `langgraph`                | ≥1.1.3     |
| LLM Chains     | `langchain`                | ≥1.2.13    |
| Telegram Bot   | `python-telegram-bot`      | ≥22.0      |
| Embeddings     | `sentence-transformers`    | all-MiniLM-L6-v2 (384-dim) |
| Vector Store   | `sqlite-vec`               | local SQLite, no server needed |
| Runtime        | Python                     | 3.11+      |

---

## Project Structure

```
avivo/
├── app.py                  # Entry point: python app.py
├── config.py               # All env vars and constants
├── requirements.txt
├── .env                    # TELEGRAM_TOKEN + GEMINI_API_KEY (create from .env.example)
│
├── bot/
│   ├── handlers.py         # /ask /help /summarize handlers
│   └── bot_runner.py       # ApplicationBuilder + polling
│
├── rag/
│   ├── embedder.py         # sentence-transformers wrapper
│   ├── vector_store.py     # sqlite-vec insert + KNN search
│   └── ingest.py           # chunk → embed → store pipeline
│
├── graph/
│   ├── state.py            # RAGState TypedDict
│   ├── nodes.py            # 4 node functions
│   └── pipeline.py         # StateGraph wiring
│
├── docs/                   # Knowledge base (.md files)
│   ├── company_policy.md
│   ├── tech_faq.md
│   ├── recipes.md
│   └── product_guide.md
│
├── data/rag.db             # sqlite-vec DB (auto-created)
├── cache/query_cache.json  # Query→answer cache (auto-created)
└── tests/                  # pytest unit + integration tests
```

---

## How to Run Locally

### 1. Clone & set up environment

```bash
cd avivo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in TELEGRAM_TOKEN and GEMINI_API_KEY
```

- Get a Telegram token from [@BotFather](https://t.me/BotFather)
- Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 3. Ingest the knowledge base (first run only)

```bash
python -m rag.ingest
```

### 4. Run the bot

```bash
python app.py
```

Open Telegram, find your bot, and try:
- `/help` — show available commands
- `/ask What is the return policy?`
- `/ask How do I reset my API key?`
- `/summarize` — see your last conversation

---

## Bot Commands

| Command | Description |
|---|---|
| `/ask <question>` | Query the knowledge base; returns answer + source doc(s) |
| `/image` | Not supported (This bot implements Option A: Mini-RAG) |
| `/summarize` | Show last 3 messages from your conversation |
| `/help` | Show usage instructions |
| `/start` | Same as `/help` |

---

## Features

- **Mini-RAG**: Retrieves top-3 relevant chunks from 4 indexed documents
- **Gemini 2.5 Pro**: Generates concise, grounded answers via `google-genai` SDK
- **LangGraph**: Stateful pipeline with conditional cache short-circuit
- **Query cache**: Repeated identical queries answered instantly (file-based JSON)
- **Message history**: Last 3 exchanges per user injected as conversation context
- **Source citations**: Every answer shows which knowledge doc was used
- **No external servers**: All storage is local SQLite via `sqlite-vec`

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Embedding shape and normalization
- sqlite-vec insert, KNN search, clear
- Cache hit/miss node logic
- Full pipeline with mocked Gemini client

---

## Models & APIs Used

| Model | Why |
|---|---|
| `gemini-2.5-pro` | Best reasoning quality via official `google-genai` SDK |
| `all-MiniLM-L6-v2` | Small (80 MB), fast, 384-dim, strong semantic similarity |

Local embeddings mean **no embedding API cost** and **full offline capability** (except the Gemini call).
