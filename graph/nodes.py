import hashlib
import json
from pathlib import Path

import google.genai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL, CACHE_PATH, MAX_CACHE_ENTRIES, TOP_K
from rag.embedder import embed_text
from rag.vector_store import get_conn, search
from graph.state import RAGState

_gemini_client: genai.Client | None = None
_db_conn = None
_user_histories: dict[int, list[dict]] = {}


def _get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _get_db():
    global _db_conn
    if _db_conn is None:
        _db_conn = get_conn()
    return _db_conn


def _load_cache() -> dict:
    path = Path(CACHE_PATH)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    if len(cache) > MAX_CACHE_ENTRIES:
        keys = list(cache.keys())
        for k in keys[: len(cache) - MAX_CACHE_ENTRIES]:
            del cache[k]
    Path(CACHE_PATH).write_text(json.dumps(cache, indent=2))


def _query_hash(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def get_user_history(user_id: int) -> list[dict]:
    return _user_histories.get(user_id, [])


def _update_history(user_id: int, query: str, answer: str, max_msgs: int = 3) -> None:
    hist = _user_histories.get(user_id, [])
    hist.append({"role": "user", "content": query})
    hist.append({"role": "assistant", "content": answer})
    _user_histories[user_id] = hist[-(max_msgs * 2):]


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no previous conversation)"
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def check_cache(state: RAGState) -> RAGState:
    cache = _load_cache()
    key = _query_hash(state["query"])
    if key in cache:
        cached = cache[key]
        state["answer"] = cached["answer"]
        state["sources"] = cached["sources"]
        state["cache_hit"] = True
        print(f"[cache] hit: {state['query'][:60]}")
    else:
        state["cache_hit"] = False
    return state


def retrieve_chunks(state: RAGState) -> RAGState:
    query_emb = embed_text(state["query"])
    chunks = search(_get_db(), query_emb, top_k=TOP_K)
    state["retrieved_chunks"] = chunks
    print(f"[retrieve] {len(chunks)} chunks for: {state['query'][:60]}")
    return state


def generate_answer(state: RAGState) -> RAGState:
    chunks = state["retrieved_chunks"]
    if not chunks:
        state["answer"] = (
            "I could not find relevant information in the knowledge base. "
            "Try rephrasing your question."
        )
        state["sources"] = []
        return state

    context = "\n\n".join(
        f"[{i}] (from {c['source']})\n{c['text']}" for i, c in enumerate(chunks, 1)
    )
    history_str = _format_history(state["history"])

    prompt = f"""You are a helpful assistant. Answer ONLY using the context below.
If the answer is not in the context, say so honestly. Be concise.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_str}

QUESTION:
{state["query"]}

ANSWER:"""

    response = _get_gemini().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    state["answer"] = response.text.strip()
    return state


def format_response(state: RAGState) -> RAGState:
    sources = list({c["source"] for c in state.get("retrieved_chunks", [])})
    state["sources"] = sources

    _update_history(state["user_id"], state["query"], state["answer"])

    if state["answer"] and not state["cache_hit"]:
        cache = _load_cache()
        key = _query_hash(state["query"])
        cache[key] = {"answer": state["answer"], "sources": sources}
        _save_cache(cache)
        print(f"[cache] saved: {state['query'][:60]}")

    return state
