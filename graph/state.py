from typing import TypedDict


class RAGState(TypedDict):
    query: str
    user_id: int
    history: list[dict]
    retrieved_chunks: list[dict]
    answer: str
    sources: list[str]
    cache_hit: bool
