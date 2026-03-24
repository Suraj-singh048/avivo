"""Tests for the LangGraph RAG pipeline (no real Gemini call)."""

import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
config.DB_PATH = _tmp.name
_tmp.close()

from graph.state import RAGState
from graph.nodes import check_cache, retrieve_chunks, format_response, _query_hash


# ── Cache node ────────────────────────────────────────────────────────────────

def test_cache_miss():
    state: RAGState = {
        "query": "unique_uncached_query_xyz_12345",
        "user_id": 1,
        "history": [],
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "cache_hit": False,
    }
    # Patch cache to return empty
    with patch("graph.nodes._load_cache", return_value={}):
        result = check_cache(state)
    assert result["cache_hit"] is False


def test_cache_hit():
    query = "what is the return policy"
    cached_answer = "30 days."
    cache = {_query_hash(query): {"answer": cached_answer, "sources": ["company_policy"]}}

    state: RAGState = {
        "query": query,
        "user_id": 1,
        "history": [],
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "cache_hit": False,
    }
    with patch("graph.nodes._load_cache", return_value=cache):
        result = check_cache(state)
    assert result["cache_hit"] is True
    assert result["answer"] == cached_answer
    assert "company_policy" in result["sources"]


# ── Retrieve node ─────────────────────────────────────────────────────────────

def test_retrieve_chunks_populates_state():
    from rag.vector_store import get_conn, insert_chunk, clear_all
    from rag.embedder import embed_text

    conn = get_conn()
    clear_all(conn)
    emb = embed_text("return policy refund 30 days")
    insert_chunk(conn, emb, "Customers can return within 30 days.", "company_policy")
    conn.close()

    # Patch _get_db to use fresh conn
    with patch("graph.nodes._get_db", return_value=get_conn()):
        state: RAGState = {
            "query": "What is the return policy?",
            "user_id": 1,
            "history": [],
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
            "cache_hit": False,
        }
        result = retrieve_chunks(state)

    assert len(result["retrieved_chunks"]) >= 1
    assert "text" in result["retrieved_chunks"][0]
    assert "source" in result["retrieved_chunks"][0]


# ── Format node ───────────────────────────────────────────────────────────────

def test_format_response_sets_sources():
    state: RAGState = {
        "query": "test",
        "user_id": 42,
        "history": [],
        "retrieved_chunks": [
            {"text": "some text", "source": "tech_faq", "distance": 0.1},
            {"text": "other text", "source": "tech_faq", "distance": 0.2},
            {"text": "policy text", "source": "company_policy", "distance": 0.3},
        ],
        "answer": "Great answer.",
        "sources": [],
        "cache_hit": False,
    }
    with patch("graph.nodes._save_cache"), patch("graph.nodes._load_cache", return_value={}):
        result = format_response(state)

    assert set(result["sources"]) == {"tech_faq", "company_policy"}


# ── Pipeline integration (mocked LLM) ────────────────────────────────────────

def test_full_pipeline_mock_llm():
    from graph.pipeline import build_graph

    mock_response = MagicMock()
    mock_response.text = "The return policy is 30 days."

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("graph.nodes._get_gemini", return_value=mock_client), \
         patch("graph.nodes._load_cache", return_value={}), \
         patch("graph.nodes._save_cache"):

        graph = build_graph()
        result = graph.invoke({
            "query": "What is the return policy?",
            "user_id": 99,
            "history": [],
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
            "cache_hit": False,
        })

    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
