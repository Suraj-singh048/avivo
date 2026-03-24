"""Tests for the rag/ embedding and vector store layer."""

import os
import sys
import tempfile
import json
import pytest

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Override DB path to a temp file for tests
import config
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
config.DB_PATH = _tmp.name
_tmp.close()

from rag.embedder import embed_text, embed_batch
from rag.vector_store import get_conn, insert_chunk, search, clear_all, chunk_count


# ── Embedding tests ────────────────────────────────────────────────────────────

def test_embed_text_returns_list():
    result = embed_text("What is the return policy?")
    assert isinstance(result, list)
    assert len(result) == 384  # all-MiniLM-L6-v2 dim


def test_embed_text_normalized():
    result = embed_text("test sentence")
    magnitude = sum(x**2 for x in result) ** 0.5
    assert abs(magnitude - 1.0) < 1e-3  # should be unit-norm


def test_embed_batch():
    texts = ["hello world", "how are you", "return policy"]
    results = embed_batch(texts)
    assert len(results) == 3
    assert all(len(r) == 384 for r in results)


# ── Vector store tests ────────────────────────────────────────────────────────

@pytest.fixture
def db_conn():
    conn = get_conn()
    clear_all(conn)
    yield conn
    conn.close()


def test_insert_and_count(db_conn):
    emb = embed_text("return policy for avivo products")
    insert_chunk(db_conn, emb, "Customers may return within 30 days.", "company_policy")
    assert chunk_count(db_conn) == 1


def test_search_returns_correct_source(db_conn):
    # Insert two chunks from different sources
    emb1 = embed_text("return policy refund")
    insert_chunk(db_conn, emb1, "Return within 30 days for a refund.", "company_policy")

    emb2 = embed_text("API rate limits requests per minute")
    insert_chunk(db_conn, emb2, "Free tier: 100 req/min. Pro: 1000 req/min.", "tech_faq")

    # Query for return policy → should rank company_policy first
    q_emb = embed_text("How many days do I have to return a product?")
    results = search(db_conn, q_emb, top_k=2)

    assert len(results) >= 1
    assert results[0]["source"] == "company_policy"


def test_search_top_k_limit(db_conn):
    for i in range(5):
        emb = embed_text(f"sample text number {i}")
        insert_chunk(db_conn, emb, f"text {i}", "test_doc")

    results = search(db_conn, embed_text("sample"), top_k=3)
    assert len(results) <= 3


def test_clear_all(db_conn):
    emb = embed_text("temporary chunk")
    insert_chunk(db_conn, emb, "some text", "test")
    assert chunk_count(db_conn) >= 1
    clear_all(db_conn)
    assert chunk_count(db_conn) == 0
