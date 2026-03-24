import json
import sqlite3

import sqlite_vec

from config import DB_PATH, EMBED_DIM, TOP_K


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0(
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding   FLOAT[{EMBED_DIM}],
            +text       TEXT NOT NULL,
            +source     TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def insert_chunk(conn: sqlite3.Connection, embedding: list[float], text: str, source: str) -> None:
    conn.execute(
        "INSERT INTO chunks(embedding, text, source) VALUES (?, ?, ?)",
        [json.dumps(embedding), text, source],
    )
    conn.commit()


def clear_all(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunks")
    conn.commit()


def search(conn: sqlite3.Connection, query_embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    rows = conn.execute(
        """
        SELECT text, source, distance
        FROM chunks
        WHERE embedding MATCH ?
          AND k = ?
        ORDER BY distance
        """,
        [json.dumps(query_embedding), top_k],
    ).fetchall()
    return [{"text": r["text"], "source": r["source"], "distance": r["distance"]} for r in rows]


def chunk_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT count(*) as cnt FROM chunks").fetchone()
    return row["cnt"] if row else 0
