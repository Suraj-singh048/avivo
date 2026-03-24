import sys
from pathlib import Path

from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from rag.embedder import embed_batch
from rag.vector_store import get_conn, insert_chunk, clear_all, chunk_count


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i : i + chunk_size])
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


def load_docs(docs_dir: Path) -> list[tuple[str, str]]:
    return [
        (path.stem, path.read_text(encoding="utf-8"))
        for path in sorted(docs_dir.glob("*.md"))
    ]


def ingest(force: bool = False) -> None:
    conn = get_conn()

    if chunk_count(conn) > 0 and not force:
        print(f"[ingest] already has {chunk_count(conn)} chunks, skipping")
        return

    clear_all(conn)

    docs = load_docs(DOCS_DIR)
    if not docs:
        print(f"[ingest] no .md files found in {DOCS_DIR}")
        sys.exit(1)

    all_chunks: list[tuple[str, str]] = []
    for source, text in docs:
        chunks = chunk_text(text)
        all_chunks.extend((chunk, source) for chunk in chunks)
        print(f"[ingest] {source}: {len(chunks)} chunks")

    print(f"[ingest] embedding {len(all_chunks)} chunks...")
    embeddings = embed_batch([c[0] for c in all_chunks])

    for (text, source), embedding in zip(all_chunks, embeddings):
        insert_chunk(conn, embedding, text, source)

    print(f"[ingest] done, total: {chunk_count(conn)}")
    conn.close()


if __name__ == "__main__":
    ingest(force="--force" in sys.argv)
