import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from app.services.chunking import Chunk



def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()



def ensure_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path.as_posix())
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            page_no INTEGER NOT NULL,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            snippet TEXT NOT NULL,
            source_path TEXT NOT NULL,
            meta_json TEXT NOT NULL,
            content_hash TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vec BLOB NOT NULL,
            norm REAL NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks(doc_id, page_no)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model)")
    conn.commit()
    return conn



def clear_doc(conn: sqlite3.Connection, doc_id: str) -> None:
    rows = conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
    chunk_ids = [int(r[0]) for r in rows]
    if chunk_ids:
        marks = ",".join(["?"] * len(chunk_ids))
        conn.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({marks})", chunk_ids)
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.commit()



def insert_chunks(conn: sqlite3.Connection, chunks: list[Chunk], *, embedding_model: str) -> list[dict]:
    inserted: list[dict] = []
    for ch in chunks:
        cur = conn.execute(
            """
            INSERT INTO chunks(doc_id,page_no,type,content,snippet,source_path,meta_json,content_hash)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                ch.doc_id,
                ch.page_no,
                ch.type,
                ch.content,
                ch.snippet,
                ch.source_path,
                ch.meta_json,
                _sha256_hex(embedding_model + "\n" + ch.content),
            ),
        )
        inserted.append({
            "chunk_id": int(cur.lastrowid),
            "page_no": ch.page_no,
            "type": ch.type,
            "content": ch.content,
            "snippet": ch.snippet,
        })
    conn.commit()
    return inserted



def save_embedding(conn: sqlite3.Connection, *, chunk_id: int, model: str, vector: list[float]) -> None:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    conn.execute(
        "INSERT OR REPLACE INTO embeddings(chunk_id,model,dim,vec,norm) VALUES (?,?,?,?,?)",
        (chunk_id, model, int(arr.shape[0]), arr.tobytes(), norm),
    )



def search_topk(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    model: str,
    query_vector: list[float],
    topk: int,
) -> list[dict]:
    q = np.asarray(query_vector, dtype=np.float32)
    q_norm = float(np.linalg.norm(q)) + 1e-8
    cur = conn.execute(
        """
        SELECT c.chunk_id, c.page_no, c.type, c.snippet, c.meta_json, e.vec, e.norm
        FROM chunks c
        JOIN embeddings e ON e.chunk_id = c.chunk_id AND e.model = ?
        WHERE c.doc_id = ?
        """,
        (model, doc_id),
    )
    scored = []
    for chunk_id, page_no, typ, snippet, meta_json, vec_blob, v_norm in cur:
        v = np.frombuffer(vec_blob, dtype=np.float32)
        score = float(np.dot(q, v) / ((float(v_norm) + 1e-8) * q_norm))
        scored.append(
            {
                "score": score,
                "chunk_id": int(chunk_id),
                "page_no": int(page_no),
                "type": typ,
                "snippet": snippet,
                "meta": json.loads(meta_json or "{}"),
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:topk]
