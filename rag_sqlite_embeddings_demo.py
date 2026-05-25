import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    page_no: int
    type: str
    content: str
    snippet: str
    source_path: str
    meta_json: str


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_secret(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _load_tencent_cred() -> tuple[str, str]:
    sid = _load_secret("TENCENTCLOUD_SECRET_ID") or _load_secret("TENCENT_SECRET_ID")
    sk = _load_secret("TENCENTCLOUD_SECRET_KEY") or _load_secret("TENCENT_SECRET_KEY")
    return sid, sk


def _ensure_db(db_path: Path) -> sqlite3.Connection:
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
            content_hash TEXT NOT NULL UNIQUE
        );
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
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks(doc_id, page_no);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(type);")
    conn.commit()
    return conn


def _iter_page_text_files(ocr_dir: Path) -> Iterable[tuple[int, Path]]:
    text_dir = ocr_dir / "text"
    for p in sorted(text_dir.glob("page_*.txt")):
        m = re.search(r"page_(\d+)\.txt$", p.name)
        if not m:
            continue
        yield int(m.group(1)), p


def _iter_table_row_files(ocr_dir: Path) -> Iterable[tuple[int, Path]]:
    rows_dir = ocr_dir / "tables_rows"
    if not rows_dir.exists():
        return []
    for p in sorted(rows_dir.glob("page_*.jsonl")):
        m = re.search(r"page_(\d+)\.jsonl$", p.name)
        if not m:
            continue
        yield int(m.group(1)), p


def _chunk_text_by_lines(
    text: str,
    *,
    max_chars: int,
    overlap: int,
) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        chunks.append("\n".join(buf).strip())
        if overlap <= 0:
            buf = []
            buf_len = 0
            return
        carry: list[str] = []
        carry_len = 0
        for ln in reversed(buf):
            if carry_len + len(ln) + 1 > overlap:
                break
            carry.append(ln)
            carry_len += len(ln) + 1
        carry.reverse()
        buf = carry
        buf_len = sum(len(x) + 1 for x in buf)

    for ln in lines:
        add_len = len(ln) + (1 if buf else 0)
        if buf and buf_len + add_len > max_chars:
            flush()
        buf.append(ln)
        buf_len += add_len
    flush()
    return chunks


def load_chunks_from_ocr_dir(
    ocr_dir: Path,
    *,
    doc_id: str,
    text_chunk_chars: int,
    text_overlap_chars: int,
    snippet_chars: int,
) -> list[Chunk]:
    out: list[Chunk] = []

    for page_no, p in _iter_page_text_files(ocr_dir):
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        for i, part in enumerate(
            _chunk_text_by_lines(text, max_chars=text_chunk_chars, overlap=text_overlap_chars),
            start=1,
        ):
            snippet = part[:snippet_chars].strip()
            meta = {"chunk_no": i}
            out.append(
                Chunk(
                    doc_id=doc_id,
                    page_no=page_no,
                    type="text",
                    content=part,
                    snippet=snippet,
                    source_path=p.as_posix(),
                    meta_json=json.dumps(meta, ensure_ascii=False),
                )
            )

    for page_no, p in _iter_table_row_files(ocr_dir):
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            content = (obj.get("row_text") or "").strip()
            if not content:
                continue
            snippet = content[:snippet_chars].strip()
            meta = {
                "row_name": obj.get("row_name"),
                "header": obj.get("header"),
                "row_index": obj.get("row_index"),
            }
            out.append(
                Chunk(
                    doc_id=doc_id,
                    page_no=page_no,
                    type="table_row",
                    content=content,
                    snippet=snippet,
                    source_path=p.as_posix(),
                    meta_json=json.dumps(meta, ensure_ascii=False),
                )
            )

    return out


def upsert_chunks(conn: sqlite3.Connection, chunks: list[Chunk], *, embedding_model: str) -> int:
    inserted = 0
    for ch in chunks:
        content_hash = _sha256_hex(embedding_model + "\n" + ch.content)
        try:
            conn.execute(
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
                    content_hash,
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    return inserted


def _extract_embedding_from_hunyuan_response(obj: dict) -> list[float]:
    if not isinstance(obj, dict):
        raise ValueError(f"unexpected embedding response type: {type(obj)}")

    data = obj.get("Data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            v = first.get("Embedding")
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                return [float(x) for x in v]

    if isinstance(data, dict):
        for k in ("Embedding", "embedding", "Vector", "vector"):
            v = data.get(k)
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                return [float(x) for x in v]

    for k in ("Embedding", "embedding", "Vector", "vector"):
        v = obj.get(k)
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            return [float(x) for x in v]

    raise ValueError(f"unexpected embedding response keys: {list(obj.keys())}")


def hunyuan_get_embedding(text: str, *, region: str, retry: int = 3) -> list[float]:
    secret_id, secret_key = _load_tencent_cred()
    if not secret_id or not secret_key:
        raise RuntimeError("missing credentials: set TENCENTCLOUD_SECRET_ID/TENCENTCLOUD_SECRET_KEY")

    from tencentcloud.common import credential
    from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.hunyuan.v20230901 import hunyuan_client, models

    cred = credential.Credential(secret_id, secret_key)
    http_profile = HttpProfile()
    http_profile.endpoint = "hunyuan.tencentcloudapi.com"
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    client = hunyuan_client.HunyuanClient(cred, region, client_profile)

    req = models.GetEmbeddingRequest()
    req.from_json_string(json.dumps({"Input": text}, ensure_ascii=False))

    last_err: Optional[Exception] = None
    for attempt in range(retry + 1):
        try:
            resp = client.GetEmbedding(req)
            obj = json.loads(resp.to_json_string())
            return _extract_embedding_from_hunyuan_response(obj)
        except TencentCloudSDKException as e:
            last_err = e
            if attempt >= retry:
                break
            time.sleep(0.8 * (2**attempt))
    raise RuntimeError(f"hunyuan embedding failed: {last_err}") from last_err


def _vec_to_blob(vec: list[float]) -> tuple[bytes, int, float]:
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return arr.tobytes(), int(arr.shape[0]), n


def embed_missing(conn: sqlite3.Connection, *, embedding_model: str, region: str, max_chars: int) -> int:
    cur = conn.execute(
        """
        SELECT c.chunk_id, c.content
        FROM chunks c
        LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id AND e.model = ?
        WHERE e.chunk_id IS NULL
        ORDER BY c.chunk_id
        """,
        (embedding_model,),
    )
    rows = cur.fetchall()
    done = 0
    for chunk_id, content in rows:
        text = (content or "").strip()
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
        vec = hunyuan_get_embedding(text, region=region)
        blob, dim, nrm = _vec_to_blob(vec)
        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings(chunk_id, model, dim, vec, norm)
            VALUES (?,?,?,?,?)
            """,
            (int(chunk_id), embedding_model, dim, blob, nrm),
        )
        done += 1
        if done % 10 == 0:
            conn.commit()
    conn.commit()
    return done


def search(conn: sqlite3.Connection, *, query: str, doc_id: str, embedding_model: str, region: str, topk: int):
    q_vec = hunyuan_get_embedding(query, region=region)
    q_arr = np.asarray(q_vec, dtype=np.float32)
    q_norm = float(np.linalg.norm(q_arr)) + 1e-8

    cur = conn.execute(
        """
        SELECT c.chunk_id, c.page_no, c.type, c.snippet, e.vec, e.norm
        FROM chunks c
        JOIN embeddings e ON e.chunk_id = c.chunk_id AND e.model = ?
        WHERE c.doc_id = ?
        """,
        (embedding_model, doc_id),
    )
    scored = []
    for chunk_id, page_no, typ, snippet, vec_blob, v_norm in cur:
        v = np.frombuffer(vec_blob, dtype=np.float32)
        denom = (float(v_norm) + 1e-8) * q_norm
        s = float(np.dot(q_arr, v) / denom)
        scored.append((s, int(chunk_id), int(page_no), typ, snippet))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topk]


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--db", type=str, default="rag.sqlite")
    p_build.add_argument("--ocr-dir", type=str, default="artifacts/tencent_ocr_final")
    p_build.add_argument("--doc-id", type=str, default="gbt1568")
    p_build.add_argument("--embedding-model", type=str, default="hunyuan-embedding")
    p_build.add_argument("--region", type=str, default="")
    p_build.add_argument("--text-chunk-chars", type=int, default=800)
    p_build.add_argument("--text-overlap-chars", type=int, default=120)
    p_build.add_argument("--snippet-chars", type=int, default=240)
    p_build.add_argument("--embed-max-chars", type=int, default=1500)

    p_search = sub.add_parser("search")
    p_search.add_argument("--db", type=str, default="rag.sqlite")
    p_search.add_argument("--doc-id", type=str, default="gbt1568")
    p_search.add_argument("--embedding-model", type=str, default="hunyuan-embedding")
    p_search.add_argument("--region", type=str, default="")
    p_search.add_argument("--topk", type=int, default=5)
    p_search.add_argument("--query", type=str, required=True)

    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    conn = _ensure_db(db_path)

    if args.cmd == "build":
        ocr_dir = Path(args.ocr_dir).expanduser().resolve()
        chunks = load_chunks_from_ocr_dir(
            ocr_dir,
            doc_id=args.doc_id,
            text_chunk_chars=args.text_chunk_chars,
            text_overlap_chars=args.text_overlap_chars,
            snippet_chars=args.snippet_chars,
        )
        ins = upsert_chunks(conn, chunks, embedding_model=args.embedding_model)
        emb = embed_missing(
            conn,
            embedding_model=args.embedding_model,
            region=args.region,
            max_chars=args.embed_max_chars,
        )
        print(
            json.dumps(
                {
                    "db": db_path.as_posix(),
                    "ocr_dir": ocr_dir.as_posix(),
                    "doc_id": args.doc_id,
                    "chunks_loaded": len(chunks),
                    "chunks_inserted": ins,
                    "embeddings_created": emb,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.cmd == "search":
        results = search(
            conn,
            query=args.query,
            doc_id=args.doc_id,
            embedding_model=args.embedding_model,
            region=args.region,
            topk=args.topk,
        )
        out = []
        for score, chunk_id, page_no, typ, snippet in results:
            out.append(
                {
                    "score": round(score, 6),
                    "chunk_id": chunk_id,
                    "page_no": page_no,
                    "type": typ,
                    "snippet": snippet,
                }
            )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
