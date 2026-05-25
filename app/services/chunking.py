import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    page_no: int
    type: str
    content: str
    snippet: str
    source_path: str
    meta_json: str



def _chunk_text_by_lines(text: str, *, max_chars: int, overlap: int) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    current_clause = ""

    def extract_clause_id(line: str) -> str:
        m = re.match(r"^(\d+(?:\.\d+)*)(?![-\d])\s*(.+)$", line.strip())
        if not m:
            return ""
        clause_id = m.group(1)
        rest = (m.group(2) or "").strip()
        if not rest:
            return ""
        if rest.startswith(":"):
            return ""
        if re.fullmatch(r"[\d.:%~\-]+", rest):
            return ""
        return clause_id

    def maybe_tag(line: str) -> str:
        nonlocal current_clause
        clause_id = extract_clause_id(line)
        if clause_id:
            current_clause = clause_id
        if current_clause and not line.startswith("【"):
            return f"【{current_clause}】 {line}"
        return line

    def flush() -> None:
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
            extra = len(ln) + (1 if carry else 0)
            if carry_len + extra > overlap:
                break
            carry.append(ln)
            carry_len += extra
        carry.reverse()
        buf = carry
        buf_len = sum(len(x) + 1 for x in buf)

    for raw_line in lines:
        clause_id = extract_clause_id(raw_line)
        if clause_id and buf:
            flush()
        line = maybe_tag(raw_line)
        extra = len(line) + (1 if buf else 0)
        if buf and buf_len + extra > max_chars:
            flush()
        buf.append(line)
        buf_len += extra
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
    chunks: list[Chunk] = []

    text_dir = ocr_dir / "text"
    for p in sorted(text_dir.glob("page_*.txt")):
        m = re.search(r"page_(\d+)\.txt$", p.name)
        if not m:
            continue
        page_no = int(m.group(1))
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        for i, part in enumerate(
            _chunk_text_by_lines(text, max_chars=text_chunk_chars, overlap=text_overlap_chars), start=1
        ):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    page_no=page_no,
                    type="text",
                    content=part,
                    snippet=part[:snippet_chars].strip(),
                    source_path=p.as_posix(),
                    meta_json=json.dumps({"chunk_no": i}, ensure_ascii=False),
                )
            )

    rows_dir = ocr_dir / "tables_rows"
    if rows_dir.exists():
        for p in sorted(rows_dir.glob("page_*.jsonl")):
            m = re.search(r"page_(\d+)\.jsonl$", p.name)
            if not m:
                continue
            page_no = int(m.group(1))
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                content = (obj.get("row_text") or "").strip()
                if not content:
                    continue
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        page_no=page_no,
                        type="table_row",
                        content=content,
                        snippet=content[:snippet_chars].strip(),
                        source_path=p.as_posix(),
                        meta_json=json.dumps(
                            {
                                "row_name": obj.get("row_name"),
                                "header": obj.get("header"),
                                "row_index": obj.get("row_index"),
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
    return chunks
