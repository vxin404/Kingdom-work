import shutil
import re
from pathlib import Path

from app.config import Settings
from app.services.chat_tencent import synthesize_answer
from app.services.chunking import load_chunks_from_ocr_dir
from app.services.embedding_tencent import get_embedding
from app.services.ocr_tencent import load_ocr_config_from_env, process_images_to_artifacts, reset_workdir
from app.services.pdf_render import render_pdf_to_jpgs
from app.storage.sqlite_store import clear_doc, ensure_db, insert_chunks, save_embedding, search_topk



def process_pdf(pdf_path: Path, settings: Settings) -> dict:
    reset_workdir(settings.current_doc_dir)
    original_pdf = settings.current_doc_dir / "original.pdf"
    shutil.copyfile(pdf_path, original_pdf)

    images_dir = settings.current_doc_dir / "images"
    image_paths = render_pdf_to_jpgs(
        original_pdf,
        images_dir,
        dpi=settings.render_dpi,
        jpg_quality=settings.render_jpg_quality,
    )

    ocr_cfg = load_ocr_config_from_env(region=settings.ocr_region)
    ocr_summary = process_images_to_artifacts(
        image_paths,
        settings.current_doc_dir,
        cfg=ocr_cfg,
        table_mode=settings.table_mode,
        table_policy=settings.table_policy,
    )

    chunks = load_chunks_from_ocr_dir(
        settings.current_doc_dir,
        doc_id=settings.current_doc_id,
        text_chunk_chars=settings.text_chunk_chars,
        text_overlap_chars=settings.text_overlap_chars,
        snippet_chars=settings.snippet_chars,
    )
    conn = ensure_db(settings.db_path)
    clear_doc(conn, settings.current_doc_id)
    inserted = insert_chunks(conn, chunks, embedding_model=settings.embedding_model)
    for item in inserted:
        vec = get_embedding(
            item["content"],
            region=settings.hunyuan_region,
            max_chars=settings.embed_max_chars,
        )
        save_embedding(conn, chunk_id=item["chunk_id"], model=settings.embedding_model, vector=vec)
    conn.commit()

    return {
        "doc_id": settings.current_doc_id,
        "pages": len(image_paths),
        "chunks": len(inserted),
        "final_ocr_path": ocr_summary["final_ocr_path"],
    }


def _extract_query_keywords(question: str) -> list[str]:
    q = question.strip()
    q = re.sub(r"(请问|一下|一下子|帮我|告诉我)", "", q)
    q = re.sub(r"(是多少|是什么|是多少呢|多少|多大|数值|值|吗|呢|\?|？)", " ", q)
    kws = re.findall(r"[\u4e00-\u9fffA-Za-z0-9\-]{2,}", q)
    seen = set()
    out = []
    for kw in kws:
        if kw not in seen:
            out.append(kw)
            seen.add(kw)
    return out


def _rerank_citations(question: str, citations: list[dict]) -> list[dict]:
    keywords = _extract_query_keywords(question)
    is_value_question = bool(re.search(r"(多少|多大|是多少|数值|值)", question))
    unit_pattern = re.compile(r"\d+(?:\.\d+)?\s*(MPa|mm|cm|kg|%|级|AQL)?", re.I)

    reranked = []
    for item in citations:
        text = item["snippet"]
        bonus = 0.0
        for kw in keywords:
            if kw in text:
                bonus += 0.18
                if re.search(re.escape(kw) + r".{0,20}?\d", text, re.I | re.S):
                    bonus += 0.2
        if is_value_question and unit_pattern.search(text):
            bonus += 0.08
        if is_value_question and "试验" in text and "应大于等于" not in text:
            bonus -= 0.04
        item = dict(item)
        item["_hybrid_score"] = item["score"] + bonus
        reranked.append(item)

    reranked.sort(key=lambda x: x["_hybrid_score"], reverse=True)
    for item in reranked:
        item.pop("_hybrid_score", None)
    return reranked


def ask_question(question: str, settings: Settings, *, topk: int) -> dict:

    conn = ensure_db(settings.db_path)
    total = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (settings.current_doc_id,)
    ).fetchone()[0]
    if not total:
        raise RuntimeError("no indexed document found, please upload a PDF first")

    query_vec = get_embedding(question, region=settings.hunyuan_region, max_chars=settings.embed_max_chars)
    citations = search_topk(
        conn,
        doc_id=settings.current_doc_id,
        model=settings.embedding_model,
        query_vector=query_vec,
        topk=max(topk * 3, 10),
    )
    citations = _rerank_citations(question, citations)[:topk]
    answer_lines = [f"Top {len(citations)} related snippets:"]
    for i, item in enumerate(citations, start=1):
        answer_lines.append(f"{i}. [page {item['page_no']}] {item['snippet']}")
    fallback_answer = "\n".join(answer_lines)
    try:
        answer = synthesize_answer(
            question=question,
            citations=[
                {
                    "page_no": x["page_no"],
                    "type": x["type"],
                    "score": round(x["score"], 6),
                    "snippet": x["snippet"],
                }
                for x in citations
            ],
            region=settings.hunyuan_region,
            model=settings.chat_model,
        )
    except Exception:
        answer = fallback_answer
    return {
        "answer": answer,
        "citations": [
            {
                "page_no": x["page_no"],
                "type": x["type"],
                "score": round(x["score"], 6),
                "snippet": x["snippet"],
            }
            for x in citations
        ],
        "refused": False,
        "refuse_reason": "",
    }
