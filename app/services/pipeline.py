import shutil
import re
import json
from pathlib import Path

from app.config import Settings
from app.services.chat_tencent import synthesize_answer
from app.services.chunking import load_chunks_from_ocr_dir
from app.services.embedding_tencent import get_embedding
from app.services.ocr_tencent import load_ocr_config_from_env, process_images_to_artifacts, reset_workdir
from app.services.pdf_classifier import classify_pdf, log_pdf_classification
from app.services.pdf_render import render_pdf_to_jpgs
from app.storage.sqlite_store import clear_doc, ensure_db, insert_chunks, save_embedding, search_topk



def process_pdf(pdf_path: Path, settings: Settings) -> dict:
    reset_workdir(settings.current_doc_dir)
    original_pdf = settings.current_doc_dir / "original.pdf"
    shutil.copyfile(pdf_path, original_pdf)
    pdf_profile = classify_pdf(original_pdf)
    log_pdf_classification(pdf_profile)
    (settings.current_doc_dir / "pdf_classification.json").write_text(
        json.dumps(pdf_profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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
        "pdf_profile": pdf_profile,
        "pages": len(image_paths),
        "chunks": len(inserted),
        "final_ocr_path": ocr_summary["final_ocr_path"],
    }


def _extract_query_keywords(question: str) -> list[str]:
    q = question.strip()
    q = re.sub(r"(请问|一下|一下子|帮我|告诉我|这个标准|文档里|文中|表1里|对)", "", q)
    q = re.sub(r"(是多少|是什么|是多少呢|多少|多大|数值|值|吗|呢|有没有|有无|有什么要求|要求|分别|规定|各类型|类型|\?|？)", " ", q)
    kws = re.findall(r"[\u4e00-\u9fffA-Za-z0-9\-]{2,}", q)
    seen = set()
    out = []
    for kw in kws:
        if kw not in seen:
            out.append(kw)
            seen.add(kw)
        if "的" in kw:
            merged = kw.replace("的", "")
            if len(merged) >= 2 and merged not in seen:
                out.append(merged)
                seen.add(merged)
            for part in kw.split("的"):
                if len(part) >= 2 and part not in seen:
                    out.append(part)
                    seen.add(part)
    return out


def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("Mpa", "MPa").replace("mPa", "MPa").replace("mpa", "MPa")
    text = re.sub(r"\s+", "", text)
    return text


def _extract_numeric_facts(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", _normalize_text(text), re.I))


def _extract_unit_facts(text: str) -> set[str]:
    normalized = _normalize_text(text)
    normalized = normalized.replace("MPA", "MPa").replace("MPa", "MPa")
    return set(re.findall(r"\d+(?:\.\d+)?(?:MPa|mm|cm|kg|%|级|AQL)", normalized, re.I))


def _extract_answer_terms(text: str) -> list[str]:
    normalized = _normalize_text(text)
    raw_terms = re.findall(r"[\u4e00-\u9fffA-Za-z0-9\-]{2,12}", normalized)
    stopwords = {
        "根据当前检索到的片段",
        "根据当前文档检索结果",
        "暂时无法确定",
        "根据现有证据",
        "该标准",
        "见第",
        "证据",
    }
    terms = []
    seen = set()
    for term in raw_terms:
        if term in stopwords:
            continue
        if term not in seen:
            terms.append(term)
            seen.add(term)
    return terms[:12]


def _extract_page_refs(text: str) -> list[int]:
    return sorted({int(x) for x in re.findall(r"第\s*(\d+)\s*页", text or "")})


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


def _build_self_check(question: str, answer: str, citations: list[dict]) -> dict:
    question_keywords = _extract_query_keywords(question)
    top1_score = round(citations[0]["score"], 6) if citations else 0.0
    top3_avg_score = round(sum(x["score"] for x in citations[:3]) / max(len(citations[:3]), 1), 6) if citations else 0.0
    combined_text = "\n".join(x["snippet"] for x in citations)
    citation_pages = sorted({int(x["page_no"]) for x in citations})
    is_value_question = bool(re.search(r"(多少|多大|是多少|数值|值)", question))
    is_yes_no_question = bool(re.search(r"(是否|有没有|有无|是否有|有没有规定|规定没有)", question))
    number_pattern = re.compile(r"\d+(?:\.\d+)?\s*(MPa|mm|cm|kg|%|级|AQL)?", re.I)
    evidence_has_number = bool(number_pattern.search(combined_text))
    matched_keywords = [kw for kw in question_keywords if kw in combined_text][:5]
    keyword_coverage = round(len(matched_keywords) / max(len(question_keywords), 1), 4)
    evidence_has_conclusion = bool(re.search(r"(应|不允许|允许|未提及|没有规定|未规定|应大于等于|可对|符合)", combined_text))
    answer_for_grounding = re.sub(r"见第\d+页(?:证据[\d、,，\s]+)?", "", answer)
    answer_for_grounding = re.sub(r"证据[\d、,，\s]+", "", answer_for_grounding)
    answer_page_refs = _extract_page_refs(answer)
    invalid_page_refs = [x for x in answer_page_refs if x not in citation_pages]
    answer_numbers = _extract_numeric_facts(answer_for_grounding)
    evidence_numbers = _extract_numeric_facts(combined_text)
    missing_answer_numbers = sorted(x for x in (answer_numbers - evidence_numbers) if x not in {"1", "2", "3", "4", "5"})
    answer_unit_facts = _extract_unit_facts(answer_for_grounding)
    evidence_unit_facts = _extract_unit_facts(combined_text)
    unsupported_unit_facts = sorted(answer_unit_facts - evidence_unit_facts)
    answer_terms = _extract_answer_terms(answer_for_grounding)
    evidence_normalized = _normalize_text(combined_text)
    grounded_terms = [term for term in answer_terms if term in evidence_normalized]
    answer_term_overlap = round(len(grounded_terms) / max(len(answer_terms), 1), 4) if answer_terms else 1.0

    no_relevant_citation = not citations
    retrieval_weak = (
        top1_score < 0.22
        or (top3_avg_score < 0.24 and keyword_coverage < 0.3)
        or (bool(question_keywords) and keyword_coverage == 0.0 and top1_score < 0.68)
    )
    insufficient_evidence = (
        (is_value_question and not evidence_has_number)
        or (is_yes_no_question and not evidence_has_conclusion)
    )
    answer_not_grounded = bool(invalid_page_refs) or bool(unsupported_unit_facts) or bool(missing_answer_numbers) or (
        bool(answer_terms) and answer_term_overlap < 0.25 and keyword_coverage < 0.3
    )

    triggered_reasons = []
    if no_relevant_citation:
        triggered_reasons.append("no_relevant_citation")
    if retrieval_weak:
        triggered_reasons.append("low_retrieval_score")
    if insufficient_evidence:
        triggered_reasons.append("insufficient_evidence")
    if answer_not_grounded:
        triggered_reasons.append("answer_not_grounded")

    refused = bool(triggered_reasons)
    refuse_reason = triggered_reasons[0] if triggered_reasons else ""
    needs_retry = bool(invalid_page_refs or unsupported_unit_facts or missing_answer_numbers)

    feedback_lines = []
    if invalid_page_refs:
        feedback_lines.append(
            f"答案出现了证据中不存在的页码：{','.join(f'第{x}页' for x in invalid_page_refs)}；可用页码只有：{','.join(f'第{x}页' for x in citation_pages)}。"
        )
    if unsupported_unit_facts:
        feedback_lines.append(
            f"答案出现了证据中不存在的数字+单位表达：{', '.join(unsupported_unit_facts)}。"
        )
    if missing_answer_numbers:
        feedback_lines.append(
            f"答案出现了证据中不存在的数字：{', '.join(missing_answer_numbers)}。"
        )
    if not feedback_lines and answer_not_grounded:
        feedback_lines.append("答案包含证据中无法直接支撑的结论或术语，请只保留证据原文可以支持的内容。")

    return {
        "top1_score": top1_score,
        "top3_avg_score": top3_avg_score,
        "matched_keywords": matched_keywords,
        "keyword_coverage": keyword_coverage,
        "citation_pages": citation_pages,
        "answer_page_refs": answer_page_refs,
        "invalid_page_refs": invalid_page_refs,
        "no_relevant_citation": no_relevant_citation,
        "retrieval_weak": retrieval_weak,
        "insufficient_evidence": insufficient_evidence,
        "answer_not_grounded": answer_not_grounded,
        "missing_answer_numbers": missing_answer_numbers,
        "unsupported_unit_facts": unsupported_unit_facts,
        "answer_term_overlap": answer_term_overlap,
        "needs_retry": needs_retry,
        "validation_feedback": " ".join(feedback_lines),
        "triggered_reasons": triggered_reasons,
        "refused": refused,
        "refuse_reason": refuse_reason,
    }


def _build_refusal_answer(question: str, reason: str) -> str:
    if reason == "no_relevant_citation":
        return f"根据当前检索结果，暂时没有找到与问题“{question}”直接相关的有效证据，因此无法给出可靠答案。"
    if reason == "low_retrieval_score":
        return f"根据当前检索到的证据，问题“{question}”的相关片段命中较弱，暂时无法给出可靠答案。"
    if reason == "insufficient_evidence":
        return f"当前检索结果与问题“{question}”存在一定相关性，但证据不足以支持明确结论，因此暂时无法确定。"
    if reason == "answer_not_grounded":
        return f"当前生成答案无法被现有证据充分支撑，因此对问题“{question}”暂时不输出确定性结论。"
    return "根据当前检索到的证据，暂时无法给出可靠答案。"


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
    model_citations = [
        {
            "page_no": x["page_no"],
            "type": x["type"],
            "score": round(x["score"], 6),
            "snippet": x["snippet"],
        }
        for x in citations
    ]
    answer_lines = [f"Top {len(citations)} related snippets:"]
    for i, item in enumerate(citations, start=1):
        answer_lines.append(f"{i}. [page {item['page_no']}] {item['snippet']}")
    fallback_answer = "\n".join(answer_lines)
    used_fallback = False
    try:
        answer = synthesize_answer(
            question=question,
            citations=model_citations,
            region=settings.hunyuan_region,
            model=settings.chat_model,
        )
    except Exception:
        answer = fallback_answer
        used_fallback = True
    self_check = _build_self_check(question, answer, citations)
    if not used_fallback and self_check["needs_retry"]:
        try:
            retried_answer = synthesize_answer(
                question=question,
                citations=model_citations,
                region=settings.hunyuan_region,
                model=settings.chat_model,
                validation_feedback=self_check["validation_feedback"],
            )
            retried_check = _build_self_check(question, retried_answer, citations)
            if len(retried_check["triggered_reasons"]) <= len(self_check["triggered_reasons"]):
                answer = retried_answer
                self_check = retried_check
        except Exception:
            pass
    if self_check["refused"]:
        answer = _build_refusal_answer(question, self_check["refuse_reason"])
    return {
        "answer": answer,
        "citations": model_citations,
        "self_check": self_check,
        "refused": self_check["refused"],
        "refuse_reason": self_check["refuse_reason"],
    }
