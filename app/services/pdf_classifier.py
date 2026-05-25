import json
from pathlib import Path

import fitz


def classify_pdf(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    try:
        page_count = doc.page_count
        text_chars_per_page: list[int] = []
        image_pages = 0

        for page_index in range(page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            text_chars = len("".join(text.split()))
            text_chars_per_page.append(text_chars)
            if page.get_images(full=True):
                image_pages += 1

        avg_text_chars = (sum(text_chars_per_page) / page_count) if page_count else 0.0
        max_text_chars = max(text_chars_per_page) if text_chars_per_page else 0
        image_page_ratio = (image_pages / page_count) if page_count else 0.0

        if avg_text_chars < 30 and image_page_ratio >= 0.8:
            pdf_type = "scan_pdf"
            reason = "text layer is weak and most pages contain images"
        elif avg_text_chars >= 80 and image_page_ratio < 0.5:
            pdf_type = "text_pdf"
            reason = "text layer is strong and image ratio is low"
        else:
            pdf_type = "mixed_pdf"
            reason = "text layer and image features are mixed"

        return {
            "pdf_type": pdf_type,
            "reason": reason,
            "page_count": page_count,
            "avg_text_chars": round(avg_text_chars, 2),
            "max_text_chars": max_text_chars,
            "image_pages": image_pages,
            "image_page_ratio": round(image_page_ratio, 4),
            "current_strategy": "still_use_ocr_pipeline",
        }
    finally:
        doc.close()


def log_pdf_classification(result: dict) -> None:
    print("[pdf_classifier]", json.dumps(result, ensure_ascii=False))
