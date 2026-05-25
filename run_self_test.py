import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def post_multipart(url: str, field_name: str, file_path: Path) -> dict:
    boundary = "----TraeBoundary" + uuid.uuid4().hex
    file_bytes = file_path.read_bytes()
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'.encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    req = urllib.request.Request(url, data=b"".join(parts), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def evaluate_answer(item: dict, result: dict) -> dict:
    answer = result.get("answer", "")
    citations = result.get("citations") or []
    top1_page = citations[0]["page_no"] if citations else 0
    must_contain = item.get("must_contain") or []
    expected_page = int(item.get("expected_page") or 0)
    expect_refused = bool(item.get("expect_refused"))
    refused = bool(result.get("refused"))

    keywords_pass = all(token in answer for token in must_contain) if must_contain else True
    page_pass = expected_page == 0 or top1_page == expected_page
    refusal_pass = refused == expect_refused

    return {
        "question": item["question"],
        "top1_page": top1_page,
        "keywords_pass": keywords_pass,
        "page_pass": page_pass,
        "refusal_pass": refusal_pass,
        "refused": refused,
        "answer_preview": answer[:120],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--questions", default="eval_questions.json")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    pdf_path = Path(args.pdf).expanduser().resolve()
    questions_path = Path(args.questions).expanduser().resolve()

    if not pdf_path.exists():
        raise SystemExit(f"pdf not found: {pdf_path}")
    if not questions_path.exists():
        raise SystemExit(f"questions file not found: {questions_path}")

    try:
        upload_result = post_multipart(f"{base_url}/api/upload", "file", pdf_path)
    except urllib.error.URLError as e:
        raise SystemExit(f"upload failed: {e}") from e

    print("=== Upload Result ===")
    print(json.dumps(upload_result, ensure_ascii=False, indent=2))

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    summaries = []
    for item in questions:
        result = post_json(
            f"{base_url}/api/ask",
            {"question": item["question"], "topk": 5},
        )
        summary = evaluate_answer(item, result)
        summaries.append(summary)
        print("\n=== Question ===")
        print(item["question"])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("=== Check ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    total = len(summaries)
    passed = sum(1 for x in summaries if x["keywords_pass"] and x["page_pass"] and x["refusal_pass"])
    print("\n=== Summary ===")
    print(json.dumps({"total": total, "passed": passed, "failed": total - passed}, ensure_ascii=False, indent=2))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
