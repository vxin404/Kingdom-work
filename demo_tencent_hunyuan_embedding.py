import argparse
import json
import os
from pathlib import Path
from typing import Optional


def _load_secret(name: str) -> str:
    return (os.getenv(name) or "").strip()


def load_credentials() -> tuple[str, str]:
    sid = _load_secret("TENCENTCLOUD_SECRET_ID") or _load_secret("TENCENT_SECRET_ID")
    sk = _load_secret("TENCENTCLOUD_SECRET_KEY") or _load_secret("TENCENT_SECRET_KEY")
    return sid, sk


def read_text_arg(text: Optional[str], text_file: Optional[str], max_chars: int) -> str:
    if text is not None:
        s = text
    elif text_file is not None:
        s = Path(text_file).expanduser().read_text(encoding="utf-8")
    else:
        raise ValueError("Provide --text or --text-file")
    s = s.strip()
    if max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars]
        print(f"warning: input truncated to {max_chars} chars")
    return s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--region", type=str, default="")
    args = parser.parse_args()

    secret_id, secret_key = load_credentials()
    if not secret_id or not secret_key:
        print("missing credentials. set one of:")
        print("  export TENCENTCLOUD_SECRET_ID='...'; export TENCENTCLOUD_SECRET_KEY='...'")
        print("  export TENCENT_SECRET_ID='...'; export TENCENT_SECRET_KEY='...'")
        return 2

    try:
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
    except Exception as e:
        print(f"missing tencent hunyuan sdk: {e}")
        print("try: pip install -U tencentcloud-sdk-python-hunyuan")
        return 3

    text = read_text_arg(args.text, args.text_file, args.max_chars)

    cred = credential.Credential(secret_id, secret_key)
    http_profile = HttpProfile()
    http_profile.endpoint = "hunyuan.tencentcloudapi.com"
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    client = hunyuan_client.HunyuanClient(cred, args.region, client_profile)

    req = models.GetEmbeddingRequest()
    req.from_json_string(json.dumps({"Input": text}, ensure_ascii=False))
    resp = client.GetEmbedding(req)
    obj = json.loads(resp.to_json_string())

    dim = None
    vec = None
    if isinstance(obj, dict):
        data = obj.get("Data")
        if isinstance(data, dict):
            dim = data.get("Dim")
            vec = data.get("Embedding")
        if dim is None:
            dim = obj.get("Dim")
        if vec is None:
            vec = obj.get("Embedding")

    if isinstance(vec, list):
        print(f"embedding_dim={dim or len(vec)}")
        print("embedding_preview=", vec[:8])

    print(json.dumps(obj, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
