import json
import os
import time
from typing import Optional



def _load_secret(name: str) -> str:
    return (os.getenv(name) or "").strip()



def _load_tencent_cred() -> tuple[str, str]:
    sid = _load_secret("TENCENTCLOUD_SECRET_ID") or _load_secret("TENCENT_SECRET_ID")
    sk = _load_secret("TENCENTCLOUD_SECRET_KEY") or _load_secret("TENCENT_SECRET_KEY")
    if not sid or not sk:
        raise RuntimeError("missing Hunyuan credentials in environment")
    return sid, sk



def _extract_embedding_from_response(obj: dict) -> list[float]:
    if not isinstance(obj, dict):
        raise ValueError(f"unexpected embedding response type: {type(obj)}")
    if obj.get("ErrorMsg"):
        raise RuntimeError(f"embedding API returned error: {obj.get('ErrorMsg')}")
    data = obj.get("Data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        emb = data[0].get("Embedding")
        if isinstance(emb, list):
            return [float(x) for x in emb]
    if isinstance(data, dict):
        emb = data.get("Embedding") or data.get("Vector")
        if isinstance(emb, list):
            return [float(x) for x in emb]
    emb = obj.get("Embedding") or obj.get("Vector")
    if isinstance(emb, list):
        return [float(x) for x in emb]
    raise ValueError(f"unexpected embedding response keys: {list(obj.keys())}")



def get_embedding(text: str, *, region: str, max_chars: int, retry: int = 3) -> list[float]:
    sid, sk = _load_tencent_cred()
    from tencentcloud.common import credential
    from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.hunyuan.v20230901 import hunyuan_client, models

    payload = (text or "").strip()
    if not payload:
        raise ValueError("embedding input text is empty")
    if max_chars > 0 and len(payload) > max_chars:
        payload = payload[:max_chars]

    cred = credential.Credential(sid, sk)
    http_profile = HttpProfile()
    http_profile.endpoint = "hunyuan.tencentcloudapi.com"
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    client = hunyuan_client.HunyuanClient(cred, region, client_profile)

    req = models.GetEmbeddingRequest()
    req.from_json_string(json.dumps({"Input": payload}, ensure_ascii=False))

    last_err: Optional[Exception] = None
    for attempt in range(retry + 1):
        try:
            resp = client.GetEmbedding(req)
            obj = json.loads(resp.to_json_string())
            vec = _extract_embedding_from_response(obj)
            if not vec:
                raise RuntimeError("embedding API returned empty vector")
            return vec
        except TencentCloudSDKException as e:
            last_err = e
            if attempt >= retry:
                break
            time.sleep(0.8 * (2**attempt))
    raise RuntimeError(f"hunyuan embedding failed: {last_err}") from last_err
