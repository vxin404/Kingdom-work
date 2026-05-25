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


def _build_client(region: str):
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.hunyuan.v20230901 import hunyuan_client

    sid, sk = _load_tencent_cred()
    cred = credential.Credential(sid, sk)
    http_profile = HttpProfile()
    http_profile.endpoint = "hunyuan.tencentcloudapi.com"
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    return hunyuan_client.HunyuanClient(cred, region, client_profile)


def _build_messages(question: str, citations: list[dict], validation_feedback: str = "") -> tuple[str, str]:
    allowed_pages = sorted({int(item["page_no"]) for item in citations})
    allowed_page_text = "、".join(str(x) for x in allowed_pages)
    system_prompt = (
        "你是一个文档问答助手。"
        "你只能依据给定的证据片段回答，不要使用外部知识，不要编造。"
        "如果证据不足，请明确说明“根据当前检索到的片段，暂时无法确定”。"
        f"如果需要引用页码，只能引用这些页码：{allowed_page_text}。"
        "不要把证据顺序、条款编号、表格行号当作页码。"
        "不要补充证据中没有出现的单位、数字、结论或属性。"
        "回答尽量简洁。"
    )
    evidence_lines = []
    for item in citations:
        evidence_lines.append(
            f"来源页码：第{item['page_no']}页\n片段类型：{item['type']}\n证据片段：\n{item['snippet']}"
        )
    user_prompt = (
        f"问题：{question}\n\n"
        "以下是检索得到的证据片段，请仅基于这些证据作答：\n"
        f"{chr(10).join(evidence_lines)}\n\n"
        "请输出一段中文答案，不要输出 JSON，不要重复全部证据原文。"
        "如果引用页码，只能写证据中真实存在的页码。"
        "如果证据没有明确单位，就不要自行补单位。"
    )
    if validation_feedback:
        user_prompt += f"\n\n上一次回答存在以下问题，请严格修正后重新回答：\n{validation_feedback}"
    return system_prompt, user_prompt


def synthesize_answer(
    *,
    question: str,
    citations: list[dict],
    region: str,
    model: str,
    validation_feedback: str = "",
    retry: int = 2,
) -> str:
    from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
    from tencentcloud.hunyuan.v20230901 import models

    if not citations:
        return "当前没有检索到可用证据。"

    system_prompt, user_prompt = _build_messages(question, citations, validation_feedback=validation_feedback)
    client = _build_client(region)

    req = models.ChatCompletionsRequest()
    req.Model = model
    req.Stream = False
    req.Temperature = 0.2
    req.TopP = 0.8
    req.EnableEnhancement = False
    req.SearchInfo = False
    req.Citation = False

    system_msg = models.Message()
    system_msg.Role = "system"
    system_msg.Content = system_prompt
    user_msg = models.Message()
    user_msg.Role = "user"
    user_msg.Content = user_prompt
    req.Messages = [system_msg, user_msg]

    last_err: Optional[Exception] = None
    for attempt in range(retry + 1):
        try:
            resp = client.ChatCompletions(req)
            if resp.ErrorMsg:
                raise RuntimeError(f"chat API returned error: {resp.ErrorMsg}")
            if not resp.Choices:
                raise RuntimeError("chat API returned no choices")
            finish_reason = (resp.Choices[0].FinishReason or "").strip().lower()
            if finish_reason == "sensitive":
                raise RuntimeError("chat API output blocked by moderation")
            if resp.Choices and resp.Choices[0].Message and resp.Choices[0].Message.Content:
                return resp.Choices[0].Message.Content.strip()
            raise RuntimeError("empty chat completion response")
        except TencentCloudSDKException as e:
            last_err = e
            if attempt >= retry:
                break
            time.sleep(0.8 * (2**attempt))
    raise RuntimeError(f"hunyuan chat failed: {last_err}") from last_err
