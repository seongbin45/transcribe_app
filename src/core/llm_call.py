"""제공자별(xai/openai/gemini/claude) 실제 채팅/생성 API 호출.

응답 형식이 제공자마다 달라서(OpenAI 계열은 choices[].message.content, Claude는
content[].text, Gemini는 candidates[].content.parts[].text) 여기서 통일된
`call_llm()` 인터페이스로 감싼다. core/llm_catalog.py(모델 후보 검증)와
core/llm_refine.py(실제 화자 병합 제안)가 같이 쓴다.

xai/openai는 OpenAI 호환 Chat Completions 포맷이라 같은 함수를 공유한다
(Automating_automatic_message_sending/src/aam/catalog.py가 같은 이유로 두 제공자를
`_openai_style_models`로 묶어 처리하는 것과 같은 이유).
"""
from __future__ import annotations

import requests

from .llm_providers import ResolvedProvider

ANTHROPIC_VERSION = "2023-06-01"
GEMINI_GENERATE_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def call_llm(
    provider: ResolvedProvider,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int = 700,
    json_mode: bool = False,
    timeout: float = 120,
) -> str:
    """provider.id에 맞는 API를 호출해서 응답 텍스트(문자열)를 반환한다."""
    if provider.id in ("xai", "openai"):
        return _call_openai_compatible(provider, model, system_prompt, user_content, max_tokens, json_mode, timeout)
    if provider.id == "claude":
        return _call_claude(provider, model, system_prompt, user_content, max_tokens, timeout)
    if provider.id == "gemini":
        return _call_gemini(provider, model, system_prompt, user_content, max_tokens, json_mode, timeout)
    raise ValueError(f"알 수 없는 LLM 제공자입니다: {provider.id}")


def _call_openai_compatible(
    provider: ResolvedProvider,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    json_mode: bool,
    timeout: float,
) -> str:
    base_url = provider.base_url or "https://api.openai.com/v1"
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_claude(
    provider: ResolvedProvider,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    timeout: float,
) -> str:
    base_url = provider.base_url or "https://api.anthropic.com"
    url = f"{base_url.rstrip('/')}/v1/messages"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
    }
    resp = requests.post(
        url,
        headers={
            "x-api-key": provider.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _call_gemini(
    provider: ResolvedProvider,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    json_mode: bool,
    timeout: float,
) -> str:
    url = GEMINI_GENERATE_URL_TMPL.format(model=model)
    # thinkingBudget=0: 일부 Gemini 모델(예: gemini-2.5-flash)이 "생각" 토큰을 먼저 쓰고
    # 남는 걸로 답을 생성하는데, 그러다 maxOutputTokens를 다 써서 JSON이 중간에 잘리는 걸
    # 실제로 겪었다(reasoning 필드 자체를 출력에 요구하므로 내부 thinking은 필요 없음).
    # thinking을 지원하지 않는 구형 모델에서는 이 필드가 무시되거나 무해하게 실패할 수 있음.
    generation_config = {"maxOutputTokens": max_tokens, "thinkingConfig": {"thinkingBudget": 0}}
    if json_mode:
        generation_config["responseMimeType"] = "application/json"
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_content}]}],
        "generationConfig": generation_config,
    }
    resp = requests.post(f"{url}?key={provider.api_key}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
