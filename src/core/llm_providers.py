"""LLM 제공자(xai/openai/gemini/claude) 레지스트리와 API 키 해석.

`C:\\...\\Automating_automatic_message_sending\\src\\aam\\settings.py`의 설계를 그대로 가져왔다
(.env의 환경변수 이름 규칙 포함) — 그 프로젝트가 이미 검증해둔 "여러 LLM 제공자를 하나의
.env로 관리하는" 패턴을 재사용해서, 이 앱도 같은 .env에 있는 XAI_API_KEY / OPENAI_API_KEY /
GEMINI_API_KEY(GOOGLE_API_KEY) / ANTHROPIC_API_KEY를 그대로 인식한다.

이 앱만의 추가 사항: "gemini_free"(GEMINI_FREE_KEY)는 이 레지스트리에 없는 이 앱 전용 슬롯이다
(무료 티어라 레이트리밋이 잦아서 재시도를 훨씬 많이 하는 별도 후보로 취급 — core/llm_refine.py 참고).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from . import secrets as _secrets  # noqa: F401  (import 시점에 .env가 os.environ에 로드됨)

PROVIDER_IDS = ("xai", "openai", "gemini", "claude")
_ALIASES = {"anthropic": "claude", "google": "gemini"}


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    env_keys: tuple[str, ...]
    """여러 개가 설정되어 있으면 첫 번째가 우선 (gemini는 예외 — resolve_api_key 참고)."""

    model_env: str
    base_url_env: str | None = None
    default_base_url: str | None = None
    gemini_google_wins: bool = False


PROVIDERS: dict[str, ProviderSpec] = {
    "xai": ProviderSpec(
        id="xai",
        env_keys=("XAI_API_KEY",),
        model_env="XAI_MODEL",
        base_url_env="XAI_BASE_URL",
        default_base_url="https://api.x.ai/v1",
    ),
    "openai": ProviderSpec(
        id="openai",
        env_keys=("OPENAI_API_KEY",),
        model_env="OPENAI_MODEL",
        base_url_env="OPENAI_BASE_URL",
        default_base_url="https://api.openai.com/v1",
    ),
    "gemini": ProviderSpec(
        id="gemini",
        env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        model_env="GEMINI_MODEL",
        gemini_google_wins=True,
    ),
    "claude": ProviderSpec(
        id="claude",
        env_keys=("ANTHROPIC_API_KEY",),
        model_env="ANTHROPIC_MODEL",
        default_base_url="https://api.anthropic.com",
    ),
}


@dataclass(frozen=True)
class ResolvedProvider:
    id: str
    model: str | None
    api_key: str
    base_url: str | None
    key_env: str

    def masked_key(self) -> str:
        return mask_secret(self.api_key)


class SettingsError(ValueError):
    pass


def mask_secret(value: str) -> str:
    """로그에 키를 그대로 남기지 않기 위함 (실수로 API 키를 콘솔에 그대로 출력한 적이 있어서 추가)."""
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}…{value[-4:]}"


def normalize_provider(name: str) -> str:
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in PROVIDERS:
        known = ", ".join(PROVIDER_IDS)
        raise SettingsError(f"알 수 없는 LLM 제공자입니다: {name!r}. 사용 가능: {known}")
    return key


def _clean(value: str | None) -> str:
    return (value or "").strip()


def resolve_api_key(spec: ProviderSpec, environ: Mapping[str, str]) -> tuple[str, str] | None:
    if spec.gemini_google_wins:
        google = _clean(environ.get("GOOGLE_API_KEY"))
        gemini = _clean(environ.get("GEMINI_API_KEY"))
        if google:
            return "GOOGLE_API_KEY", google
        if gemini:
            return "GEMINI_API_KEY", gemini
        return None
    for env_name in spec.env_keys:
        value = _clean(environ.get(env_name))
        if value:
            return env_name, value
    return None


def resolve_provider(name: str, environ: Mapping[str, str] | None = None) -> ResolvedProvider | None:
    env = environ if environ is not None else os.environ
    spec = PROVIDERS[normalize_provider(name)]
    found = resolve_api_key(spec, env)
    if found is None:
        return None
    key_env, api_key = found
    model = _clean(env.get(spec.model_env)) or None
    base_url = spec.default_base_url
    if spec.base_url_env:
        base_url = _clean(env.get(spec.base_url_env)) or spec.default_base_url
    return ResolvedProvider(id=spec.id, model=model, api_key=api_key, base_url=base_url, key_env=key_env)


def configured_providers(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """`.env`에 키가 있어서 실제로 쓸 수 있는 제공자 id들 (PROVIDER_IDS 순서)."""
    env = environ if environ is not None else os.environ
    return tuple(pid for pid in PROVIDER_IDS if resolve_provider(pid, env) is not None)
