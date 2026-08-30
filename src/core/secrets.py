"""API 키/토큰 접근.

- HF_TOKEN(로컬 pyannote 모델 다운로드용): transcribe_app/.env 파일에서 읽음 (최초 1회 수동 설정, git에 커밋되지 않음)
- 그 외 API 엔진 키(예: AssemblyAI): 설정 화면에서 입력받으면 OS 자격 증명 저장소(keyring)에 저장됨.
  다만 .env에 직접 넣어두는 것도 지원한다 (HF_TOKEN과 동일한 패턴을 기대하는 사용자를 위해) —
  .env 값이 있으면 그걸 우선 쓰고, 없으면 keyring에 저장된 값을 쓴다.
"""
from __future__ import annotations

import os

import keyring
from dotenv import load_dotenv

from .config import APP_ROOT

load_dotenv(APP_ROOT / ".env")

KEYRING_SERVICE = "transcribe_app"

# provider -> .env에서 확인할 환경변수 이름 후보들
_ENV_VAR_CANDIDATES = {
    "assemblyai": ["ASSEMBLYAI_API_KEY", "ASSEMBLY_AI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "gemini_free": ["GEMINI_FREE_KEY"],
}


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def get_api_key(provider: str) -> str | None:
    for env_var in _ENV_VAR_CANDIDATES.get(provider, []):
        value = os.environ.get(env_var)
        if value:
            return value

    try:
        return keyring.get_password(KEYRING_SERVICE, provider) or None
    except keyring.errors.KeyringError:
        return None


def set_api_key(provider: str, api_key: str) -> None:
    keyring.set_password(KEYRING_SERVICE, provider, api_key)


def delete_api_key(provider: str) -> None:
    try:
        keyring.delete_password(KEYRING_SERVICE, provider)
    except keyring.errors.PasswordDeleteError:
        pass
