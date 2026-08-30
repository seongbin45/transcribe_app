"""사용자 설정(언어, 엔진 모드 등)을 로컬 JSON 파일로 저장/로드. API 키 등 비밀값은 여기 담지 않음(secrets.py 참고)."""
from __future__ import annotations

import json
import logging

from .config import SETTINGS_PATH, Settings

logger = logging.getLogger(__name__)

# Settings 데이터클래스 필드 중 JSON에 저장할 것들만 (models_dir/output_dir은 경로 고정값이라 제외)
_PERSISTED_FIELDS = [
    "languages",
    "multilingual_mode",
    "engine_mode",
    "api_provider",
    "whisper_model_size",
    "diarize_default",
    "cross_validate_merges",
]


def load_settings() -> Settings:
    if not SETTINGS_PATH.exists():
        return Settings()

    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("설정 파일을 읽지 못해 기본값을 사용합니다: %s", e)
        return Settings()

    settings = Settings()
    for key in _PERSISTED_FIELDS:
        if key in data:
            setattr(settings, key, data[key])
    return settings


def save_settings(settings: Settings) -> None:
    data = {key: getattr(settings, key) for key in _PERSISTED_FIELDS}
    SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
