"""Gemini 모델 카탈로그: 라이브 API에서 모델 목록을 가져오고, 선택 결과를 로컬 JSON에 저장.

모델 id를 코드에 하드코딩하지 않기 위한 모듈. Gemini 모델명은 자주 바뀌는데(실제로
`gemini-2.5-flash`가 무료 키에서 어느 순간 404로 막히고 API가 직접 "이제
`gemini-3.6-flash`를 쓰라"고 안내하는 걸 겪음), 코드에 문자열로 박아두면 이런 변경마다
코드를 고쳐야 한다. 대신 `GET /v1beta/models`로 그 키가 실제 쓸 수 있는 모델을 물어보고,
사용자가 그중 하나를 고르면 `llm_selection.json`에 저장해 재사용한다.

C:\\...\\Automating_automatic_message_sending\\src\\aam\\catalog.py 의 설계를 참고했다
(그 프로젝트는 xai/openai/gemini/claude 여러 제공자를 지원하는 범용 CLI 도구이고,
이 앱은 Gemini만 쓰므로 그 부분만 단순화해서 가져왔다).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from .config import APP_ROOT

logger = logging.getLogger(__name__)

SELECTION_PATH = APP_ROOT / "llm_selection.json"
GEMINI_LIST_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_GENERATE_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# 선택할 모델이 하나도 없을 때(최초 실행 등) 자동으로 골라줄 기준.
# 정확한 모델명이 아니라 "이 키워드가 id에 들어간 것 중 첫 번째"를 고르는 방식이라
# 모델명이 바뀌어도(예: 3.6 -> 3.7) 계속 동작한다.
#
# 주의: `GET /models` 목록에 나온다고 실제로 호출 가능하다는 보장은 없다 — 실제로
# gemini-2.5-flash가 목록에는 있지만 무료 키로 generateContent를 호출하면
# "신규 사용자에게 더 이상 제공되지 않음"이라는 404가 났다. 그래서 자동 선택 시
# 후보를 실제로 한 번 호출해 검증한다(`_probe_model`).
AUTO_PICK_KEYWORDS = ("flash",)
MAX_AUTO_PICK_CANDIDATES = 6  # 검증 호출 횟수를 제한(느려지는 것 방지)


@dataclass(frozen=True)
class RemoteModel:
    id: str
    display_name: str


class CatalogError(RuntimeError):
    pass


def _gemini_id(name: str) -> str:
    raw = name.strip()
    return raw[len("models/") :] if raw.startswith("models/") else raw


def fetch_gemini_models(api_key: str) -> list[RemoteModel]:
    """이 키로 실제 접근 가능한 Gemini 모델 목록을 실시간으로 가져온다.

    텍스트 생성(generateContent)을 지원하지 않는 모델(임베딩, TTS 전용 등)은 제외한다.
    """
    models: list[RemoteModel] = []
    page_token: str | None = None
    while True:
        params = {"key": api_key, "pageSize": "100"}
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(GEMINI_LIST_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        for item in payload.get("models") or []:
            methods = item.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue
            model_id = _gemini_id(str(item.get("name") or ""))
            if not model_id:
                continue
            display = str(item.get("displayName") or model_id)
            models.append(RemoteModel(id=model_id, display_name=display))

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    unique: dict[str, RemoteModel] = {}
    for m in models:
        unique.setdefault(m.id, m)
    return sorted(unique.values(), key=lambda m: m.id.lower())


def _read_selection_file() -> dict:
    if not SELECTION_PATH.exists():
        return {}
    try:
        return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("%s 파싱 실패, 빈 값으로 취급합니다.", SELECTION_PATH)
        return {}


def load_selected_model(provider: str) -> str | None:
    """provider별로 따로 저장한다 — 무료 키와 일반 키가 실제로 쓸 수 있는 모델이 다를 수 있어서
    (예: gemini-2.5-flash가 무료 키에서는 404, 일반 키에서는 정상 호출됨을 실제로 확인)."""
    models = _read_selection_file().get("models") or {}
    return models.get(provider) or None


def save_selected_model(provider: str, model_id: str) -> None:
    data = _read_selection_file()
    data["_comment"] = (
        "Gemini 모델 id는 자주 바뀌므로 코드에 하드코딩하지 않고 여기 저장한다. "
        "설정 화면에서 실시간 목록을 불러와 변경할 수 있다. "
        "provider별로 따로 저장하는 이유: 무료 키/일반 키가 접근 가능한 모델이 다를 수 있음."
    )
    models = data.get("models") or {}
    models[provider] = model_id
    data["models"] = models
    data["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    SELECTION_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_selected_model(provider: str) -> None:
    """선택된 모델이 더 이상 동작하지 않는 걸 확인했을 때 초기화(다음 호출 때 재선택하도록)."""
    data = _read_selection_file()
    models = data.get("models") or {}
    if provider in models:
        del models[provider]
        data["models"] = models
        SELECTION_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _probe_model(model_id: str, api_key: str) -> bool:
    """모델이 이 키로 실제 호출 가능한지 최소 요청으로 확인.

    404/400 같은 "이 키로는 이 모델 자체가 안 됨" 오류는 영구적이라고 보고 즉시 False.
    429/503 같은 "지금 혼잡함" 오류는 몇 번 재시도해보고, 계속 그러면 판단을 보류하고
    False를 반환한다(자동 선택 단계에서는 보수적으로 다음 후보를 본다 — 실제 사용 시에는
    core/llm_refine.py의 재시도 로직이 이런 일시적 오류를 알아서 처리한다).
    """
    url = GEMINI_GENERATE_URL_TMPL.format(model=model_id)
    payload = {
        "contents": [{"parts": [{"text": "ping"}]}],
        "generationConfig": {"maxOutputTokens": 5},
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, params={"key": api_key}, json=payload, timeout=20)
        except requests.exceptions.RequestException:
            time.sleep(2)
            continue
        if resp.status_code == 200:
            return True
        if resp.status_code in (429, 503):
            time.sleep(3)
            continue
        return False
    return False


def select_model(provider: str, model_id: str, api_key: str) -> RemoteModel:
    """고른 모델이 실제 라이브 카탈로그에 있는지 확인한 뒤 저장한다."""
    catalog = fetch_gemini_models(api_key)
    wanted = model_id.strip().lower()
    match = next(
        (m for m in catalog if m.id.lower() == wanted or m.display_name.lower() == wanted), None
    )
    if match is None:
        raise CatalogError(f"'{model_id}'는 현재 이 키로 접근 가능한 Gemini 모델 목록에 없습니다.")
    save_selected_model(provider, match.id)
    return match


def ensure_selected_model(provider: str, api_key: str) -> str:
    """선택된 모델을 반환. 없으면 라이브 목록에서 실제로 호출되는 걸 확인한 뒤 저장한다.

    (설정 화면에서 언제든 바꿀 수 있으므로, 처음 쓸 때 매번 고르라고 막지는 않는다.)
    """
    selected = load_selected_model(provider)
    if selected:
        return selected

    catalog = fetch_gemini_models(api_key)
    if not catalog:
        raise CatalogError("이 키로 접근 가능한 Gemini 모델이 없습니다.")

    keyword_matches = [m for m in catalog if any(kw in m.id.lower() for kw in AUTO_PICK_KEYWORDS)]
    others = [m for m in catalog if m not in keyword_matches]
    candidates = (keyword_matches + others)[:MAX_AUTO_PICK_CANDIDATES]

    for candidate in candidates:
        if _probe_model(candidate.id, api_key):
            save_selected_model(provider, candidate.id)
            logger.info(
                "[%s] Gemini 모델이 선택되어 있지 않아 자동으로 '%s'을(를) 골랐습니다.", provider, candidate.id
            )
            return candidate.id

    raise CatalogError(
        f"자동으로 시험해본 모델 {len(candidates)}개 모두 호출에 실패했습니다. "
        "설정 화면에서 모델 목록을 직접 확인해주세요."
    )
