"""화자분리 결과를 LLM(Gemini)으로 문맥 기반 보정.

음향(목소리) 기반 화자분리는 톤이 바뀌거나 짧게 끼어들면 같은 사람을 다른 화자로
잘못 나누는 경우가 있다. 이 모듈은 전사 "내용"을 LLM에게 보여주고, 문맥상 명백히
같은 사람인데 다른 화자 번호로 나뉜 경우의 "병합 매핑"만 받아서 적용한다.
텍스트/타임스탬프를 LLM이 다시 쓰게 하지 않고 라벨 매핑만 요청해서 환각 위험을 최소화한다.

제공자: AssemblyAI LLM Gateway는 이 용도로 사용하지 않는다(정책) — 접근 가능한 소형 모델이
명백히 다른 사람을 병합하자고 제안하거나 존재하지 않는 화자 라벨을 지어내는 걸 확인했기 때문.
대신 Gemini만 사용하며, 무료 키(GEMINI_FREE_KEY)를 우선 사용하고 실패 시 일반 키(GEMINI_API_KEY)로
롤백한다. 무료 키는 이용자가 많아 429(rate limit) 등으로 실패하기 쉬워 재시도 횟수를 넉넉히 둔다.
"""
from __future__ import annotations

import json
import logging
import random
import time

import requests

from .align import SpeakerTranscriptSegment
from .llm_catalog import GEMINI_GENERATE_URL_TMPL, clear_selected_model, ensure_selected_model
from .secrets import get_api_key

logger = logging.getLogger(__name__)

MAX_TRANSCRIPT_CHARS = 60000  # 매우 긴 녹음은 앞부분만 보고 병합 판단 (한계는 README에 명시)

# 모델 id는 하드코딩하지 않는다 — core/llm_catalog.py가 실시간으로 모델 목록을 받아와
# 사용 가능한 걸 골라 llm_selection.json에 저장해두고, 여기서는 그걸 읽어서 쓴다.
# (Gemini 모델명이 자주 바뀌어서 실제로 하드코딩된 모델이 막힌 적이 있었음 — README 참고)

# provider -> 실패 시 재시도 횟수. 무료 키는 이용자가 많아 레이트리밋에 걸리기 쉬우므로 넉넉히.
RETRY_ATTEMPTS = {"gemini_free": 25, "gemini": 3}
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_RETRY_BASE_DELAY = 2.0
_RETRY_MAX_DELAY = 15.0

SYSTEM_PROMPT = (
    "당신은 화자분리(diarization) 결과를 검토하는 보조자입니다. "
    "아래 전사록은 목소리 음향 특성만으로 자동 분리된 화자 번호가 붙어 있습니다. "
    "이 방식은 같은 사람이 톤을 바꾸거나 짧게 끼어들면 다른 화자로 잘못 나뉘는 경우가 있습니다.\n\n"
    "매우 보수적으로 판단하세요. 병합 기준은 다음 중 하나가 아니면 병합하지 마세요:\n"
    "1) 한 화자 번호가 '제 이름은 OOO입니다' 같은 자기소개를 했고, 다른 화자 번호가 그 이름으로 "
    "불리거나 자신을 그 이름으로 다시 소개하는 경우\n"
    "2) 두 화자 번호의 발화가 명백히 같은 문장이 끊겨서 나뉜 경우(예: 한 화자 번호가 문장 중간에 "
    "끊기고 바로 다음에 다른 화자 번호가 그 문장을 이어받아 완성하는 경우)\n\n"
    "역할/주제/말투가 다르면(예: 한쪽은 등록 안내, 다른 쪽은 기술 이야기) 절대 같은 사람으로 "
    "보지 마세요. 확신이 90% 미만이면 병합하지 마세요. 병합을 놓치는 것이 서로 다른 사람을 "
    "합치는 것보다 훨씬 낫습니다. 입력에 나온 화자 라벨만 사용하고, 새로운 라벨을 만들어내지 마세요.\n\n"
    '먼저 "reasoning" 필드에 후보들을 간단히 검토한 근거를 한두 문장으로 쓰고, 그다음 "merges"에 '
    "최종 결정만 담아 JSON으로 출력하세요. 다른 텍스트는 출력하지 마세요. 형식: "
    '{"reasoning": "...", "merges": {"화자 F": "화자 A"}} '
    "(화자 F를 화자 A로 병합). 병합할 게 없으면 {\"reasoning\": \"...\", \"merges\": {}}."
)


def get_provider_candidates() -> list[tuple[str, str]]:
    """우선순위대로 (provider, api_key) 후보를 반환. 무료 Gemini 키 -> 일반 Gemini 키.

    AssemblyAI LLM Gateway는 화자 병합 용도로는 사용하지 않는다(정책 — README 참고).
    """
    candidates = []
    for provider in ("gemini_free", "gemini"):
        key = get_api_key(provider)
        if key:
            candidates.append((provider, key))
    return candidates


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def _resolve(label: str, merges: dict[str, str]) -> str:
    seen: set[str] = set()
    while label in merges and label not in seen:
        seen.add(label)
        label = merges[label]
    return label


def _call_gemini(user_content: str, api_key: str, model: str) -> str:
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user_content}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    url = GEMINI_GENERATE_URL_TMPL.format(model=model)
    resp = requests.post(f"{url}?key={api_key}", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        return status in RETRYABLE_STATUS
    if isinstance(exc, requests.exceptions.RequestException):
        return True  # 타임아웃/연결 오류 등
    return False


def _call_with_retry(
    user_content: str,
    provider: str,
    api_key: str,
    model: str,
    max_attempts: int,
    status_callback=None,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _call_gemini(user_content, api_key, model)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            status = e.response.status_code if isinstance(e, requests.exceptions.HTTPError) and e.response is not None else None
            if status == 404:
                # 선택해둔 모델 자체가 더 이상 이 키로 안 되는 것(예: 신규 사용자 대상 지원 종료).
                # 다음 실행 때는 새로 골라 쓰도록 저장된 선택을 지운다.
                logger.warning("[%s] 모델 '%s' 호출이 404로 실패해 선택을 초기화합니다: %s", provider, model, e)
                clear_selected_model(provider)
            retryable = _is_retryable(e)
            if not retryable or attempt == max_attempts:
                raise
            delay = min(_RETRY_BASE_DELAY * (1.5 ** (attempt - 1)), _RETRY_MAX_DELAY) + random.uniform(0, 1)
            msg = f"Gemini 호출 실패({e}), {attempt}/{max_attempts}번째 재시도까지 {delay:.0f}초 대기..."
            logger.warning(msg)
            if status_callback:
                status_callback(f"재시도 {attempt}/{max_attempts} (약 {delay:.0f}초 후)...")
            time.sleep(delay)
    raise last_exc  # pragma: no cover — 루프가 항상 return/raise로 빠짐


def suggest_merges(
    segments: list[SpeakerTranscriptSegment],
    status_callback=None,
) -> tuple[dict[str, str], str, str]:
    """LLM에게 화자 병합을 '제안'만 받아온다 (적용하지 않음).

    무료 Gemini 키를 우선 시도(최대 25회 재시도)하고, 모두 실패하면 일반 Gemini 키로
    롤백한다. 검증 결과 이 기능은 자동 적용하지 않고 사람이 검토 후 선택 적용하는 것을
    전제로 한다 (LLM이 가끔 존재하지 않는 화자 라벨을 지어내는 것을 확인했기 때문).

    반환: (merges 딕셔너리, LLM이 남긴 판단 근거 텍스트, 실제 사용된 provider)
    """
    if not segments:
        return {}, "", ""

    candidates = get_provider_candidates()
    if not candidates:
        return {}, "", ""

    lines = []
    total = 0
    for seg in segments:
        line = f"[{seg.speaker}] {seg.text}"
        if total + len(line) > MAX_TRANSCRIPT_CHARS:
            break
        lines.append(line)
        total += len(line)
    user_content = "\n".join(lines)

    errors: list[str] = []
    for provider, api_key in candidates:
        max_attempts = RETRY_ATTEMPTS.get(provider, 3)
        try:
            if status_callback:
                status_callback(f"LLM({provider}) 사용할 모델 확인 중...")
            model = ensure_selected_model(provider, api_key)

            if status_callback:
                status_callback(f"LLM({provider}, {model}) 화자 병합 제안 요청 중...")
            content = _call_with_retry(user_content, provider, api_key, model, max_attempts, status_callback)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] 모든 재시도 실패, 다음 제공자로 롤백: %s", provider, e)
            errors.append(f"{provider}: {e}")
            continue

        parsed = json.loads(_strip_code_fence(content))
        merges = parsed.get("merges") or {}
        reasoning = parsed.get("reasoning") or ""

        # 존재하지 않는(입력에 없던) 화자 라벨을 지어내는 경우가 있어, 실제 라벨만 유효하게 취급.
        valid_labels = {seg.speaker for seg in segments}
        merges = {
            src: dst
            for src, dst in merges.items()
            if src in valid_labels and dst in valid_labels and src != dst
        }

        logger.info("[%s] LLM 화자 병합 제안: %s (근거: %s)", provider, merges, reasoning)
        return merges, reasoning, provider

    raise RuntimeError("모든 LLM 제공자 호출 실패: " + " / ".join(errors))


def apply_merges(
    segments: list[SpeakerTranscriptSegment],
    merges: dict[str, str],
) -> list[SpeakerTranscriptSegment]:
    """사용자가 승인한 병합 매핑만 실제로 적용."""
    if not merges:
        return segments
    return [
        SpeakerTranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            language=seg.language,
            language_probability=seg.language_probability,
            speaker=_resolve(seg.speaker, merges),
        )
        for seg in segments
    ]
