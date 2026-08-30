"""화자분리 결과를 LLM으로 문맥 기반 보정 — xai/openai/gemini/claude 전체 폴백 체인.

음향(목소리) 기반 화자분리는 톤이 바뀌거나 짧게 끼어들면 같은 사람을 다른 화자로
잘못 나누는 경우가 있다. 이 모듈은 전사 "내용"을 LLM에게 보여주고, 문맥상 명백히
같은 사람인데 다른 화자 번호로 나뉜 경우의 "병합 매핑"만 받아서 적용한다.
텍스트/타임스탬프를 LLM이 다시 쓰게 하지 않고 라벨 매핑만 요청해서 환각 위험을 최소화한다.

제공자: AssemblyAI LLM Gateway는 이 용도로 사용하지 않는다(정책) — 접근 가능한 소형 모델이
명백히 다른 사람을 병합하자고 제안하거나 존재하지 않는 화자 라벨을 지어내는 걸 확인했기 때문.

대신 `.env`에 있는 모든 LLM 제공자 키를 폴백 체인으로 쓴다
(Automating_automatic_message_sending/src/aam 의 xai/openai/gemini/claude 레지스트리를
core/llm_providers.py로 이식):
  1. gemini_free (GEMINI_FREE_KEY, 이 앱 전용 — 무료 티어라 이용자가 많아 실패하기 쉬워서
     재시도를 25회까지 함)
  2. gemini (GEMINI_API_KEY/GOOGLE_API_KEY)
  3. claude (ANTHROPIC_API_KEY)
  4. openai (OPENAI_API_KEY)
  5. xai (XAI_API_KEY)
모델 id는 하드코딩하지 않고 core/llm_catalog.py가 실시간으로 조회+검증해서 고른다.

증거 기반 검증(evidence grounding, 2026-08-31 추가): LLM에게 "확신도 90%"처럼 스스로
말하게 하는 것만으로는 부족하다는 걸 문헌 조사로 확인했다 — LLM-as-judge는 구조적으로
과신(overconfident)하는 경향이 있고(예: arxiv.org/abs/2508.06225), self-reported confidence는
실제 정확도와 잘 맞지 않는다. 대신 검증 가능한 신호를 요구하는 게 더 신뢰할 수 있다는 게
grounded citation/span-level verification 계열 연구(예: arxiv.org/abs/2408.04568)의 결론이라,
이제 각 병합 제안마다 "원문에서 그대로 복사한 인용문"을 화자별로 요구하고, 그 인용문이 실제
원문(세그먼트 텍스트)에 있는지 프로그램이 문자열 대조로 검증한다(`_verify_quote`). 인용문이
원문에서 확인되지 않으면 — 즉 LLM이 근거 자체를 지어냈다는 뜻이므로 — 그 병합 제안은 사람에게
보여주지도 않고 자동 폐기한다. 검증된 인용문은 리뷰 다이얼로그에 그대로 노출해서, 사람이 봐도
"그럴듯해서" 그냥 승인해버리는 automation bias를 줄인다(리뷰 화면에 실제 근거를 보여주는 것 자체가
"cognitive forcing function"이라는 human-in-the-loop 연구 결과 반영).

교차 제공자 컨센서스(2026-08-31 추가): multi-agent debate/self-consistency 계열 연구는
서로 다른 모델이 독립적으로 같은 결론에 도달했을 때 그 결론을 더 신뢰할 수 있다고 본다
(예: ChatGPT와 Bard가 각각 따로 틀렸던 답이 교차 검증 후 정답으로 수렴한 사례 등).
같은 벤더의 키 두 개(gemini_free/gemini)는 같은 모델이라 상관된 오류를 낼 수 있으므로,
1차로 성공한 제공자와 **다른 벤더**의 제공자에게 동일한 전사록을 독립적으로 다시 보여주고,
두 제공자가 모두 동의한(같은 화자 라벨 쌍을 제안한) 병합만 최종 후보로 남긴다
(`_consensus_checker_candidates`/`_run_consensus_check`). 다른 벤더 키가 없거나 호출이
전부 실패하면(예: 크레딧 부족) 조용히 1차 제공자 결과만 쓰는 것으로 후퇴하되, 그 사실을
`consensus_note`로 항상 사용자에게 알린다 — 교차검증이 됐는지 안 됐는지 사용자가 항상 알 수
있게 하는 것도 "AI에게 판단을 전부 맡기지 않는다"는 원칙의 일부다.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass

import requests

from .align import SpeakerTranscriptSegment
from .llm_call import call_llm
from .llm_catalog import clear_selected_model, ensure_selected_model
from .llm_providers import ResolvedProvider, resolve_slot

logger = logging.getLogger(__name__)

MIN_QUOTE_LEN = 4  # 너무 짧은 인용문은 아무 문장에나 우연히 걸릴 수 있어 증거로 인정하지 않음

RULE_DESCRIPTIONS = {
    1: "자기소개/호명 일치",
    2: "문장이 끊겼다가 이어짐",
}


@dataclass(frozen=True)
class MergeCandidate:
    """LLM이 제안하고, 인용문이 원문에서 실제로 확인된 병합 후보 하나."""

    src: str
    dst: str
    rule: int | None
    quote_src: str
    quote_dst: str

    def rule_description(self) -> str:
        return RULE_DESCRIPTIONS.get(self.rule, f"규칙 {self.rule}" if self.rule else "근거 불명")


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _verify_quote(quote: str, normalized_source: str) -> bool:
    """quote가 (공백 차이를 무시하고) 실제로 원문에 존재하는 부분 문자열인지 확인."""
    q = (quote or "").strip()
    if len(q) < MIN_QUOTE_LEN:
        return False
    return _normalize_for_match(q) in normalized_source


MAX_TRANSCRIPT_CHARS = 60000  # 매우 긴 녹음은 앞부분만 보고 병합 판단 (한계는 README에 명시)

# 슬롯(이 앱에서 부르는 이름) -> 실패 시 재시도 횟수.
# 무료 키는 이용자가 많아 레이트리밋에 걸리기 쉬우므로 넉넉히, 나머지는 일시적 오류 정도만.
RETRY_ATTEMPTS = {"gemini_free": 25}
DEFAULT_RETRY_ATTEMPTS = 3
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_RETRY_BASE_DELAY = 2.0
_RETRY_MAX_DELAY = 15.0

# 화자 병합 제안에 쓸 폴백 순서 (core/llm_providers.py의 SLOT_IDS는 레지스트리 나열 순서일
# 뿐이라 여기서 별도로 우선순위를 정한다). 무료 Gemini 다음으로는 같은 Gemini(일반 키),
# 그다음 추론 품질이 검증된 Claude, OpenAI를 먼저 시도하고 xAI를 마지막으로 둔다.
FALLBACK_ORDER = ("gemini_free", "gemini", "claude", "openai", "xai")

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
    "**각 병합 제안마다 반드시 증거를 제시하세요**: quote_src는 'from' 화자의 발화에서, "
    "quote_dst는 'to' 화자의 발화에서, 판단 근거가 되는 부분을 한 구절씩 **입력에 나온 텍스트 "
    "그대로** 복사해서 적으세요. 절대로 요약하거나 의역하거나 새로 만들어내지 마세요 — "
    "이 인용문은 프로그램이 원문과 문자열 대조로 검증하며, 원문에 실제로 없는 인용문이 확인되면 "
    "그 병합 제안 전체가 자동으로 폐기됩니다.\n\n"
    '다음 JSON 형식으로만 출력하세요(다른 텍스트는 출력하지 마세요): '
    '{"reasoning": "전체 판단을 요약한 한두 문장", "merges": ['
    '{"from": "화자 F", "to": "화자 A", "rule": 1, '
    '"quote_src": "화자 F 발화에서 그대로 복사한 인용문", '
    '"quote_dst": "화자 A 발화에서 그대로 복사한 인용문"}'
    ']} (화자 F를 화자 A로 병합, rule은 위 기준 1 또는 2 중 해당하는 번호). '
    '병합할 게 없으면 {"reasoning": "...", "merges": []}.'
)


def get_provider_candidates() -> list[tuple[str, ResolvedProvider]]:
    """(슬롯 이름, ResolvedProvider) 후보를 우선순위대로(FALLBACK_ORDER 순서) 반환.

    슬롯 이름은 재시도 횟수/모델 선택 저장 키/로그 표시에 쓰고, ResolvedProvider.id는
    실제 어떤 API 포맷으로 호출할지 판별하는 데 쓴다(gemini_free도 id="gemini"로 호출됨).

    AssemblyAI LLM Gateway는 화자 병합 용도로는 사용하지 않는다(정책 — README 참고).
    """
    candidates: list[tuple[str, ResolvedProvider]] = []
    for slot in FALLBACK_ORDER:
        resolved = resolve_slot(slot)
        if resolved:
            candidates.append((slot, resolved))
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


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        return status in RETRYABLE_STATUS
    if isinstance(exc, requests.exceptions.RequestException):
        return True  # 타임아웃/연결 오류 등
    return False


def _call_with_retry(
    user_content: str,
    slot: str,
    resolved: ResolvedProvider,
    model: str,
    max_attempts: int,
    status_callback=None,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return call_llm(
                resolved, model, SYSTEM_PROMPT, user_content, max_tokens=1500, json_mode=True
            )
        except Exception as e:  # noqa: BLE001
            last_exc = e
            status = e.response.status_code if isinstance(e, requests.exceptions.HTTPError) and e.response is not None else None
            if status == 404:
                # 선택해둔 모델 자체가 더 이상 이 키로 안 되는 것(예: 신규 사용자 대상 지원 종료).
                # 다음 실행 때는 새로 골라 쓰도록 저장된 선택을 지운다.
                logger.warning("[%s] 모델 '%s' 호출이 404로 실패해 선택을 초기화합니다: %s", slot, model, e)
                clear_selected_model(slot)
            retryable = _is_retryable(e)
            if not retryable or attempt == max_attempts:
                raise
            delay = min(_RETRY_BASE_DELAY * (1.5 ** (attempt - 1)), _RETRY_MAX_DELAY) + random.uniform(0, 1)
            logger.warning("LLM 호출 실패(%s), %d/%d번째 재시도까지 %.0f초 대기...", e, attempt, max_attempts, delay)
            if status_callback:
                status_callback(f"재시도 {attempt}/{max_attempts} (약 {delay:.0f}초 후)...")
            time.sleep(delay)
    raise last_exc  # pragma: no cover — 루프가 항상 return/raise로 빠짐


def _verify_and_filter_merges(
    raw_merges: object,
    segments: list[SpeakerTranscriptSegment],
    slot: str,
) -> list[MergeCandidate]:
    """LLM이 내놓은 원시 merges를 검증해 실제로 원문 근거가 확인된 것만 남긴다.

    라벨 자체(입력에 없는 화자 라벨을 지어내는 것)뿐 아니라, 이제 인용문까지 원문과
    문자열 대조로 검증한다 — 인용문이 지어낸 것이면 판단 근거 자체가 환각일 가능성이
    높다고 보고 그 병합 제안 전체를 폐기한다(사람에게 보여주지도 않음).
    """
    if not isinstance(raw_merges, list):
        if raw_merges:
            logger.warning("[%s] merges가 예상한 리스트 형식이 아니라 전부 무시합니다: %r", slot, raw_merges)
        return []

    valid_labels = {seg.speaker for seg in segments}
    normalized_blobs = {
        label: _normalize_for_match("".join(seg.text for seg in segments if seg.speaker == label))
        for label in valid_labels
    }

    result: list[MergeCandidate] = []
    for item in raw_merges:
        if not isinstance(item, dict):
            continue
        src = str(item.get("from") or "").strip()
        dst = str(item.get("to") or "").strip()
        quote_src = str(item.get("quote_src") or "").strip()
        quote_dst = str(item.get("quote_dst") or "").strip()
        rule = item.get("rule")

        if src not in valid_labels or dst not in valid_labels or src == dst:
            logger.info("[%s] 유효하지 않은 화자 라벨의 병합 제안 폐기: %s -> %s", slot, src, dst)
            continue

        if not _verify_quote(quote_src, normalized_blobs[src]) or not _verify_quote(
            quote_dst, normalized_blobs[dst]
        ):
            logger.warning(
                "[%s] 원문에서 확인되지 않는 인용문 — 환각 의심으로 병합 제안 폐기: "
                "%s -> %s (quote_src=%r, quote_dst=%r)",
                slot, src, dst, quote_src, quote_dst,
            )
            continue

        result.append(MergeCandidate(src=src, dst=dst, rule=rule, quote_src=quote_src, quote_dst=quote_dst))

    return result


def _consensus_checker_candidates(
    provider_candidates: list[tuple[str, ResolvedProvider]],
    primary_slot: str,
    primary_resolved: ResolvedProvider,
) -> list[tuple[str, ResolvedProvider]]:
    """1차 응답과 다른 "벤더"의 제공자만 교차검증 후보로 남긴다.

    ResolvedProvider.id가 실제 API 벤더를 나타낸다(gemini_free도 id="gemini") — 같은
    벤더의 키 두 개는 사실상 같은 모델이라 상관된 오류를 낼 수 있으므로 제외한다.
    """
    return [
        (slot, resolved)
        for slot, resolved in provider_candidates
        if slot != primary_slot and resolved.id != primary_resolved.id
    ]


def _run_consensus_check(
    user_content: str,
    checker_pool: list[tuple[str, ResolvedProvider]],
    segments: list[SpeakerTranscriptSegment],
    status_callback=None,
) -> tuple[str, list[MergeCandidate]] | None:
    """checker_pool을 순서대로 시도해, 성공한 첫 제공자의 (슬롯, 검증된 후보)를 반환.

    전부 실패하면(크레딧 부족 등) None — 호출 쪽이 1차 결과만 쓰는 것으로 후퇴한다.
    """
    for slot, resolved in checker_pool:
        try:
            if status_callback:
                status_callback(f"교차검증({slot})용 모델 확인 중...")
            model = ensure_selected_model(slot, resolved)

            if status_callback:
                status_callback(f"교차검증({slot}, {model}) 요청 중...")
            max_attempts = RETRY_ATTEMPTS.get(slot, DEFAULT_RETRY_ATTEMPTS)
            content = _call_with_retry(user_content, slot, resolved, model, max_attempts, status_callback)
            parsed = json.loads(_strip_code_fence(content))
            candidates = _verify_and_filter_merges(parsed.get("merges"), segments, slot)
            return slot, candidates
        except Exception as e:  # noqa: BLE001
            logger.warning("[교차검증] %s 호출 실패, 다음 후보로 넘어감: %s", slot, e)
            continue
    return None


def suggest_merges(
    segments: list[SpeakerTranscriptSegment],
    status_callback=None,
    cross_validate: bool = True,
) -> tuple[list[MergeCandidate], str, str, str]:
    """LLM에게 화자 병합을 '제안'만 받아온다 (적용하지 않음).

    무료 Gemini 키 -> 일반 Gemini -> Claude -> OpenAI -> xAI 순으로 시도하고,
    (.env에 키가 없는 제공자는 건너뜀) 각 단계는 실패하면 다음으로 롤백한다.
    검증 결과 이 기능은 자동 적용하지 않고 사람이 검토 후 선택 적용하는 것을
    전제로 한다 (LLM이 가끔 존재하지 않는 화자 라벨을 지어내는 것을 확인했기 때문).
    각 제안은 원문에서 실제로 확인된 인용문(quote_src/quote_dst)이 있어야만
    후보로 남는다 — `_verify_and_filter_merges` 참고.

    cross_validate=True(기본값)이고 1차 응답에서 병합 제안이 하나라도 나오면, 가능한
    경우 **다른 벤더**의 제공자에게 동일한 전사록을 독립적으로 다시 보여주고 두 제공자가
    모두 동의한 것만 최종 후보로 남긴다(교차 제공자 컨센서스). 다른 벤더 키가 없거나
    전부 실패하면 1차 결과만 쓰되 그 사실을 항상 consensus_note로 알린다.
    cross_validate=False면(설정 화면에서 사용자가 끈 경우) 애초에 교차검증을 시도하지
    않는다 — API 호출을 절반으로 줄이는 대신 단일 제공자 판단만 신뢰하게 됨을 의미하며,
    이 트레이드오프도 consensus_note로 명시한다.

    반환: (최종 MergeCandidate 리스트, LLM이 남긴 판단 근거 텍스트, 1차 사용 슬롯 이름,
           교차검증 결과를 사람이 읽을 수 있게 설명하는 문자열)
    """
    if not segments:
        return [], "", "", ""

    provider_candidates = get_provider_candidates()
    if not provider_candidates:
        return [], "", "", ""

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
    primary_slot: str | None = None
    primary_resolved: ResolvedProvider | None = None
    primary_candidates: list[MergeCandidate] = []
    reasoning = ""
    for slot, resolved in provider_candidates:
        max_attempts = RETRY_ATTEMPTS.get(slot, DEFAULT_RETRY_ATTEMPTS)
        try:
            if status_callback:
                status_callback(f"LLM({slot}) 사용할 모델 확인 중...")
            model = ensure_selected_model(slot, resolved)

            if status_callback:
                status_callback(f"LLM({slot}, {model}) 화자 병합 제안 요청 중...")
            content = _call_with_retry(user_content, slot, resolved, model, max_attempts, status_callback)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] 모든 재시도 실패, 다음 제공자로 롤백: %s", slot, e)
            errors.append(f"{slot}: {e}")
            continue

        parsed = json.loads(_strip_code_fence(content))
        reasoning = parsed.get("reasoning") or ""
        primary_candidates = _verify_and_filter_merges(parsed.get("merges"), segments, slot)
        primary_slot, primary_resolved = slot, resolved
        logger.info(
            "[%s] LLM 화자 병합 제안(증거 검증 후 %d건): %s (근거: %s)",
            slot, len(primary_candidates), primary_candidates, reasoning,
        )
        break

    if primary_slot is None or primary_resolved is None:
        raise RuntimeError("모든 LLM 제공자 호출 실패: " + " / ".join(errors))

    if not cross_validate:
        return (
            primary_candidates, reasoning, primary_slot,
            "교차검증 꺼짐(설정에서 비활성화됨) — 단일 제공자 결과만 사용합니다.",
        )

    if not primary_candidates:
        return primary_candidates, reasoning, primary_slot, "제안된 병합이 없어 교차검증을 생략했습니다."

    checker_pool = _consensus_checker_candidates(provider_candidates, primary_slot, primary_resolved)
    if not checker_pool:
        return (
            primary_candidates, reasoning, primary_slot,
            "교차검증 불가(다른 벤더의 API 키가 없음) — 단일 제공자 결과만 사용합니다.",
        )

    checker_result = _run_consensus_check(user_content, checker_pool, segments, status_callback)
    if checker_result is None:
        return (
            primary_candidates, reasoning, primary_slot,
            "교차검증을 시도했지만 다른 제공자 호출이 모두 실패해 단일 제공자 결과만 사용합니다.",
        )

    checker_slot, checker_candidates = checker_result
    checker_pairs = {(c.src, c.dst) for c in checker_candidates}
    agreed = [c for c in primary_candidates if (c.src, c.dst) in checker_pairs]
    dropped = len(primary_candidates) - len(agreed)

    if dropped:
        note = (
            f"{checker_slot}로 교차검증: {len(primary_candidates)}건 중 {len(agreed)}건 일치, "
            f"{dropped}건은 다른 제공자가 동의하지 않아 제외했습니다."
        )
    else:
        note = f"{checker_slot}로 교차검증: {len(agreed)}건 모두 일치했습니다."
    logger.info("[교차검증] primary=%s checker=%s -> 최종 %d건 (%s)", primary_slot, checker_slot, len(agreed), note)

    return agreed, reasoning, primary_slot, note


def apply_merges(
    segments: list[SpeakerTranscriptSegment],
    candidates: list[MergeCandidate],
) -> list[SpeakerTranscriptSegment]:
    """사용자가 승인한 병합 후보만 실제로 적용."""
    if not candidates:
        return segments
    merges = {c.src: c.dst for c in candidates}
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
