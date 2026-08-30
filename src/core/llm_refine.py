"""화자분리 결과를 LLM(AssemblyAI LLM Gateway)으로 문맥 기반 보정.

음향(목소리) 기반 화자분리는 톤이 바뀌거나 짧게 끼어들면 같은 사람을 다른 화자로
잘못 나누는 경우가 있다. 이 모듈은 전사 "내용"을 LLM에게 보여주고, 문맥상 명백히
같은 사람인데 다른 화자 번호로 나뉜 경우의 "병합 매핑"만 받아서 적용한다.
텍스트/타임스탬프를 LLM이 다시 쓰게 하지 않고 라벨 매핑만 요청해서 환각 위험을 최소화한다.
"""
from __future__ import annotations

import json
import logging

import requests

from .align import SpeakerTranscriptSegment

logger = logging.getLogger(__name__)

LLM_GATEWAY_URL = "https://llm-gateway.assemblyai.com/v1/chat/completions"
# 계정 등급에 따라 접근 가능한 모델이 다름. claude-sonnet-4-6 등 상위 모델은 이 계정에서
# "account does not have access" 400 에러가 나서, 실제로 접근 가능한 것으로 확인된 모델 사용.
MODEL = "qwen3.5-4b-32k-fast"
MAX_TRANSCRIPT_CHARS = 60000  # 매우 긴 녹음은 앞부분만 보고 병합 판단 (한계는 README에 명시)

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
    "합치는 것보다 훨씬 낫습니다.\n\n"
    '먼저 "reasoning" 필드에 후보들을 간단히 검토한 근거를 한두 문장으로 쓰고, 그다음 "merges"에 '
    "최종 결정만 담아 JSON으로 출력하세요. 다른 텍스트는 출력하지 마세요. 형식: "
    '{"reasoning": "...", "merges": {"화자 F": "화자 A"}} '
    "(화자 F를 화자 A로 병합). 병합할 게 없으면 {\"reasoning\": \"...\", \"merges\": {}}."
)


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


def suggest_merges(
    segments: list[SpeakerTranscriptSegment],
    api_key: str,
) -> tuple[dict[str, str], str]:
    """LLM에게 화자 병합을 '제안'만 받아온다 (적용하지 않음).

    검증 결과 이 기능이 쓸 수 있는 모델(계정 등급상 qwen3.5-4b-32k-fast만 접근 가능)이
    가끔 명백히 다른 사람을 병합하자고 하거나, 존재하지 않는 화자 라벨을 지어내는 것을
    확인했다. 그래서 자동 적용하지 않고 사람이 검토 후 선택 적용하는 것을 전제로 한다.

    반환: (merges 딕셔너리, LLM이 남긴 판단 근거 텍스트)
    """
    if not segments or not api_key:
        return {}, ""

    lines = []
    total = 0
    for seg in segments:
        line = f"[{seg.speaker}] {seg.text}"
        if total + len(line) > MAX_TRANSCRIPT_CHARS:
            break
        lines.append(line)
        total += len(line)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ],
        "max_tokens": 700,
    }

    resp = requests.post(
        LLM_GATEWAY_URL,
        headers={"authorization": api_key, "content-type": "application/json"},
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
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

    logger.info("LLM 화자 병합 제안: %s (근거: %s)", merges, reasoning)
    return merges, reasoning


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
