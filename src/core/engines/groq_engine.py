"""Groq(Whisper large-v3-turbo) 기반 초고속 클라우드 STT 엔진.

로컬(faster-whisper)이나 AssemblyAI보다 훨씬 빠른 처리 속도가 필요할 때 쓴다 — Groq 공식
문서 기준 실시간의 200배 이상(60분 오디오를 16초 만에 처리)이라는 조사 결과를 참고해 도입.
다만 Groq 자체는 화자분리를 지원하지 않는 순수 STT라서, 화자 구분이 필요하면
core/engines/pyannoteai_engine.py의 화자분리 결과와 core/align.py의 assign_speakers()로
따로 정렬해서 합쳐야 한다(TranscribeWorker에서 조합).

요청당 파일 크기 제한(무료 티어 25MB)이 있어, core/audio_chunk.py로 미리 잘라 각 조각을
독립적으로 전사한 뒤 결과 타임스탬프에 조각이 시작하는 시각을 더해 이어붙인다. 조각 경계에서
문장이 살짝 끊길 수 있는 건 알려진 한계(README 참고, 겹침 없는 단순 분할 방식 사용).
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable

import requests

from ..audio_chunk import split_audio
from .base import STTEngine, TranscriptSegment

BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "whisper-large-v3-turbo"
REQUEST_TIMEOUT_SEC = 300


class GroqError(RuntimeError):
    pass


class GroqEngine(STTEngine):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API 키가 필요합니다. 설정 화면에서 입력해주세요.")
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def transcribe(
        self,
        wav_path: Path,
        languages: list[str],
        multilingual_mode: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[TranscriptSegment]:
        # 언어 힌트: Groq(OpenAI 호환 엔드포인트)는 ISO-639-1 단일 코드 하나만 받는다
        # (이 앱의 핵심 기능인 구간별 코드스위칭 재판정은 지원 안 함). 다국어 모드이거나
        # 언어가 2개 이상 선택돼 있으면 잘못된 단일 언어를 강제하는 게 더 위험하므로
        # 힌트 없이 Whisper 자체 자동 감지에 맡긴다.
        language_hint = None if multilingual_mode or len(languages) != 1 else languages[0]

        tmp_dir = Path(tempfile.mkdtemp(prefix="groq_chunks_"))
        try:
            chunks = split_audio(wav_path, tmp_dir)
            results: list[TranscriptSegment] = []
            for i, chunk in enumerate(chunks, start=1):
                results.extend(self._transcribe_chunk(chunk, language_hint))
                if progress_callback:
                    progress_callback(i, len(chunks))
            return results
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _transcribe_chunk(self, chunk, language_hint: str | None) -> list[TranscriptSegment]:
        data = {"model": MODEL, "response_format": "verbose_json"}
        if language_hint:
            data["language"] = language_hint

        with open(chunk.path, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/audio/transcriptions",
                headers=self._headers,
                data=data,
                files={"file": (chunk.path.name, f, "audio/mpeg")},
                timeout=REQUEST_TIMEOUT_SEC,
            )
        if resp.status_code >= 400:
            raise GroqError(f"Groq STT 요청 실패({resp.status_code}): {resp.text[:500]}")

        payload = resp.json()
        language = payload.get("language") or "unknown"

        segments: list[TranscriptSegment] = []
        for seg in payload.get("segments") or []:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start=chunk.offset_sec + float(seg["start"]),
                    end=chunk.offset_sec + float(seg["end"]),
                    text=text,
                    language=language,
                    # Groq(OpenAI 호환) 응답은 세그먼트별 언어 신뢰도를 제공하지 않는다.
                    # 없는 값을 그럴듯하게 지어내지 않고 "신뢰도 정보 없음"을 뜻하는
                    # 자리표시자로 1.0을 씀 — UI에 항상 100%로 표시되는 건 실제 신뢰도가
                    # 아니라 이 한계 때문임을 여기 명시해둔다.
                    language_probability=1.0,
                )
            )
        return segments
