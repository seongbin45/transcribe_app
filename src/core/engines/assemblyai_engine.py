"""AssemblyAI 기반 API 엔진. 한 번의 API 호출로 전사와 화자분리를 함께 받을 수 있다.

로컬 엔진과 달리 언어 감지는 파일 전체 기준 1회(코드스위칭 세그먼트별 재판정 없음).
실제 API 키로 검증 완료 (2026-08-29, Day1 자기소개 세션 3분 클립).
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

from ..align import SpeakerTranscriptSegment
from .base import STTEngine, TranscriptSegment

BASE_URL = "https://api.assemblyai.com/v2"
POLL_INTERVAL_SEC = 3
POLL_TIMEOUT_SEC = 3600
# speech_models는 선택 항목이라 생략하면 계정 기본값(구버전일 수 있음)이 쓰이므로,
# 현재 플래그십 모델을 명시적으로 지정 (미지원 시 universal-2로 자동 폴백).
SPEECH_MODELS = ["universal-3-5-pro", "universal-2"]


class AssemblyAIError(RuntimeError):
    pass


class AssemblyAIEngine(STTEngine):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("AssemblyAI API 키가 필요합니다. 설정 화면에서 입력해주세요.")
        self._headers = {"authorization": api_key}

    def transcribe(
        self,
        wav_path: Path,
        languages: list[str],
        multilingual_mode: bool = False,
    ) -> list[TranscriptSegment]:
        transcript_id = self._submit(wav_path, speaker_labels=False)
        data = self._poll(transcript_id)
        return self._build_from_paragraphs(transcript_id, data)

    def transcribe_with_diarization(
        self,
        wav_path: Path,
        languages: list[str],
        multilingual_mode: bool = False,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[SpeakerTranscriptSegment]:
        transcript_id = self._submit(
            wav_path, speaker_labels=True, min_speakers=min_speakers, max_speakers=max_speakers
        )
        data = self._poll(transcript_id)
        return self._build_from_utterances(data)

    # --- API 호출 -------------------------------------------------------
    def _upload(self, wav_path: Path) -> str:
        with open(wav_path, "rb") as f:
            resp = requests.post(f"{BASE_URL}/upload", headers=self._headers, data=f)
        resp.raise_for_status()
        return resp.json()["upload_url"]

    def _submit(
        self,
        wav_path: Path,
        speaker_labels: bool,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> str:
        upload_url = self._upload(wav_path)
        payload = {
            "audio_url": upload_url,
            "speaker_labels": speaker_labels,
            "language_detection": True,
            "speech_models": SPEECH_MODELS,
        }
        # 화자 수 힌트가 없으면 아주 긴/다양한 오디오에서 실제보다 훨씬 적은 화자 수로
        # 뭉뚱그려지는 경우가 있어(예: 5시간 녹음이 3명으로만 분류), 대략적인 범위라도 있으면 도움이 됨.
        if speaker_labels and (min_speakers is not None or max_speakers is not None):
            payload["speaker_options"] = {
                k: v
                for k, v in {
                    "min_speakers_expected": min_speakers,
                    "max_speakers_expected": max_speakers,
                }.items()
                if v is not None
            }
        resp = requests.post(
            f"{BASE_URL}/transcript",
            headers={**self._headers, "content-type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def _poll(self, transcript_id: str) -> dict:
        elapsed = 0
        while elapsed < POLL_TIMEOUT_SEC:
            resp = requests.get(f"{BASE_URL}/transcript/{transcript_id}", headers=self._headers)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            if status == "completed":
                return data
            if status == "error":
                raise AssemblyAIError(data.get("error", "알 수 없는 오류"))
            time.sleep(POLL_INTERVAL_SEC)
            elapsed += POLL_INTERVAL_SEC
        raise AssemblyAIError("전사 대기 시간이 초과되었습니다.")

    # --- 응답 -> 내부 데이터 모델 변환 -------------------------------------
    def _build_from_utterances(self, data: dict) -> list[SpeakerTranscriptSegment]:
        language = data.get("language_code") or "unknown"
        language_prob = data.get("language_confidence") or 0.0
        segments = []
        for u in data.get("utterances") or []:
            segments.append(
                SpeakerTranscriptSegment(
                    start=u["start"] / 1000.0,
                    end=u["end"] / 1000.0,
                    text=u["text"].strip(),
                    language=language,
                    language_probability=language_prob,
                    speaker=f"화자 {u['speaker']}",
                )
            )
        return segments

    def _build_from_paragraphs(self, transcript_id: str, data: dict) -> list[TranscriptSegment]:
        language = data.get("language_code") or "unknown"
        language_prob = data.get("language_confidence") or 0.0

        resp = requests.get(f"{BASE_URL}/transcript/{transcript_id}/paragraphs", headers=self._headers)
        resp.raise_for_status()
        paragraphs = resp.json().get("paragraphs") or []

        segments = []
        for p in paragraphs:
            segments.append(
                TranscriptSegment(
                    start=p["start"] / 1000.0,
                    end=p["end"] / 1000.0,
                    text=p["text"].strip(),
                    language=language,
                    language_probability=language_prob,
                )
            )
        return segments
