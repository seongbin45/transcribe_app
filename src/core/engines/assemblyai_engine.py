"""AssemblyAI 기반 API 엔진. 한 번의 API 호출로 전사와 화자분리를 함께 받을 수 있다.

로컬 엔진과 달리 언어 감지는 파일 전체 기준 1회(코드스위칭 세그먼트별 재판정 없음).
실제 API 키로 검증 완료 (2026-08-29, Day1 자기소개 세션 3분 클립).

길이 제한(2026-08-31 추가): 실제로 10시간 31분짜리 파일을 API 엔진으로 돌렸다가
"Audio duration is too long."이라는 영문 오류를 그대로 받아 사용자에게 노출된 사례가
있었음. AssemblyAI 공식 문서(FAQ "Are there any limits on file size or file duration
for files submitted to the API?")에 따르면 `/v2/transcript`는 최대 10시간까지만
받는다 — 파일을 업로드하고 한참 기다린 뒤에야(전사 잡이 "error" 상태가 되어야) 이
사실을 알게 되는 게 아니라, 업로드 전에 미리 확인해서 한국어로 원인과 대안(파일
분할 또는 로컬 엔진 전환 — 로컬 엔진은 이런 제한이 없음)을 바로 알려준다.
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

from ..align import SpeakerTranscriptSegment
from ..audio_extract import probe_media
from .base import STTEngine, TranscriptSegment

BASE_URL = "https://api.assemblyai.com/v2"
POLL_INTERVAL_SEC = 3
POLL_TIMEOUT_SEC = 3600
# speech_models는 선택 항목이라 생략하면 계정 기본값(구버전일 수 있음)이 쓰이므로,
# 현재 플래그십 모델을 명시적으로 지정 (미지원 시 universal-2로 자동 폴백).
SPEECH_MODELS = ["universal-3-5-pro", "universal-2"]

# AssemblyAI 공식 문서 기준 단일 전사 요청의 최대 오디오 길이(10시간). 최소 길이(160ms)는
# 이 앱에서 그렇게 짧은 파일을 다룰 일이 거의 없어 별도로 검사하지 않음.
MAX_DURATION_SEC = 10 * 3600


class AssemblyAIError(RuntimeError):
    pass


def _format_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}시간 {m}분 {s}초"


class AssemblyAIEngine(STTEngine):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("AssemblyAI API 키가 필요합니다. 설정 화면에서 입력해주세요.")
        self._headers = {"authorization": api_key}

    def _check_duration(self, wav_path: Path) -> None:
        duration = probe_media(wav_path).duration_sec
        if duration > MAX_DURATION_SEC:
            raise AssemblyAIError(
                f"AssemblyAI는 한 번에 최대 10시간까지만 처리할 수 있는데, 이 파일은 "
                f"{_format_hms(duration)}로 그보다 깁니다. 파일을 10시간 이하로 나눠서 "
                f"각각 처리하거나, 설정 화면에서 엔진을 '로컬(faster-whisper + pyannote)'"
                f"로 바꿔서 처리해주세요 — 로컬 엔진은 이런 길이 제한이 없습니다."
            )

    def transcribe(
        self,
        wav_path: Path,
        languages: list[str],
        multilingual_mode: bool = False,
    ) -> list[TranscriptSegment]:
        self._check_duration(wav_path)
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
        self._check_duration(wav_path)
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
