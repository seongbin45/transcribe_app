"""faster-whisper 기반 로컬 STT 엔진.

언어 감지 정확도를 위해 파일 전체를 한 번에 돌리지 않고,
VAD로 나눈 '발화 구간 묶음(윈도우)' 단위로 언어를 다시 판정한다.
- 너무 짧은 조각으로 나누면 언어 판정 자체가 불안정해지므로 window_seconds 길이로 묶어서 판정
- 기본(ko+en) 모드에서는 all_language_probs를 ko/en 두 후보로만 재정규화해 오판(일본어/중국어 등과 혼동)을 줄임
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

from .base import STTEngine, TranscriptSegment

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Whisper가 무음/저에너지 구간 끝에서 지어내는 것으로 잘 알려진 문구들
# (유튜브 자막 데이터로 학습된 영향). 실제 발화에서 이 문구가 그대로 나올 확률은 낮다고 보고 제거.
HALLUCINATION_PHRASES = {
    "시청해주셔서 감사합니다",
    "구독과 좋아요 부탁드립니다",
    "구독과 좋아요",
    "다음 영상에서 만나요",
    "많은 시청 부탁드립니다",
    "구독 좋아요 알림설정",
    "thank you for watching",
    "please subscribe",
    "don't forget to subscribe",
    "like and subscribe",
    "see you in the next video",
}

# no_speech_prob이 이 값보다 높으면(모델 스스로 '거의 무음'이라고 판단) 텍스트를 신뢰하지 않고 버림
NO_SPEECH_PROB_THRESHOLD = 0.85


def _is_hallucination(text: str, no_speech_prob: float) -> bool:
    normalized = text.strip().strip(".!?~ ").lower()
    if normalized in HALLUCINATION_PHRASES:
        return True
    if no_speech_prob > NO_SPEECH_PROB_THRESHOLD:
        return True
    return False


class LocalWhisperEngine(STTEngine):
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Path | None = None,
    ):
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root) if download_root else None,
        )

    def transcribe(
        self,
        wav_path: Path,
        languages: list[str],
        multilingual_mode: bool = False,
        window_seconds: float = 25.0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[TranscriptSegment]:
        """progress_callback(처리된 윈도우 수, 전체 윈도우 수)를 윈도우 하나 끝날 때마다 호출.

        VAD 윈도우 단위로 순차 처리하기 때문에 "지금까지 처리한 윈도우 수 / 전체 윈도우 수"가
        곧 실제 진행률의 합리적인 근사치다(윈도우 길이가 대체로 비슷해서 윈도우당 처리 시간도
        비슷함). GUI의 "로딩 진행률 표시" 요청에 대응하기 위해 추가.
        """
        audio = decode_audio(str(wav_path))
        speech_ts = get_speech_timestamps(
            audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=200)
        )
        if not speech_ts:
            logger.warning("음성 구간을 찾지 못했습니다: %s", wav_path)
            return []

        windows = self._group_into_windows(speech_ts, window_seconds)
        total_windows = len(windows)
        results: list[TranscriptSegment] = []

        for window_index, (w_start, w_end) in enumerate(windows, start=1):
            chunk = audio[w_start:w_end]
            segments, info = self.model.transcribe(
                chunk,
                language=None,
                task="transcribe",
                beam_size=5,
                vad_filter=False,  # 이미 위에서 VAD로 구간을 골라냈음
                condition_on_previous_text=False,  # 윈도우 간 언어/문맥 오염 방지
            )
            lang, lang_prob = self._resolve_language(info, languages, multilingual_mode)
            offset = w_start / SAMPLE_RATE

            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                if _is_hallucination(text, seg.no_speech_prob):
                    logger.info(
                        "환각 의심 세그먼트 제거 [%.1f-%.1f] (no_speech_prob=%.2f): %s",
                        offset + seg.start,
                        offset + seg.end,
                        seg.no_speech_prob,
                        text,
                    )
                    continue
                results.append(
                    TranscriptSegment(
                        start=offset + seg.start,
                        end=offset + seg.end,
                        text=text,
                        language=lang,
                        language_probability=lang_prob,
                    )
                )

            if progress_callback:
                progress_callback(window_index, total_windows)

        return results

    @staticmethod
    def _group_into_windows(
        speech_ts: list[dict], window_seconds: float
    ) -> list[tuple[int, int]]:
        max_samples = int(window_seconds * SAMPLE_RATE)
        windows: list[tuple[int, int]] = []
        cur_start: int | None = None
        cur_end: int | None = None

        for seg in speech_ts:
            s, e = seg["start"], seg["end"]
            if cur_start is None:
                cur_start, cur_end = s, e
            elif e - cur_start <= max_samples:
                cur_end = e
            else:
                windows.append((cur_start, cur_end))
                cur_start, cur_end = s, e

        if cur_start is not None:
            windows.append((cur_start, cur_end))

        return windows

    @staticmethod
    def _resolve_language(
        info, languages: list[str], multilingual_mode: bool
    ) -> tuple[str, float]:
        if multilingual_mode or not info.all_language_probs:
            return info.language, info.language_probability

        restricted = [(l, p) for l, p in info.all_language_probs if l in languages]
        total = sum(p for _, p in restricted)
        if not restricted or total <= 0:
            return info.language, info.language_probability

        restricted = sorted(((l, p / total) for l, p in restricted), key=lambda x: -x[1])
        return restricted[0]
