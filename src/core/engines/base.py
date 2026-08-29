"""STT 엔진 공통 인터페이스. 로컬(faster-whisper)/API 엔진을 동일한 방식으로 교체하기 위함."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscriptSegment:
    start: float  # 초
    end: float  # 초
    text: str
    language: str
    language_probability: float


class STTEngine(ABC):
    @abstractmethod
    def transcribe(
        self,
        wav_path: Path,
        languages: list[str],
        multilingual_mode: bool = False,
    ) -> list[TranscriptSegment]:
        """wav_path(16kHz mono)를 전사해 시간 정렬된 세그먼트 목록을 반환."""
        raise NotImplementedError
