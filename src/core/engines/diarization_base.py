"""화자분리 엔진 공통 인터페이스."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SpeakerSegment:
    start: float  # 초
    end: float  # 초
    speaker: str  # 예: "SPEAKER_00"


class DiarizationEngine(ABC):
    @abstractmethod
    def diarize(self, wav_path: Path) -> list[SpeakerSegment]:
        """wav_path(16kHz mono)를 분석해 화자별 발화 구간 목록을 반환."""
        raise NotImplementedError
