"""STT 세그먼트(시간, 텍스트, 언어)와 화자분리 세그먼트(시간, 화자)를 시간 겹침 기준으로 병합."""
from __future__ import annotations

from dataclasses import dataclass

from .engines.base import TranscriptSegment
from .engines.diarization_base import SpeakerSegment

UNKNOWN_SPEAKER = "화자 미상"


@dataclass
class SpeakerTranscriptSegment:
    start: float
    end: float
    text: str
    language: str
    language_probability: float
    speaker: str


def assign_speakers(
    transcript_segments: list[TranscriptSegment],
    speaker_segments: list[SpeakerSegment],
) -> list[SpeakerTranscriptSegment]:
    results: list[SpeakerTranscriptSegment] = []

    for t in transcript_segments:
        best_speaker = UNKNOWN_SPEAKER
        best_overlap = 0.0
        for s in speaker_segments:
            overlap = min(t.end, s.end) - max(t.start, s.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = s.speaker

        results.append(
            SpeakerTranscriptSegment(
                start=t.start,
                end=t.end,
                text=t.text,
                language=t.language,
                language_probability=t.language_probability,
                speaker=best_speaker,
            )
        )

    return results
