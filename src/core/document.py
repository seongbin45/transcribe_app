"""전사 세그먼트를 문서 출력용 '화자 턴' 단위로 묶는다.

docx/markdown/txt 출력은 같은 화자가 연속으로 말한 세그먼트를 하나의 문단으로 합쳐서
회의록처럼 읽기 좋게 만든다. srt/vtt 자막은 이 묶음을 쓰지 않고 세그먼트 단위 그대로 사용한다
(영상과 타이밍을 맞추려면 짧은 단위가 유리하기 때문).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranscriptTurn:
    speaker: str | None
    start: float
    end: float
    text: str


def build_turns(segments) -> list[TranscriptTurn]:
    turns: list[TranscriptTurn] = []
    for seg in segments:
        speaker = getattr(seg, "speaker", None)
        if turns and turns[-1].speaker == speaker:
            turns[-1].end = seg.end
            turns[-1].text += " " + seg.text.strip()
        else:
            turns.append(
                TranscriptTurn(speaker=speaker, start=seg.start, end=seg.end, text=seg.text.strip())
            )
    return turns
