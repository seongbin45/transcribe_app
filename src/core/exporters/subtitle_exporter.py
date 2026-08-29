"""srt/vtt 자막 출력. 문서용(docx/md) 출력과 달리 화자 턴으로 묶지 않고
세그먼트 단위 그대로 내보내 영상과의 타이밍을 맞춘다."""
from __future__ import annotations

from pathlib import Path

from ..format_utils import fmt_srt_timestamp, fmt_vtt_timestamp

MIN_DURATION = 0.2  # 재생기 호환을 위한 최소 표시 시간(초)


def _label(seg) -> str:
    speaker = getattr(seg, "speaker", None)
    text = seg.text.strip()
    return f"[{speaker}] {text}" if speaker else text


def _safe_end(seg) -> float:
    return max(seg.end, seg.start + MIN_DURATION)


def export_srt(segments, path: Path) -> None:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{fmt_srt_timestamp(seg.start)} --> {fmt_srt_timestamp(_safe_end(seg))}")
        lines.append(_label(seg))
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def export_vtt(segments, path: Path) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{fmt_vtt_timestamp(seg.start)} --> {fmt_vtt_timestamp(_safe_end(seg))}")
        lines.append(_label(seg))
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
