from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..document import build_turns
from ..format_utils import fmt_hhmmss


def export_markdown(segments, path: Path, title: str = "전사 결과") -> None:
    turns = build_turns(segments)
    lines = [f"# {title}", "", f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    for t in turns:
        speaker_label = t.speaker or "화자"
        lines.append(f"### {speaker_label} `[{fmt_hhmmss(t.start)} - {fmt_hhmmss(t.end)}]`")
        lines.append("")
        lines.append(t.text)
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def export_txt(segments, path: Path) -> None:
    turns = build_turns(segments)
    lines = []
    for t in turns:
        speaker_label = t.speaker or "화자"
        lines.append(f"[{fmt_hhmmss(t.start)} - {fmt_hhmmss(t.end)}] {speaker_label}: {t.text}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
