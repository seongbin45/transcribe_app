"""LLM이 제안한 화자 병합을 사람이 검토/선택 승인하는 다이얼로그.

LLM(특히 이 앱에서 실제로 접근 가능한 소형 모델)이 가끔 명백히 틀린 병합을 제안하는 것을
검증으로 확인했기 때문에, 어떤 제안도 자동 적용하지 않고 항목별로 승인/거부를 받는다.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


def _sample_text(segments, speaker: str, max_len: int = 60) -> str:
    for seg in segments:
        if seg.speaker == speaker and seg.text.strip():
            text = seg.text.strip()
            return text if len(text) <= max_len else text[: max_len - 1] + "…"
    return "(발화 없음)"


class MergeReviewDialog(QDialog):
    def __init__(self, segments, merges: dict[str, str], reasoning: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("화자 병합 제안 검토 (LLM)")
        self.setMinimumSize(560, 420)
        self._checks: dict[tuple[str, str], QCheckBox] = {}

        layout = QVBoxLayout(self)

        warn = QLabel(
            "LLM이 문맥을 보고 제안한 병합 목록입니다. 자동으로 적용되지 않으며,\n"
            "체크한 항목만 '적용' 버튼을 눌러야 실제로 반영됩니다.\n"
            "제안이 잘못된 경우가 있을 수 있으니 아래 발화 예시를 보고 판단해주세요."
        )
        warn.setWordWrap(True)
        layout.addWidget(warn)

        if reasoning:
            reasoning_label = QLabel(f"LLM 판단 근거: {reasoning}")
            reasoning_label.setWordWrap(True)
            reasoning_label.setStyleSheet("color: gray;")
            layout.addWidget(reasoning_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)

        if not merges:
            container_layout.addWidget(QLabel("제안된 병합이 없습니다."))
        else:
            for src, dst in merges.items():
                cb = QCheckBox(f"{src}  →  {dst} 로 병합")
                cb.setChecked(False)
                container_layout.addWidget(cb)

                sample = QLabel(
                    f"    {src} 예시: \"{_sample_text(segments, src)}\"\n"
                    f"    {dst} 예시: \"{_sample_text(segments, dst)}\""
                )
                sample.setStyleSheet("color: gray;")
                sample.setWordWrap(True)
                container_layout.addWidget(sample)

                self._checks[(src, dst)] = cb

        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("체크한 항목 적용")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def approved_merges(self) -> dict[str, str]:
        return {src: dst for (src, dst), cb in self._checks.items() if cb.isChecked()}
