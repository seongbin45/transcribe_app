"""LLM이 제안한 화자 병합을 사람이 검토/선택 승인하는 다이얼로그.

LLM(특히 이 앱에서 실제로 접근 가능한 소형 모델)이 가끔 명백히 틀린 병합을 제안하는 것을
검증으로 확인했기 때문에, 어떤 제안도 자동 적용하지 않고 항목별로 승인/거부를 받는다.

여기 표시되는 인용문(quote_src/quote_dst)은 core/llm_refine.py가 원문과 문자열 대조로
이미 검증한 것만 남긴 것이다(원문에 없는 인용문이 확인되면 그 제안은 여기 도달하기 전에
자동 폐기됨). 그래도 최종 판단은 사람이 직접 근거를 읽고 승인 여부를 정하도록 한다 —
"그럴듯해 보이면 그냥 승인"하는 automation bias를 줄이기 위해 실제 근거를 보여준다.

consensus_note는 다른 벤더의 제공자에게 같은 전사록을 독립적으로 다시 물어봤을 때(교차
제공자 컨센서스, core/llm_refine.py의 suggest_merges 참고) 두 제공자가 얼마나 일치했는지
알려준다 — 교차검증이 실제로 됐는지, 아니면 다른 키가 없어서/실패해서 단일 제공자 결과만
쓴 것인지 사람이 항상 알 수 있게 한다.
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

from core.llm_refine import MergeCandidate


class MergeReviewDialog(QDialog):
    def __init__(
        self,
        candidates: list[MergeCandidate],
        reasoning: str,
        parent=None,
        consensus_note: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("화자 병합 제안 검토 (LLM)")
        self.setMinimumSize(560, 420)
        self._candidates = candidates
        self._checks: dict[int, QCheckBox] = {}

        layout = QVBoxLayout(self)

        warn = QLabel(
            "LLM이 문맥을 보고 제안한 병합 목록입니다. 자동으로 적용되지 않으며,\n"
            "체크한 항목만 '적용' 버튼을 눌러야 실제로 반영됩니다.\n"
            "아래 인용문은 프로그램이 원문에 실제로 있는지 대조 확인한 것입니다(확인되지 않은 "
            "인용문이 있던 제안은 이미 자동으로 제외되었습니다). 그래도 근거를 직접 읽고 판단해주세요."
        )
        warn.setWordWrap(True)
        layout.addWidget(warn)

        if consensus_note:
            consensus_label = QLabel(f"교차검증: {consensus_note}")
            consensus_label.setWordWrap(True)
            consensus_label.setStyleSheet("color: #b06000; font-weight: bold;")
            layout.addWidget(consensus_label)

        if reasoning:
            reasoning_label = QLabel(f"LLM 판단 근거(전체 요약): {reasoning}")
            reasoning_label.setWordWrap(True)
            reasoning_label.setStyleSheet("color: gray;")
            layout.addWidget(reasoning_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)

        if not candidates:
            container_layout.addWidget(QLabel("제안된 병합이 없습니다."))
        else:
            for i, c in enumerate(candidates):
                cb = QCheckBox(f"{c.src}  →  {c.dst} 로 병합  (근거 유형: {c.rule_description()})")
                cb.setChecked(False)
                container_layout.addWidget(cb)

                sample = QLabel(
                    f"    {c.src} 인용(원문 확인됨): \"{c.quote_src}\"\n"
                    f"    {c.dst} 인용(원문 확인됨): \"{c.quote_dst}\""
                )
                sample.setStyleSheet("color: gray;")
                sample.setWordWrap(True)
                container_layout.addWidget(sample)

                self._checks[i] = cb

        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("체크한 항목 적용")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def approved_merges(self) -> list[MergeCandidate]:
        return [self._candidates[i] for i, cb in self._checks.items() if cb.isChecked()]
