"""설정 화면: LLM 제공자별로 실시간 모델 목록을 조회하고 직접 모델을 선택하는 다이얼로그.

지금까지는 core/llm_catalog.ensure_selected_model()이 자동으로(키워드 매칭 + 실호출 검증)
모델을 골랐다. 이 다이얼로그는 그 자동 선택을 사용자가 원하면 무시하고, 실시간 목록에서
직접 골라 core/llm_catalog.select_model()로 저장할 수 있게 한다.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.llm_catalog import RemoteModel, clear_selected_model, fetch_models, load_selected_model, select_model
from core.llm_providers import SLOT_LABELS, ResolvedProvider, configured_slots, resolve_slot


class _FetchModelsWorker(QThread):
    succeeded = Signal(list)  # list[RemoteModel]
    failed = Signal(str)

    def __init__(self, resolved: ResolvedProvider):
        super().__init__()
        self._resolved = resolved

    def run(self) -> None:
        try:
            models = fetch_models(self._resolved)
            self.succeeded.emit(models)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"모델 목록 조회 실패: {e}")


class _SelectModelWorker(QThread):
    succeeded = Signal(str)  # 저장된 model id
    failed = Signal(str)

    def __init__(self, slot: str, model_id: str, resolved: ResolvedProvider):
        super().__init__()
        self._slot = slot
        self._model_id = model_id
        self._resolved = resolved

    def run(self) -> None:
        try:
            match = select_model(self._slot, self._model_id, self._resolved)
            self.succeeded.emit(match.id)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"모델 저장 실패: {e}")


class LlmModelDialog(QDialog):
    """화자 병합 제안(LLM) 기능이 슬롯(gemini_free/gemini/claude/openai/xai)별로
    어떤 모델을 쓸지 실시간 목록에서 조회하고 직접 지정한다."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LLM 모델 선택")
        self.setMinimumSize(480, 420)
        self._fetch_worker: _FetchModelsWorker | None = None
        self._select_worker: _SelectModelWorker | None = None
        self._models: list[RemoteModel] = []
        self._build_ui()
        self._refresh_slots()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(
            QLabel(
                "화자 병합 제안 기능이 사용할 모델입니다. 지정하지 않으면 자동으로 고릅니다.\n"
                "(우선순위: 무료 Gemini → Gemini → Claude → OpenAI → xAI, .env에 키가 있는 것만 표시)"
            )
        )

        slot_row = QHBoxLayout()
        slot_row.addWidget(QLabel("제공자:"))
        self.slot_combo = QComboBox()
        self.slot_combo.currentIndexChanged.connect(self._on_slot_changed)
        slot_row.addWidget(self.slot_combo, stretch=1)
        layout.addLayout(slot_row)

        self.current_label = QLabel()
        layout.addWidget(self.current_label)

        fetch_row = QHBoxLayout()
        self.fetch_btn = QPushButton("실시간 모델 목록 불러오기")
        self.fetch_btn.clicked.connect(self._on_fetch_clicked)
        fetch_row.addWidget(self.fetch_btn)
        fetch_row.addStretch()
        layout.addLayout(fetch_row)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.model_list = QListWidget()
        layout.addWidget(self.model_list, stretch=1)

        action_row = QHBoxLayout()
        self.save_btn = QPushButton("선택한 모델로 저장")
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.reset_btn = QPushButton("자동 선택으로 초기화")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        action_row.addWidget(self.save_btn)
        action_row.addWidget(self.reset_btn)
        layout.addLayout(action_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _refresh_slots(self) -> None:
        self.slot_combo.blockSignals(True)
        self.slot_combo.clear()
        slots = configured_slots()
        if not slots:
            self.slot_combo.addItem("(.env에 설정된 LLM 키가 없습니다)", userData=None)
            self.slot_combo.blockSignals(False)
            self._set_controls_enabled(False)
            self.current_label.setText("")
            return
        for slot in slots:
            self.slot_combo.addItem(SLOT_LABELS.get(slot, slot), userData=slot)
        self.slot_combo.blockSignals(False)
        self._set_controls_enabled(True)
        self._update_current_label()

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.fetch_btn.setEnabled(enabled)
        self.model_list.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.reset_btn.setEnabled(enabled)

    def _current_slot(self) -> str | None:
        return self.slot_combo.currentData()

    def _on_slot_changed(self, _index: int) -> None:
        self.model_list.clear()
        self._models = []
        self.status_label.setText("")
        self._update_current_label()

    def _update_current_label(self) -> None:
        slot = self._current_slot()
        if not slot:
            self.current_label.setText("")
            return
        selected = load_selected_model(slot)
        self.current_label.setText(
            f"현재 선택된 모델: {selected}" if selected else "현재 선택된 모델: (자동 선택, 아직 지정 안 됨)"
        )

    def _on_fetch_clicked(self) -> None:
        slot = self._current_slot()
        if not slot:
            return
        resolved = resolve_slot(slot)
        if resolved is None:
            QMessageBox.warning(self, "키 없음", "이 제공자의 API 키를 찾을 수 없습니다.")
            return

        self.fetch_btn.setEnabled(False)
        self.status_label.setText("모델 목록을 불러오는 중...")
        self.model_list.clear()

        self._fetch_worker = _FetchModelsWorker(resolved)
        self._fetch_worker.succeeded.connect(self._on_fetch_succeeded)
        self._fetch_worker.failed.connect(self._on_fetch_failed)
        self._fetch_worker.start()

    def _on_fetch_succeeded(self, models: list[RemoteModel]) -> None:
        self._models = models
        self.fetch_btn.setEnabled(True)
        self.status_label.setText(f"{len(models)}개 모델을 찾았습니다.")
        self.model_list.clear()
        selected = load_selected_model(self._current_slot() or "")
        for m in models:
            label = m.id if m.display_name == m.id else f"{m.id}  —  {m.display_name}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, m.id)
            self.model_list.addItem(item)
            if selected and m.id == selected:
                item.setSelected(True)
                self.model_list.setCurrentItem(item)

    def _on_fetch_failed(self, message: str) -> None:
        self.fetch_btn.setEnabled(True)
        self.status_label.setText("")
        QMessageBox.warning(self, "조회 실패", message)

    def _on_save_clicked(self) -> None:
        slot = self._current_slot()
        if not slot:
            return
        item = self.model_list.currentItem()
        if item is None:
            QMessageBox.information(self, "선택 필요", "먼저 목록에서 모델을 선택해주세요.")
            return
        resolved = resolve_slot(slot)
        if resolved is None:
            QMessageBox.warning(self, "키 없음", "이 제공자의 API 키를 찾을 수 없습니다.")
            return
        model_id = item.data(Qt.ItemDataRole.UserRole)

        self._set_controls_enabled(False)
        self.status_label.setText(f"'{model_id}' 확인 후 저장 중...")

        self._select_worker = _SelectModelWorker(slot, model_id, resolved)
        self._select_worker.succeeded.connect(self._on_save_succeeded)
        self._select_worker.failed.connect(self._on_save_failed)
        self._select_worker.start()

    def _on_save_succeeded(self, model_id: str) -> None:
        self._set_controls_enabled(True)
        self.status_label.setText(f"'{model_id}' 저장 완료.")
        self._update_current_label()

    def _on_save_failed(self, message: str) -> None:
        self._set_controls_enabled(True)
        self.status_label.setText("")
        QMessageBox.warning(self, "저장 실패", message)

    def _on_reset_clicked(self) -> None:
        slot = self._current_slot()
        if not slot:
            return
        clear_selected_model(slot)
        self._update_current_label()
        self.status_label.setText("자동 선택으로 초기화했습니다 (다음 사용 시 자동으로 다시 고릅니다).")
