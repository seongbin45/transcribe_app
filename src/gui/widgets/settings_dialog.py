"""설정 화면: 언어 선택, 로컬/API 엔진 전환, API 키 입력."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from core.config import API_PROVIDERS, AVAILABLE_LANGUAGES, Settings
from core.secrets import delete_api_key, get_api_key, set_api_key
from gui.constants import MODEL_CHOICES
from gui.widgets.llm_model_dialog import LlmModelDialog


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.setMinimumWidth(420)
        self._settings = settings
        self._build_ui()
        self._load_from_settings(settings)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- 언어 -------------------------------------------------------
        lang_group = QGroupBox("언어")
        lang_layout = QVBoxLayout(lang_group)

        self.multilingual_check = QCheckBox("모든 언어 자동 감지 (다국어 모드)")
        self.multilingual_check.toggled.connect(self._on_multilingual_toggled)
        lang_layout.addWidget(self.multilingual_check)

        self.language_checks: dict[str, QCheckBox] = {}
        for label, code in AVAILABLE_LANGUAGES:
            cb = QCheckBox(f"{label} ({code})")
            self.language_checks[code] = cb
            lang_layout.addWidget(cb)

        layout.addWidget(lang_group)

        # --- 엔진 ---------------------------------------------------------
        engine_group = QGroupBox("STT 엔진")
        engine_layout = QVBoxLayout(engine_group)

        self.local_radio = QRadioButton("로컬 (faster-whisper + pyannote, 무료, CPU 사용)")
        self.api_radio = QRadioButton("API (AssemblyAI, 유료, 화자분리 포함 한 번에 처리)")
        engine_layout.addWidget(self.local_radio)
        engine_layout.addWidget(self.api_radio)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("로컬 모델 크기:"))
        self.model_combo = QComboBox()
        for label, _value in MODEL_CHOICES:
            self.model_combo.addItem(label)
        model_row.addWidget(self.model_combo)
        engine_layout.addLayout(model_row)

        layout.addWidget(engine_group)

        # --- API 키 -------------------------------------------------------
        api_group = QGroupBox("AssemblyAI API 키")
        api_layout = QVBoxLayout(api_group)

        self.api_status_label = QLabel()
        api_layout.addWidget(self.api_status_label)

        key_row = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("새 API 키 입력 (교체할 때만)")
        key_row.addWidget(self.api_key_edit)

        self.show_key_btn = QPushButton("표시")
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.toggled.connect(self._on_toggle_show_key)
        key_row.addWidget(self.show_key_btn)
        api_layout.addLayout(key_row)

        key_btn_row = QHBoxLayout()
        save_key_btn = QPushButton("이 키 저장")
        save_key_btn.clicked.connect(self._on_save_key)
        delete_key_btn = QPushButton("저장된 키 삭제")
        delete_key_btn.clicked.connect(self._on_delete_key)
        key_btn_row.addWidget(save_key_btn)
        key_btn_row.addWidget(delete_key_btn)
        api_layout.addLayout(key_btn_row)

        layout.addWidget(api_group)

        # --- LLM 모델(화자 병합 제안) ----------------------------------------
        llm_group = QGroupBox("LLM 모델 (화자 병합 제안)")
        llm_layout = QVBoxLayout(llm_group)
        llm_layout.addWidget(
            QLabel("지정하지 않으면 자동으로 고릅니다. 실시간 목록을 보고 직접 고르려면:")
        )
        llm_model_btn = QPushButton("LLM 모델 선택...")
        llm_model_btn.clicked.connect(self._on_open_llm_model_dialog)
        llm_layout.addWidget(llm_model_btn)

        self.cross_validate_check = QCheckBox("다른 제공자로 교차검증 (권장)")
        self.cross_validate_check.setToolTip(
            "병합 제안을 준 제공자와 다른 벤더의 제공자에게 같은 전사록을 독립적으로\n"
            "다시 보여줘서, 두 곳이 모두 동의한 병합만 남깁니다. LLM 판단을 그대로 믿지 않고\n"
            "교차검증하기 위한 기능이라 API 호출이 최대 2배로 늘어납니다.\n"
            "끄면 1차 제공자 판단만 사용합니다(호출은 절반, 신뢰도는 낮아짐)."
        )
        llm_layout.addWidget(self.cross_validate_check)

        layout.addWidget(llm_group)

        # --- 화자분리 기본값 ------------------------------------------------
        self.diarize_default_check = QCheckBox("전사 시작 시 기본적으로 화자분리 포함")
        layout.addWidget(self.diarize_default_check)

        # --- 확인/취소 ------------------------------------------------------
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_from_settings(self, settings: Settings) -> None:
        self.multilingual_check.setChecked(settings.multilingual_mode)
        for code, cb in self.language_checks.items():
            cb.setChecked(code in settings.languages)
        self._on_multilingual_toggled(settings.multilingual_mode)

        if settings.engine_mode == "api":
            self.api_radio.setChecked(True)
        else:
            self.local_radio.setChecked(True)

        for i, (_label, value) in enumerate(MODEL_CHOICES):
            if value == settings.whisper_model_size:
                self.model_combo.setCurrentIndex(i)
                break

        self.diarize_default_check.setChecked(settings.diarize_default)
        self.cross_validate_check.setChecked(settings.cross_validate_merges)
        self._refresh_api_status()

    def _refresh_api_status(self) -> None:
        provider = API_PROVIDERS[0]
        has_key = bool(get_api_key(provider))
        self.api_status_label.setText(
            "현재 상태: 저장된 키 있음" if has_key else "현재 상태: 저장된 키 없음"
        )

    def _on_multilingual_toggled(self, checked: bool) -> None:
        for cb in self.language_checks.values():
            cb.setEnabled(not checked)

    def _on_toggle_show_key(self, checked: bool) -> None:
        self.api_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        self.show_key_btn.setText("숨기기" if checked else "표시")

    def _on_save_key(self) -> None:
        key = self.api_key_edit.text().strip()
        if not key:
            QMessageBox.warning(self, "입력 필요", "저장할 API 키를 입력해주세요.")
            return
        set_api_key(API_PROVIDERS[0], key)
        self.api_key_edit.clear()
        self._refresh_api_status()
        QMessageBox.information(self, "저장 완료", "API 키를 OS 자격 증명 저장소에 저장했습니다.")

    def _on_delete_key(self) -> None:
        delete_api_key(API_PROVIDERS[0])
        self._refresh_api_status()

    def _on_open_llm_model_dialog(self) -> None:
        dialog = LlmModelDialog(self)
        dialog.exec()

    def result_settings(self) -> Settings:
        languages = [code for code, cb in self.language_checks.items() if cb.isChecked()]
        if not languages:
            languages = ["ko", "en"]

        self._settings.languages = languages
        self._settings.multilingual_mode = self.multilingual_check.isChecked()
        self._settings.engine_mode = "api" if self.api_radio.isChecked() else "local"
        self._settings.whisper_model_size = MODEL_CHOICES[self.model_combo.currentIndex()][1]
        self._settings.diarize_default = self.diarize_default_check.isChecked()
        self._settings.cross_validate_merges = self.cross_validate_check.isChecked()
        return self._settings
