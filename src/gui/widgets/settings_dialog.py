"""설정 화면: 언어 선택, 로컬/API 엔진 전환, API 키 입력.

옵션이 늘어날수록(특히 API 엔진별 키 입력칸) 창이 위아래로 계속 길어진다는 피드백을 받아,
왼쪽 카테고리 목록 + 오른쪽 내용 패널 구조("환경설정" 창에서 흔한 마스터-디테일 패턴)로
바꿨다. 카테고리를 눌러도, 엔진 라디오를 눌러 API 키 입력칸이 바뀌어도 창 높이는 항상
고정 — 새로 나타나는 내용은 옆(패널) 안에서 자리를 바꿀 뿐 창을 늘리지 않는다.

API 엔진은 두 가지: AssemblyAI(한 번의 호출로 전사+화자분리, 최대 10시간)와
Groq + pyannoteAI(Groq가 초고속 STT, pyannoteAI가 화자분리 — 둘을 조합해서 씀,
core/engines/groq_engine.py・pyannoteai_engine.py 참고). 후자를 고르면 키 입력칸이
두 개(Groq, pyannoteAI) 나온다.
"""
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
    QListWidget,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.config import AVAILABLE_LANGUAGES, Settings
from core.gpu_detect import get_cached_capability
from core.secrets import delete_api_key, get_api_key, set_api_key
from gui.constants import MODEL_CHOICES
from gui.widgets.llm_model_dialog import LlmModelDialog

# 창 크기를 이 값으로 고정 — 카테고리를 넘나들거나 엔진 라디오를 바꿔도 내용은
# 항상 이 크기 안(오른쪽 패널 내부)에서만 바뀌고 창 자체는 늘어나지 않는다.
# 처음엔 560을 썼다가 실제 스크린샷으로 API 키 패널이 잘려 보이는 걸 확인 — 원인은
# (1) 라디오 라벨에 줄바꿈을 넣었더니 자동 줄바꿈이 안 되고 가장 긴 줄 기준으로 폭을
# 요구한 것, (2) 로컬 모델 크기 콤보박스의 "large-v3 (권장, 느림·최초 실행 시 다운로드)"
# 항목이 길어서 sizeHint가 390px까지 벌어진 것. 둘 다 고친 뒤 실측 sizeHint에 맞춰
# 최종적으로 이 값으로 정함.
_DIALOG_WIDTH = 860
_DIALOG_HEIGHT = 460


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.setFixedSize(_DIALOG_WIDTH, _DIALOG_HEIGHT)
        self._settings = settings
        self._key_edits: dict[str, QLineEdit] = {}
        self._key_status_labels: dict[str, QLabel] = {}
        self._build_ui()
        self._load_from_settings(settings)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        body = QHBoxLayout()
        root.addLayout(body, stretch=1)

        # --- 왼쪽: 카테고리 목록 ---------------------------------------------
        self.category_list = QListWidget()
        self.category_list.setFixedWidth(130)
        self.category_list.addItems(["언어", "엔진 / API 키", "LLM / 고급"])
        self.category_list.currentRowChanged.connect(self._on_category_changed)
        body.addWidget(self.category_list)

        # --- 오른쪽: 카테고리별 내용 패널 (선택 카테고리만 보이고 창 높이는 고정) ---
        self.pages = QStackedWidget()
        body.addWidget(self.pages, stretch=1)

        self.pages.addWidget(self._build_language_page())
        self.pages.addWidget(self._build_engine_page())
        self.pages.addWidget(self._build_advanced_page())

        # --- 확인/취소 (카테고리와 무관하게 항상 하단 고정) ---------------------
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _build_language_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

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
        layout.addStretch()
        return page

    def _build_engine_page(self) -> QWidget:
        page = QWidget()
        page_layout = QHBoxLayout(page)

        # 왼쪽: 엔진 선택 라디오 + 로컬 모델 크기
        left = QVBoxLayout()
        engine_group = QGroupBox("STT 엔진")
        engine_group.setMaximumWidth(340)  # 오른쪽 key_stack과 균형 맞춤(남는 폭으로 안 늘어나게)
        engine_layout = QVBoxLayout(engine_group)

        # 라디오 라벨은 한 줄로 짧게 두고(왼쪽 칸이 넓어지는 걸 방지), 설명은 툴팁으로 —
        # 원래 라벨에 줄바꿈을 넣어봤다가 라디오버튼이 자동으로 줄바꿈하지 않고 가장 긴
        # 줄 기준으로 폭을 요구해서 오른쪽 API 키 패널이 창 밖으로 잘리는 걸 실제
        # 스크린샷으로 확인하고(STT 엔진 그룹 sizeHint가 528px까지 벌어짐) 이렇게 고침.
        # GPU 가속(2026-08-31 추가): core/gpu_detect.py가 이 기기에서 NVIDIA GPU를 실제로
        # 쓸 수 있다고 판정하면 라디오 라벨/툴팁이 "CPU" 대신 "GPU 가속"으로 안내된다 —
        # 이름 목록으로 특정 GPU 모델을 골라내는 게 아니라 실제 CUDA 사용 가능 여부 +
        # VRAM 용량을 직접 측정한 결과라, 아직 나오지 않은 미래 GPU도 자동으로 통과한다.
        gpu_capability = get_cached_capability()
        if gpu_capability.qualifies_for_gpu_mode:
            self.local_radio = QRadioButton("로컬 (무료, GPU 가속)")
        else:
            self.local_radio = QRadioButton("로컬 (무료, CPU)")
        self.local_radio.setToolTip(
            "faster-whisper + pyannote로 이 컴퓨터에서 처리합니다.\n" + gpu_capability.status_message
        )
        self.assemblyai_radio = QRadioButton("API - AssemblyAI")
        self.assemblyai_radio.setToolTip(
            "유료. 전사+화자분리를 한 번의 호출로 처리합니다. 최대 10시간 길이 제한이 있습니다."
        )
        self.groq_radio = QRadioButton("API - Groq + pyannoteAI")
        self.groq_radio.setToolTip(
            "Groq(초고속 STT) + pyannoteAI(화자분리, 화자 수 힌트 지원)를 조합해서 씁니다.\n"
            "길이 제한은 요청당 크기(무료 티어 25MB)라서 앱이 자동으로 45분 단위 조각으로\n"
            "나눠 보냅니다 — 조각 경계에서 문장이 살짝 끊길 수 있습니다.\n"
            "언어는 조각(45분)마다 자동 감지되며, 로컬 엔진의 25초 단위 재판정만큼\n"
            "촘촘하지는 않습니다."
        )
        for radio in (self.local_radio, self.assemblyai_radio, self.groq_radio):
            radio.toggled.connect(self._on_engine_choice_changed)
            engine_layout.addWidget(radio)

        model_row = QVBoxLayout()
        model_row.addWidget(QLabel("로컬 모델 크기:"))
        self.model_combo = QComboBox()
        for label, _value in MODEL_CHOICES:
            self.model_combo.addItem(label)
        # 항목 중 "large-v3 (권장, 느림·최초 실행 시 다운로드)"가 길어서 combo box의
        # sizeHint가 390px까지 벌어지는 바람에 옆의 API 키 패널이 창 밖으로 밀려 잘리는
        # 걸 실제로 확인했음 — 닫힌 상태 폭은 좁게 고정하고(길면 말줄임표), 펼치면
        # 목록에서는 전체 텍스트가 그대로 보인다.
        self.model_combo.setMaximumWidth(190)
        self.model_combo.setToolTip(self.model_combo.currentText())
        self.model_combo.currentTextChanged.connect(self.model_combo.setToolTip)
        model_row.addWidget(self.model_combo)
        engine_layout.addLayout(model_row)

        left.addWidget(engine_group)
        left.addStretch()
        page_layout.addLayout(left)

        # 오른쪽: 선택한 엔진에 필요한 API 키 입력칸만 옆에 표시(스택 전환 — 라디오를
        # 바꿔도 이 패널 자리만 바뀔 뿐 페이지/창 높이는 그대로).
        self.key_stack = QStackedWidget()
        self.key_stack.setFixedWidth(340)
        page_layout.addWidget(self.key_stack)

        self.local_key_page = QLabel("로컬 엔진은 API 키가 필요 없습니다.")
        self.local_key_page.setWordWrap(True)
        self.local_key_page.setStyleSheet("color: gray;")
        self.key_stack.addWidget(self._wrap_top(self.local_key_page))

        self.assemblyai_key_group = self._build_key_section("assemblyai", "AssemblyAI")
        self.key_stack.addWidget(self._wrap_top(self.assemblyai_key_group))

        groq_keys_page = QWidget()
        groq_keys_layout = QVBoxLayout(groq_keys_page)
        groq_keys_layout.setContentsMargins(0, 0, 0, 0)
        groq_keys_layout.addWidget(self._build_key_section("groq", "Groq"))
        groq_keys_layout.addWidget(self._build_key_section("pyannoteai", "pyannoteAI"))
        self.key_stack.addWidget(self._wrap_top(groq_keys_page))

        return page

    @staticmethod
    def _wrap_top(widget: QWidget) -> QWidget:
        """위쪽에 붙여 놓기 위한 래퍼(안 그러면 QStackedWidget 안에서 세로 가운데 정렬됨)."""
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        layout.addStretch()
        return wrapper

    def _build_advanced_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

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

        self.diarize_default_check = QCheckBox("전사 시작 시 기본적으로 화자분리 포함")
        layout.addWidget(self.diarize_default_check)

        layout.addStretch()
        return page

    def _build_key_section(self, provider: str, display_name: str) -> QGroupBox:
        """provider 하나에 대한 "상태 표시 + 키 입력 + 표시/저장/삭제" 묶음을 만든다.
        AssemblyAI는 이 묶음 하나, Groq+pyannoteAI는 이 묶음을 두 개(provider별로) 쓴다.
        """
        group = QGroupBox(f"{display_name} API 키")
        group_layout = QVBoxLayout(group)

        status_label = QLabel()
        group_layout.addWidget(status_label)

        key_row = QHBoxLayout()
        key_edit = QLineEdit()
        key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        key_edit.setPlaceholderText("새 API 키 입력 (교체할 때만)")
        key_row.addWidget(key_edit)

        show_btn = QPushButton("표시")
        show_btn.setCheckable(True)
        show_btn.toggled.connect(
            lambda checked, e=key_edit, b=show_btn: self._on_toggle_show_key(checked, e, b)
        )
        key_row.addWidget(show_btn)
        group_layout.addLayout(key_row)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("이 키 저장")
        save_btn.clicked.connect(lambda _checked=False, p=provider, e=key_edit: self._on_save_key(p, e))
        delete_btn = QPushButton("저장된 키 삭제")
        delete_btn.clicked.connect(lambda _checked=False, p=provider: self._on_delete_key(p))
        btn_row.addWidget(save_btn)
        btn_row.addWidget(delete_btn)
        group_layout.addLayout(btn_row)

        self._key_edits[provider] = key_edit
        self._key_status_labels[provider] = status_label
        return group

    def _load_from_settings(self, settings: Settings) -> None:
        self.multilingual_check.setChecked(settings.multilingual_mode)
        for code, cb in self.language_checks.items():
            cb.setChecked(code in settings.languages)
        self._on_multilingual_toggled(settings.multilingual_mode)

        if settings.engine_mode == "api" and settings.api_provider == "groq":
            self.groq_radio.setChecked(True)
        elif settings.engine_mode == "api":
            self.assemblyai_radio.setChecked(True)
        else:
            self.local_radio.setChecked(True)
        self._on_engine_choice_changed()

        for i, (_label, value) in enumerate(MODEL_CHOICES):
            if value == settings.whisper_model_size:
                self.model_combo.setCurrentIndex(i)
                break

        self.diarize_default_check.setChecked(settings.diarize_default)
        self.cross_validate_check.setChecked(settings.cross_validate_merges)
        self._refresh_api_status()

        self.category_list.setCurrentRow(0)

    def _refresh_api_status(self) -> None:
        for provider, label in self._key_status_labels.items():
            has_key = bool(get_api_key(provider))
            label.setText("현재 상태: 저장된 키 있음" if has_key else "현재 상태: 저장된 키 없음")

    def _on_category_changed(self, row: int) -> None:
        if row >= 0:
            self.pages.setCurrentIndex(row)

    def _on_engine_choice_changed(self, _checked: bool = False) -> None:
        if self.groq_radio.isChecked():
            self.key_stack.setCurrentIndex(2)
        elif self.assemblyai_radio.isChecked():
            self.key_stack.setCurrentIndex(1)
        else:
            self.key_stack.setCurrentIndex(0)

    def _on_multilingual_toggled(self, checked: bool) -> None:
        for cb in self.language_checks.values():
            cb.setEnabled(not checked)

    def _on_toggle_show_key(self, checked: bool, key_edit: QLineEdit, show_btn: QPushButton) -> None:
        key_edit.setEchoMode(QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password)
        show_btn.setText("숨기기" if checked else "표시")

    def _on_save_key(self, provider: str, key_edit: QLineEdit) -> None:
        key = key_edit.text().strip()
        if not key:
            QMessageBox.warning(self, "입력 필요", "저장할 API 키를 입력해주세요.")
            return
        set_api_key(provider, key)
        key_edit.clear()
        self._refresh_api_status()
        QMessageBox.information(self, "저장 완료", "API 키를 OS 자격 증명 저장소에 저장했습니다.")

    def _on_delete_key(self, provider: str) -> None:
        delete_api_key(provider)
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
        if self.groq_radio.isChecked():
            self._settings.engine_mode = "api"
            self._settings.api_provider = "groq"
        elif self.assemblyai_radio.isChecked():
            self._settings.engine_mode = "api"
            self._settings.api_provider = "assemblyai"
        else:
            self._settings.engine_mode = "local"
        self._settings.whisper_model_size = MODEL_CHOICES[self.model_combo.currentIndex()][1]
        self._settings.diarize_default = self.diarize_default_check.isChecked()
        self._settings.cross_validate_merges = self.cross_validate_check.isChecked()
        return self._settings
