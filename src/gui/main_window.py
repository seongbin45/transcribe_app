"""파일을 선택 -> 오디오 추출 -> STT(로컬 faster-whisper 또는 API) + 화자분리로 전사, 문서로 내보내는 GUI."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.align import SpeakerTranscriptSegment, assign_speakers
from core.audio_extract import MediaError, MediaInfo, extract_audio, is_supported, probe_media
from core.config import API_PROVIDERS, DEFAULT_SETTINGS, SUPPORTED_EXTENSIONS, Settings
from core.engines.assemblyai_engine import AssemblyAIEngine
from core.engines.base import TranscriptSegment
from core.engines.local_pyannote import LocalPyannoteEngine
from core.engines.local_whisper import LocalWhisperEngine
from core.exporters.docx_exporter import export_docx
from core.exporters.markdown_exporter import export_markdown, export_txt
from core.exporters.subtitle_exporter import export_srt, export_vtt
from core.llm_refine import apply_merges, get_provider_candidates, suggest_merges
from core.secrets import get_api_key, get_hf_token
from core.settings_store import load_settings, save_settings
from gui.constants import EXPORT_FORMATS, MODEL_CHOICES
from gui.widgets.merge_review_dialog import MergeReviewDialog
from gui.widgets.settings_dialog import SettingsDialog


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_info(info: MediaInfo) -> str:
    kind = "영상" if info.is_video else "오디오"
    return (
        f"종류: {kind}\n"
        f"길이: {_format_duration(info.duration_sec)}\n"
        f"오디오 코덱: {info.codec or '알 수 없음'}\n"
        f"샘플레이트: {info.sample_rate or '알 수 없음'} Hz\n"
        f"채널 수: {info.channels or '알 수 없음'}"
    )


def _relabel_speakers(segments: list[SpeakerTranscriptSegment]) -> None:
    """SPEAKER_00 같은 내부 라벨을 등장 순서대로 '화자 1', '화자 2'... 로 바꾼다."""
    label_map: dict[str, str] = {}
    for seg in segments:
        if seg.speaker not in label_map and seg.speaker != "화자 미상":
            label_map[seg.speaker] = f"화자 {len(label_map) + 1}"
    for seg in segments:
        seg.speaker = label_map.get(seg.speaker, seg.speaker)


def _format_transcript(segments: list[TranscriptSegment] | list[SpeakerTranscriptSegment]) -> str:
    if not segments:
        return "(감지된 음성이 없습니다)"
    lines = []
    for seg in segments:
        start = _format_duration(seg.start)
        end = _format_duration(seg.end)
        speaker = getattr(seg, "speaker", None)
        prefix = f"[{start}-{end}] "
        if speaker:
            prefix += f"{speaker} "
        lines.append(f"{prefix}({seg.language} {seg.language_probability * 100:.0f}%) {seg.text}")
    return "\n".join(lines)


class ExtractWorker(QThread):
    succeeded = Signal(object, object)  # MediaInfo, Path(wav)
    failed = Signal(str)

    def __init__(self, path: Path, output_dir: Path):
        super().__init__()
        self.path = path
        self.output_dir = output_dir

    def run(self) -> None:
        try:
            info = probe_media(self.path)
            wav_path = extract_audio(self.path, self.output_dir)
            self.succeeded.emit(info, wav_path)
        except MediaError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"예상치 못한 오류: {e}")


class TranscribeWorker(QThread):
    succeeded = Signal(object)  # list[TranscriptSegment] | list[SpeakerTranscriptSegment]
    failed = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        wav_path: Path,
        engine_mode: str,
        model_size: str,
        languages: list[str],
        multilingual_mode: bool,
        diarize: bool,
        api_key: str | None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
        super().__init__()
        self.wav_path = wav_path
        self.engine_mode = engine_mode
        self.model_size = model_size
        self.languages = languages
        self.multilingual_mode = multilingual_mode
        self.diarize = diarize
        self.api_key = api_key
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def run(self) -> None:
        try:
            if self.engine_mode == "api":
                self._run_api()
            else:
                self._run_local()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"전사 중 오류: {e}")

    def _run_local(self) -> None:
        self.status.emit("전사 중... (STT 모델 로드/실행)")
        stt_engine = LocalWhisperEngine(
            model_size=self.model_size,
            device="cpu",
            compute_type="int8",
            download_root=DEFAULT_SETTINGS.models_dir,
        )
        segments = stt_engine.transcribe(
            self.wav_path,
            languages=self.languages,
            multilingual_mode=self.multilingual_mode,
        )

        if not self.diarize:
            self.succeeded.emit(segments)
            return

        self.status.emit("화자분리 중... (pyannote 모델 로드/실행)")
        dia_engine = LocalPyannoteEngine(hf_token=get_hf_token())
        speaker_segments = dia_engine.diarize(
            self.wav_path, min_speakers=self.min_speakers, max_speakers=self.max_speakers
        )
        merged = assign_speakers(segments, speaker_segments)
        _relabel_speakers(merged)
        self.succeeded.emit(merged)

    def _run_api(self) -> None:
        engine = AssemblyAIEngine(api_key=self.api_key or "")
        if self.diarize:
            self.status.emit("API로 전사 + 화자분리 처리 중... (AssemblyAI)")
            segments = engine.transcribe_with_diarization(
                self.wav_path,
                languages=self.languages,
                multilingual_mode=self.multilingual_mode,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
        else:
            self.status.emit("API로 전사 처리 중... (AssemblyAI)")
            segments = engine.transcribe(
                self.wav_path, languages=self.languages, multilingual_mode=self.multilingual_mode
            )
        self.succeeded.emit(segments)


class MergeSuggestWorker(QThread):
    succeeded = Signal(dict, str, str)  # merges, reasoning, provider_used
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, segments: list[SpeakerTranscriptSegment]):
        super().__init__()
        self.segments = segments

    def run(self) -> None:
        try:
            merges, reasoning, provider = suggest_merges(self.segments, status_callback=self.status.emit)
            self.succeeded.emit(merges, reasoning, provider)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"화자 병합 제안 요청 중 오류: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("화자별 전사 프로그램 - 5단계 (설정 포함)")
        self.resize(720, 680)
        self.setAcceptDrops(True)

        self._settings: Settings = load_settings()
        self._selected_path: Path | None = None
        self._wav_path: Path | None = None
        self._worker: ExtractWorker | None = None
        self._transcribe_worker: TranscribeWorker | None = None
        self._merge_worker: MergeSuggestWorker | None = None
        self._last_segments: list[TranscriptSegment] | list[SpeakerTranscriptSegment] = []

        self._build_ui()
        self._apply_settings_to_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        top_row = QHBoxLayout()
        hint = QLabel(
            "영상(.mp4 .mov .mkv .avi .webm) 또는 오디오(.mp3 .wav .flac .m4a .aac .ogg) "
            "파일을 선택하거나 이 창으로 드래그하세요."
        )
        hint.setWordWrap(True)
        top_row.addWidget(hint, stretch=1)
        settings_btn = QPushButton("설정")
        settings_btn.clicked.connect(self._on_open_settings)
        top_row.addWidget(settings_btn)
        layout.addLayout(top_row)

        file_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("선택된 파일 없음")
        browse_btn = QPushButton("파일 선택")
        browse_btn.clicked.connect(self._on_browse)
        file_row.addWidget(self.path_edit)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        self.extract_btn = QPushButton("오디오 추출 및 정보 확인")
        self.extract_btn.setEnabled(False)
        self.extract_btn.clicked.connect(self._on_extract)
        layout.addWidget(self.extract_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMaximumHeight(110)
        self.info_box.setPlaceholderText("여기에 미디어 정보와 추출 결과가 표시됩니다.")
        layout.addWidget(self.info_box)

        self.engine_label = QLabel()
        layout.addWidget(self.engine_label)

        stt_row = QHBoxLayout()
        stt_row.addWidget(QLabel("로컬 모델 크기:"))
        self.model_combo = QComboBox()
        for label, _value in MODEL_CHOICES:
            self.model_combo.addItem(label)
        stt_row.addWidget(self.model_combo)

        self.diarize_check = QCheckBox("화자분리 포함")
        self.diarize_check.setToolTip(
            "화자 라벨(화자 1, 2...)은 이번 실행 안에서만 의미가 있습니다.\n"
            "다른 파일을 처리하면 같은 사람이라도 라벨이 다시 매겨질 수 있습니다\n"
            "(목소리로 신원을 식별/매칭하는 기능은 지원하지 않습니다)."
        )
        stt_row.addWidget(self.diarize_check)

        stt_row.addWidget(QLabel("화자 수(대략, 선택):"))
        self.min_speakers_spin = QSpinBox()
        self.min_speakers_spin.setRange(0, 50)
        self.min_speakers_spin.setSpecialValueText("자동")
        self.min_speakers_spin.setToolTip(
            "예상 화자 수 범위를 대략이라도 알려주면 정확도가 크게 올라갑니다.\n"
            "특히 긴 녹음(1시간 이상)에서 화자 수가 실제보다 훨씬 적게 묶이는 걸 방지합니다.\n"
            "0 = 자동(힌트 없음)"
        )
        stt_row.addWidget(self.min_speakers_spin)
        stt_row.addWidget(QLabel("~"))
        self.max_speakers_spin = QSpinBox()
        self.max_speakers_spin.setRange(0, 50)
        self.max_speakers_spin.setSpecialValueText("자동")
        self.max_speakers_spin.setToolTip(self.min_speakers_spin.toolTip())
        stt_row.addWidget(self.max_speakers_spin)

        self.transcribe_btn = QPushButton("전사 시작")
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.clicked.connect(self._on_transcribe)
        stt_row.addWidget(self.transcribe_btn)
        layout.addLayout(stt_row)

        layout.addWidget(QLabel("전사 결과:"))
        self.transcript_box = QTextEdit()
        self.transcript_box.setReadOnly(True)
        self.transcript_box.setPlaceholderText("전사 결과가 여기에 표시됩니다.")
        layout.addWidget(self.transcript_box)

        merge_row = QHBoxLayout()
        self.merge_suggest_btn = QPushButton("화자 병합 제안 받기 (LLM, 실험적)")
        self.merge_suggest_btn.setEnabled(False)
        self.merge_suggest_btn.setToolTip(
            "transcribe_app/.env 에 GEMINI_FREE_KEY 또는 GEMINI_API_KEY가 필요합니다\n"
            "(무료 키 우선 사용, 실패 시 자동으로 재시도 후 다른 키로 전환).\n"
            "LLM이 문맥을 보고 화자 병합을 '제안'만 하며,\n"
            "제안이 틀릴 수 있어 검토 창에서 직접 체크한 항목만 적용됩니다."
        )
        self.merge_suggest_btn.clicked.connect(self._on_merge_suggest)
        merge_row.addWidget(self.merge_suggest_btn)
        merge_row.addStretch()
        layout.addLayout(merge_row)

        export_row = QHBoxLayout()
        export_row.addWidget(QLabel("내보내기 형식:"))
        self.export_format_combo = QComboBox()
        for label, _fmt, _filter in EXPORT_FORMATS:
            self.export_format_combo.addItem(label)
        export_row.addWidget(self.export_format_combo)

        self.export_btn = QPushButton("파일로 저장")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        export_row.addWidget(self.export_btn)
        layout.addLayout(export_row)

        self.setCentralWidget(central)
        self.statusBar().showMessage("대기 중")

    # --- 설정 반영 -------------------------------------------------------
    def _apply_settings_to_ui(self) -> None:
        s = self._settings

        for i, (_label, value) in enumerate(MODEL_CHOICES):
            if value == s.whisper_model_size:
                self.model_combo.setCurrentIndex(i)
                break
        self.model_combo.setEnabled(s.engine_mode == "local")

        if s.multilingual_mode:
            self.transcribe_btn.setText("전사 시작 (다국어 자동 감지)")
        else:
            self.transcribe_btn.setText(f"전사 시작 ({'+'.join(s.languages)})")

        engine_desc = "로컬 (faster-whisper + pyannote)" if s.engine_mode == "local" else "API (AssemblyAI)"
        self.engine_label.setText(f"현재 엔진: {engine_desc}   (설정에서 변경 가능)")

        self._refresh_diarize_availability()

    def _refresh_diarize_availability(self) -> None:
        s = self._settings
        if s.engine_mode == "api":
            available = bool(get_api_key(API_PROVIDERS[0]))
            tooltip = "AssemblyAI API 키가 설정되어 있지 않습니다. 설정에서 입력해주세요."
        else:
            available = bool(get_hf_token())
            tooltip = "HF_TOKEN이 설정되어 있지 않습니다. transcribe_app/.env 에 Hugging Face 토큰을 추가하세요."

        self.diarize_check.setEnabled(available)
        self.diarize_check.setChecked(available and s.diarize_default)
        self.diarize_check.setToolTip("" if available else tooltip)

    def _on_open_settings(self) -> None:
        dialog = SettingsDialog(self._settings, self)
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            self._settings = dialog.result_settings()
            save_settings(self._settings)
            self._apply_settings_to_ui()

    # --- drag & drop -------------------------------------------------
    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        self._set_selected_path(path)

    # --- actions -------------------------------------------------------
    def _on_browse(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        file_path, _ = QFileDialog.getOpenFileName(
            self, "영상/오디오 파일 선택", "", f"지원 파일 ({exts})"
        )
        if file_path:
            self._set_selected_path(Path(file_path))

    def _set_selected_path(self, path: Path) -> None:
        if not is_supported(path):
            QMessageBox.warning(self, "지원하지 않는 파일", f"지원하지 않는 확장자입니다: {path.suffix}")
            return
        self._selected_path = path
        self._wav_path = None
        self.path_edit.setText(str(path))
        self.extract_btn.setEnabled(True)
        self.transcribe_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.merge_suggest_btn.setEnabled(False)
        self._last_segments = []
        self.info_box.clear()
        self.transcript_box.clear()
        self.statusBar().showMessage(f"선택됨: {path.name}")

    def _on_extract(self) -> None:
        if self._selected_path is None:
            return
        self.extract_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.statusBar().showMessage("오디오 추출 중... (ffmpeg)")

        self._worker = ExtractWorker(self._selected_path, DEFAULT_SETTINGS.output_dir)
        self._worker.succeeded.connect(self._on_extract_done)
        self._worker.failed.connect(self._on_extract_failed)
        self._worker.start()

    def _on_extract_done(self, info: MediaInfo, wav_path: Path) -> None:
        self.progress.setVisible(False)
        self.extract_btn.setEnabled(True)
        self._wav_path = wav_path
        self.transcribe_btn.setEnabled(True)
        self.info_box.setPlainText(_format_info(info) + f"\n\n추출된 STT용 오디오: {wav_path}")
        self.statusBar().showMessage("오디오 추출 완료. 전사를 시작할 수 있습니다.")

    def _on_extract_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.extract_btn.setEnabled(True)
        self.statusBar().showMessage("오류 발생")
        QMessageBox.critical(self, "오류", message)

    def _on_transcribe(self) -> None:
        if self._wav_path is None:
            return
        s = self._settings

        if s.engine_mode == "api" and not get_api_key(API_PROVIDERS[0]):
            QMessageBox.warning(self, "API 키 필요", "설정에서 AssemblyAI API 키를 먼저 입력해주세요.")
            return

        model_size = MODEL_CHOICES[self.model_combo.currentIndex()][1]

        self.transcribe_btn.setEnabled(False)
        self.extract_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.transcript_box.clear()
        if s.engine_mode == "local" and model_size == "large-v3":
            self.statusBar().showMessage("전사 중... (large-v3는 최초 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)")
        else:
            self.statusBar().showMessage("전사 중...")

        min_speakers = self.min_speakers_spin.value() or None
        max_speakers = self.max_speakers_spin.value() or None
        if min_speakers and max_speakers and min_speakers > max_speakers:
            QMessageBox.warning(self, "입력 확인", "최소 화자 수가 최대 화자 수보다 큽니다.")
            return

        self._transcribe_worker = TranscribeWorker(
            self._wav_path,
            engine_mode=s.engine_mode,
            model_size=model_size,
            languages=s.languages,
            multilingual_mode=s.multilingual_mode,
            diarize=self.diarize_check.isChecked(),
            api_key=get_api_key(API_PROVIDERS[0]) if s.engine_mode == "api" else None,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        self._transcribe_worker.succeeded.connect(self._on_transcribe_done)
        self._transcribe_worker.failed.connect(self._on_transcribe_failed)
        self._transcribe_worker.status.connect(self.statusBar().showMessage)
        self._transcribe_worker.start()

    def _on_transcribe_done(self, segments: list[TranscriptSegment]) -> None:
        self.progress.setVisible(False)
        self.transcribe_btn.setEnabled(True)
        self.extract_btn.setEnabled(True)
        self._last_segments = segments
        self.export_btn.setEnabled(bool(segments))
        has_speakers = bool(segments) and hasattr(segments[0], "speaker")
        self.merge_suggest_btn.setEnabled(has_speakers and bool(get_provider_candidates()))
        self.transcript_box.setPlainText(_format_transcript(segments))
        self.statusBar().showMessage(f"전사 완료 ({len(segments)}개 구간)")

    def _on_transcribe_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.transcribe_btn.setEnabled(True)
        self.extract_btn.setEnabled(True)
        self.statusBar().showMessage("오류 발생")
        QMessageBox.critical(self, "오류", message)

    def _on_merge_suggest(self) -> None:
        if not get_provider_candidates():
            QMessageBox.warning(
                self, "API 키 필요", "transcribe_app/.env 에 GEMINI_FREE_KEY 또는 GEMINI_API_KEY가 필요합니다."
            )
            return
        if not self._last_segments:
            return

        self.merge_suggest_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.statusBar().showMessage("LLM에게 화자 병합 제안 요청 중...")

        self._merge_worker = MergeSuggestWorker(self._last_segments)
        self._merge_worker.succeeded.connect(self._on_merge_suggest_done)
        self._merge_worker.failed.connect(self._on_merge_suggest_failed)
        self._merge_worker.status.connect(self.statusBar().showMessage)
        self._merge_worker.start()

    def _on_merge_suggest_done(self, merges: dict, reasoning: str, provider: str) -> None:
        self.progress.setVisible(False)
        self.merge_suggest_btn.setEnabled(True)
        self.statusBar().showMessage(f"화자 병합 제안 도착 (사용된 제공자: {provider})")

        dialog = MergeReviewDialog(self._last_segments, merges, reasoning, self)
        if dialog.exec() == MergeReviewDialog.DialogCode.Accepted:
            approved = dialog.approved_merges()
            if approved:
                self._last_segments = apply_merges(self._last_segments, approved)
                self.transcript_box.setPlainText(_format_transcript(self._last_segments))
                self.statusBar().showMessage(f"{len(approved)}건 병합 적용됨")

    def _on_merge_suggest_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.merge_suggest_btn.setEnabled(True)
        self.statusBar().showMessage("오류 발생")
        QMessageBox.critical(self, "오류", message)

    def _on_export(self) -> None:
        if not self._last_segments:
            return
        label, fmt, file_filter = EXPORT_FORMATS[self.export_format_combo.currentIndex()]
        default_name = (self._selected_path.stem if self._selected_path else "transcript") + f".{fmt}"
        default_dir = str(DEFAULT_SETTINGS.output_dir / default_name)

        save_path, _ = QFileDialog.getSaveFileName(self, f"{label}로 저장", default_dir, file_filter)
        if not save_path:
            return

        title = self._selected_path.stem if self._selected_path else "전사 결과"
        try:
            if fmt == "docx":
                export_docx(self._last_segments, save_path, title=title)
            elif fmt == "md":
                export_markdown(self._last_segments, save_path, title=title)
            elif fmt == "txt":
                export_txt(self._last_segments, save_path)
            elif fmt == "srt":
                export_srt(self._last_segments, save_path)
            elif fmt == "vtt":
                export_vtt(self._last_segments, save_path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "저장 오류", f"파일 저장 중 오류가 발생했습니다: {e}")
            return

        self.statusBar().showMessage(f"저장 완료: {save_path}")
