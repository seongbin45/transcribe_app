"""파일을 선택 -> 오디오 추출 -> STT(로컬 faster-whisper 또는 API) + 화자분리로 전사, 문서로 내보내는 GUI."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QEasingCurve, QThread, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGraphicsDropShadowEffect,
    QGroupBox,
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
from core.config import DEFAULT_SETTINGS, SUPPORTED_EXTENSIONS, Settings
from core.engines.assemblyai_engine import AssemblyAIEngine
from core.engines.base import TranscriptSegment
from core.engines.groq_engine import GroqEngine
from core.engines.local_pyannote import LocalPyannoteEngine
from core.engines.local_whisper import LocalWhisperEngine
from core.engines.pyannoteai_engine import PyannoteAIEngine
from core.exporters.docx_exporter import export_docx
from core.exporters.markdown_exporter import export_markdown, export_txt
from core.exporters.subtitle_exporter import export_srt, export_vtt
from core.gpu_detect import get_cached_capability, resolve_stt_device_and_compute_type
from core.llm_refine import apply_merges, get_provider_candidates, suggest_merges
from core.perf_profile import estimate_seconds, record_actual
from core.secrets import get_api_key, get_hf_token
from core.settings_store import load_settings, save_settings
from gui.constants import EXPORT_FORMATS, MODEL_CHOICES
from gui.widgets.merge_review_dialog import MergeReviewDialog
from gui.widgets.settings_dialog import SettingsDialog


# "다음 할 일" 버튼 깜빡임의 '떠오른' 상태 스타일 — 키캡(keycap)처럼 표면 위로 튀어나와
# 보이도록, 위는 밝고 아래는 진한 그라디언트(빛이 위에서 온다고 가정)로 입체감을 주고,
# 아래쪽에 진한 테두리를 둬서 눌리지 않은 볼록한 모서리처럼 보이게 한다.
# 실제 그림자(box-shadow)는 QSS가 지원하지 않아서 QGraphicsDropShadowEffect로 별도 적용한다.
#
# alpha를 매개변수로 받는 이유: 그림자만 깜빡이니 잘 안 보인다는 피드백을 받아서,
# 버튼 자체의 색(배경/테두리/글자)도 매 애니메이션 프레임마다 이 함수로 새로 만든
# QSS를 다시 적용해 투명해졌다 진해지는 게 눈에 보이게 한다(QSS는 프로퍼티 애니메이션
# 대상이 아니라서, 프레임마다 문자열을 직접 다시 만들어 setStyleSheet 하는 방식을 씀).
def _raised_style(alpha: int) -> str:
    a = max(0, min(255, alpha))
    return f"""
    QPushButton {{
        background-color: qlineargradient(
            x1:0, y1:0, x2:0, y2:1,
            stop:0 rgba(255, 224, 130, {a}),
            stop:0.45 rgba(255, 179, 0, {a}),
            stop:1 rgba(255, 143, 0, {a})
        );
        border: 2px solid rgba(230, 81, 0, {a});
        border-bottom: 3px solid rgba(191, 54, 12, {a});
        border-radius: 10px;
        color: rgba(62, 39, 35, {a});
        font-weight: bold;
        padding: 6px 14px;
    }}
    """


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_duration_kr(seconds: float) -> str:
    """예상 시간 표시용 — "1시간 5분", "3분 12초", "45초"처럼 사람이 읽기 편한 형식."""
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}시간 {m}분"
    if m:
        return f"{m}분 {s}초"
    return f"{s}초"


class _EtaTracker:
    """진행 중인 한 단계(추출/STT/화자분리/API)의 예상 남은 시간을 추적한다.

    실행 전 예상치(perf_profile.estimate_seconds)로 시작해서, 실제 진행률 신호가
    오면(그 신호가 뚝뚝 끊기지 않는 단일 카운터일 때만, on_progress 참고) 지금까지
    실제로 걸린 속도로 전체 예상 시간을 다시 계산 — "실행 중 실시간으로 변동을
    감지해서 예상 시간을 갱신"하는 요청에 대응. 매초 갱신되는 표시는 이 트래커의
    elapsed()/remaining()을 그대로 읽으면 된다.
    """

    def __init__(self, pre_estimate_sec: float):
        self._start = time.monotonic()
        self.estimated_total_sec = pre_estimate_sec

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def on_progress(self, done: int, total: int) -> None:
        """뚝뚝 끊기지 않는 단일 카운터(예: STT 윈도우 처리 수)에만 사용.

        화자분리 내부 단계처럼 done/total이 단계마다 다시 0부터 시작하는 경우 이걸
        호출하면 안 됨 — 대신 사전 예상치 기반 카운트다운을 그대로 쓴다.
        """
        if total > 0 and done > 0:
            fraction = min(1.0, done / total)
            elapsed = self.elapsed()
            self.estimated_total_sec = max(elapsed / fraction, elapsed)

    def remaining(self) -> float:
        return max(0.0, self.estimated_total_sec - self.elapsed())


_STAGE_LABELS = {
    "extract": "오디오 추출",
    "stt": "전사(STT)",
    "diarize": "화자분리",
    "api": "API 처리",
    "groq_stt": "Groq 전사",
    "pyannoteai_diarize": "pyannoteAI 화자분리",
    "stt_gpu": "전사(STT, GPU)",
    "diarize_gpu": "화자분리(GPU)",
}


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
    progress = Signal(int, int)  # (완료, 전체) — 로컬 엔진의 STT/화자분리 단계에서만 실제 값 전달
    stage_started = Signal(str, float)  # (stage: "stt"|"diarize"|"api", 실행 전 예상 시간(초))
    stage_done = Signal(str, float)  # (stage, 실제 걸린 시간(초)) — perf_profile 자기보정용

    def __init__(
        self,
        wav_path: Path,
        engine_mode: str,
        model_size: str,
        languages: list[str],
        multilingual_mode: bool,
        diarize: bool,
        api_provider: str,
        api_keys: dict[str, str],
        audio_duration_sec: float,
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
        # api_provider: "assemblyai" | "groq" (후자는 항상 pyannoteAI와 조합해서 화자분리).
        # api_keys는 provider 이름 -> 키. assemblyai면 {"assemblyai": ...}, groq면
        # {"groq": ..., "pyannoteai": ...} 둘 다 들어있다(gui/widgets/settings_dialog.py에서
        # provider별로 따로 입력받은 것을 그대로 넘김).
        self.api_provider = api_provider
        self.api_keys = api_keys
        self.audio_duration_sec = audio_duration_sec
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
        # GPU 가속(2026-08-31 추가): core/gpu_detect.py가 이 기기에서 NVIDIA GPU를 실제로
        # 쓸 수 있다고 판정했을 때만 device="cuda"가 됨 — 판정 못 받으면 기존과 동일하게
        # CPU(int8)로 동작해 회귀가 없다.
        gpu_capability = get_cached_capability()
        device, compute_type = resolve_stt_device_and_compute_type(gpu_capability)
        # CPU/GPU는 속도가 완전히 달라서 perf_profile 통계가 섞이면 예상 시간이 크게
        # 틀어지므로, stage 이름 자체를 나눈다(core/perf_profile.py 참고).
        stt_stage = "stt_gpu" if device == "cuda" else "stt"
        dia_stage = "diarize_gpu" if device == "cuda" else "diarize"

        stt_pre_estimate = estimate_seconds(stt_stage, self.audio_duration_sec, self.model_size)
        self.stage_started.emit(stt_stage, stt_pre_estimate)
        stt_start = time.monotonic()

        self.status.emit(f"전사 중... (STT 모델 로드/실행, {device.upper()})")
        stt_engine = LocalWhisperEngine(
            model_size=self.model_size,
            device=device,
            compute_type=compute_type,
            download_root=DEFAULT_SETTINGS.models_dir,
        )

        def _stt_progress(done: int, total: int) -> None:
            self.progress.emit(done, total)
            self.status.emit(f"전사 중... ({done}/{total} 구간)")

        segments = stt_engine.transcribe(
            self.wav_path,
            languages=self.languages,
            multilingual_mode=self.multilingual_mode,
            progress_callback=_stt_progress,
        )
        self.stage_done.emit(stt_stage, time.monotonic() - stt_start)

        if not self.diarize:
            self.succeeded.emit(segments)
            return

        dia_pre_estimate = estimate_seconds(dia_stage, self.audio_duration_sec)
        self.stage_started.emit(dia_stage, dia_pre_estimate)
        dia_start = time.monotonic()

        self.status.emit(f"화자분리 중... (pyannote 모델 로드/실행, {device.upper()})")
        dia_engine = LocalPyannoteEngine(hf_token=get_hf_token(), device=device)

        def _dia_progress(step_name: str, done: int, total: int) -> None:
            self.progress.emit(done, total)
            self.status.emit(f"화자분리 중: {step_name} ({done}/{total})")

        speaker_segments = dia_engine.diarize(
            self.wav_path,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            progress_callback=_dia_progress,
        )
        self.stage_done.emit(dia_stage, time.monotonic() - dia_start)
        merged = assign_speakers(segments, speaker_segments)
        _relabel_speakers(merged)
        self.succeeded.emit(merged)

    def _run_api(self) -> None:
        if self.api_provider == "groq":
            self._run_groq_pyannoteai()
        else:
            self._run_assemblyai()

    def _run_assemblyai(self) -> None:
        api_pre_estimate = estimate_seconds("api", self.audio_duration_sec)
        self.stage_started.emit("api", api_pre_estimate)
        api_start = time.monotonic()

        engine = AssemblyAIEngine(api_key=self.api_keys.get("assemblyai") or "")
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
        self.stage_done.emit("api", time.monotonic() - api_start)
        self.succeeded.emit(segments)

    def _run_groq_pyannoteai(self) -> None:
        """Groq(초고속 STT) + pyannoteAI(화자분리)를 조합 — 둘 다 순수 단일 기능이라
        core.align.assign_speakers()로 직접 정렬해서 합친다(로컬 엔진과 동일한 방식)."""
        groq_pre_estimate = estimate_seconds("groq_stt", self.audio_duration_sec, self.model_size)
        self.stage_started.emit("groq_stt", groq_pre_estimate)
        groq_start = time.monotonic()

        self.status.emit("Groq로 전사 중... (초고속 클라우드 STT)")
        stt_engine = GroqEngine(api_key=self.api_keys.get("groq") or "")

        def _groq_progress(done: int, total: int) -> None:
            self.progress.emit(done, total)
            self.status.emit(f"Groq로 전사 중... ({done}/{total} 조각)")

        segments = stt_engine.transcribe(
            self.wav_path,
            languages=self.languages,
            multilingual_mode=self.multilingual_mode,
            progress_callback=_groq_progress,
        )
        self.stage_done.emit("groq_stt", time.monotonic() - groq_start)

        if not self.diarize:
            self.succeeded.emit(segments)
            return

        dia_pre_estimate = estimate_seconds("pyannoteai_diarize", self.audio_duration_sec)
        self.stage_started.emit("pyannoteai_diarize", dia_pre_estimate)
        dia_start = time.monotonic()

        self.status.emit("pyannoteAI로 화자분리 중... (클라우드)")
        dia_engine = PyannoteAIEngine(api_key=self.api_keys.get("pyannoteai") or "")

        def _dia_progress(step_name: str, _done: int, _total: int) -> None:
            # pyannoteAI는 폴링 응답에 실제 진행률(%)이 없어(상태 문자열만 옴), 진행바를
            # 억지로 채우지 않고 상태 문구만 갱신한다 — 남은 시간은 별도 ETA 카운트다운이 담당.
            self.status.emit(f"pyannoteAI 화자분리 중: {step_name}")

        speaker_segments = dia_engine.diarize(
            self.wav_path,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            progress_callback=_dia_progress,
        )
        self.stage_done.emit("pyannoteai_diarize", time.monotonic() - dia_start)
        merged = assign_speakers(segments, speaker_segments)
        _relabel_speakers(merged)
        self.succeeded.emit(merged)


class MergeSuggestWorker(QThread):
    succeeded = Signal(list, str, str, str)  # list[MergeCandidate], reasoning, provider_used, consensus_note
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, segments: list[SpeakerTranscriptSegment], cross_validate: bool = True):
        super().__init__()
        self.segments = segments
        self.cross_validate = cross_validate

    def run(self) -> None:
        try:
            merges, reasoning, provider, consensus_note = suggest_merges(
                self.segments, status_callback=self.status.emit, cross_validate=self.cross_validate
            )
            self.succeeded.emit(merges, reasoning, provider, consensus_note)
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
        self._media_info: MediaInfo | None = None
        self._current_model_size: str | None = None
        self._worker: ExtractWorker | None = None
        self._transcribe_worker: TranscribeWorker | None = None
        self._merge_worker: MergeSuggestWorker | None = None
        self._last_segments: list[TranscriptSegment] | list[SpeakerTranscriptSegment] = []

        # 실행 전/실시간 예상 소요 시간 표시용 — core/perf_profile.py 참고.
        self._eta_tracker: _EtaTracker | None = None
        self._eta_stage: str | None = None
        self._eta_timer = QTimer(self)
        self._eta_timer.setInterval(1000)
        self._eta_timer.timeout.connect(self._update_eta_label)

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
        self.browse_btn = QPushButton("파일 선택")
        self.browse_btn.clicked.connect(self._on_browse)
        file_row.addWidget(self.path_edit)
        file_row.addWidget(self.browse_btn)
        layout.addLayout(file_row)

        self.extract_btn = QPushButton("오디오 추출 및 정보 확인")
        self.extract_btn.setEnabled(False)
        self.extract_btn.clicked.connect(self._on_extract)
        layout.addWidget(self.extract_btn)

        # "지금 뭘 눌러야 하는지" 다음 단계 버튼을 깜빡여서 알려줌(화면 구성 요소가 많아
        # 처음 보면 어디부터 시작할지 헷갈린다는 사용자 피드백 반영). 조건이 아직 안 갖춰졌거나
        # (예: 파일 미선택) 그 단계가 이미 실행됐으면(예: 추출 완료) 깜빡이지 않는다.
        self._blink_effects: dict[str, QGraphicsDropShadowEffect] = {}
        self._blink_animations: dict[str, QVariantAnimation] = {}
        self._start_blink("browse", self.browse_btn)  # 시작 시엔 파일이 선택 안 되어 있음

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        progress_row.addWidget(self.progress, stretch=1)

        # 스피너(진행바) 옆에 실행 전/실시간 예상 소요 시간을 보여줌 — core/perf_profile.py
        # 의 예상치를 실행 시작 시 표시하고, 진행 중에는 1초마다 갱신(_update_eta_label).
        self.eta_label = QLabel()
        self.eta_label.setVisible(False)
        self.eta_label.setStyleSheet("color: gray;")
        progress_row.addWidget(self.eta_label)
        layout.addLayout(progress_row)

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
        stt_row.addStretch()

        self.transcribe_btn = QPushButton("전사 시작")
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.clicked.connect(self._on_transcribe)
        stt_row.addWidget(self.transcribe_btn)
        layout.addLayout(stt_row)

        diarize_group = QGroupBox("화자분리")
        diarize_layout = QVBoxLayout(diarize_group)

        self.diarize_check = QCheckBox("화자분리 포함")
        self.diarize_check.setToolTip(
            "화자 라벨(화자 1, 2...)은 이번 실행 안에서만 의미가 있습니다.\n"
            "다른 파일을 처리하면 같은 사람이라도 라벨이 다시 매겨질 수 있습니다\n"
            "(목소리로 신원을 식별/매칭하는 기능은 지원하지 않습니다)."
        )
        diarize_layout.addWidget(self.diarize_check)

        speaker_count_row = QHBoxLayout()
        self.speaker_count_check = QCheckBox("예상 화자 수를 알고 있어요")
        self.speaker_count_check.setToolTip(
            "화자 수를 대략이라도 알려주면 화자분리 정확도가 크게 올라갑니다.\n"
            "특히 긴 녹음(1시간 이상)에서는 힌트가 없으면 실제보다 화자 수가 훨씬 적게 "
            "뭉뚱그려지는 경우가 많습니다."
        )
        self.speaker_count_check.toggled.connect(self._on_speaker_count_toggled)
        speaker_count_row.addWidget(self.speaker_count_check)

        speaker_count_row.addWidget(QLabel("최소"))
        self.min_speakers_spin = QSpinBox()
        self.min_speakers_spin.setRange(1, 50)
        self.min_speakers_spin.setValue(2)
        self.min_speakers_spin.setSuffix("명")
        speaker_count_row.addWidget(self.min_speakers_spin)

        speaker_count_row.addWidget(QLabel("~ 최대"))
        self.max_speakers_spin = QSpinBox()
        self.max_speakers_spin.setRange(1, 50)
        self.max_speakers_spin.setValue(8)
        self.max_speakers_spin.setSuffix("명")
        speaker_count_row.addWidget(self.max_speakers_spin)

        speaker_count_row.addStretch()
        diarize_layout.addLayout(speaker_count_row)
        layout.addWidget(diarize_group)

        self._on_speaker_count_toggled(False)

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

        if s.engine_mode == "local":
            gpu_capability = get_cached_capability()
            if gpu_capability.qualifies_for_gpu_mode:
                engine_desc = f"로컬 (faster-whisper + pyannote, GPU 가속 — {', '.join(gpu_capability.gpu_names)})"
            else:
                engine_desc = "로컬 (faster-whisper + pyannote, CPU)"
        elif s.api_provider == "groq":
            engine_desc = "API (Groq + pyannoteAI, 초고속)"
        else:
            engine_desc = "API (AssemblyAI)"
        self.engine_label.setText(f"현재 엔진: {engine_desc}   (설정에서 변경 가능)")
        self.engine_label.setToolTip(get_cached_capability().status_message)

        self._refresh_diarize_availability()

    def _refresh_diarize_availability(self) -> None:
        s = self._settings
        if s.engine_mode == "api" and s.api_provider == "groq":
            available = bool(get_api_key("groq")) and bool(get_api_key("pyannoteai"))
            tooltip = "Groq/pyannoteAI API 키가 모두 설정되어 있지 않습니다. 설정에서 둘 다 입력해주세요."
        elif s.engine_mode == "api":
            available = bool(get_api_key("assemblyai"))
            tooltip = "AssemblyAI API 키가 설정되어 있지 않습니다. 설정에서 입력해주세요."
        else:
            available = bool(get_hf_token())
            tooltip = "HF_TOKEN이 설정되어 있지 않습니다. transcribe_app/.env 에 Hugging Face 토큰을 추가하세요."

        self.diarize_check.setEnabled(available)
        self.diarize_check.setChecked(available and s.diarize_default)
        self.diarize_check.setToolTip("" if available else tooltip)

    def _on_speaker_count_toggled(self, checked: bool) -> None:
        self.min_speakers_spin.setEnabled(checked)
        self.max_speakers_spin.setEnabled(checked)

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

    def _start_blink(self, key: str, button: QPushButton) -> None:
        """버튼을 입체(키캡) 스타일로 띄워둔 채, 버튼 자체의 색 투명도와 그림자를 함께
        사인 곡선처럼 부드럽게 오가게 해서 "숨쉬는" 느낌으로 눈에 띄게 한다.

        예전엔 500ms마다 평평한 상태 <-> 입체 상태를 QTimer로 뚝뚝 끊어 전환했는데
        ("깜빡임이 좀 더 부드러웠으면" 하는 피드백을 받아 애니메이션으로 바꿈), 그다음엔
        그림자만 애니메이션되고 버튼 자체는 항상 불투명이라 잘 안 보인다는 피드백을 받아서,
        버튼의 배경/테두리/글자 색 알파도 같은 위상으로 같이 옅어졌다 진해지게 했다
        (QSS는 애니메이션 프로퍼티가 아니라서, 프레임마다 `_raised_style()`로 문자열을
        새로 만들어 `setStyleSheet`을 다시 호출하는 방식).
        """
        self._stop_blink(key, button)  # 이미 돌고 있었다면 깨끗이 정리하고 새로 시작

        effect = QGraphicsDropShadowEffect(button)
        effect.setOffset(0, 5)
        button.setGraphicsEffect(effect)
        self._apply_blink_frame(effect, button, 0.0)  # 애니메이션 첫 틱 전에도 스타일이 비어있지 않도록

        anim = QVariantAnimation(self)
        anim.setDuration(900)
        anim.setStartValue(0.0)
        anim.setKeyValueAt(0.5, 1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        anim.setLoopCount(-1)
        anim.valueChanged.connect(lambda v, eff=effect, btn=button: self._apply_blink_frame(eff, btn, v))
        anim.start()

        self._blink_effects[key] = effect
        self._blink_animations[key] = anim

    def _stop_blink(self, key: str, button: QPushButton) -> None:
        anim = self._blink_animations.pop(key, None)
        if anim is not None:
            anim.stop()
        self._blink_effects.pop(key, None)
        button.setStyleSheet("")
        button.setGraphicsEffect(None)  # 붙어 있던 그림자 이펙트 제거(평평한 기본 버튼으로 복귀)

    @staticmethod
    def _apply_blink_frame(effect: QGraphicsDropShadowEffect, button: QPushButton, phase: float) -> None:
        """phase(0~1, 사인 곡선처럼 부드럽게 오감)에 맞춰 버튼 색과 그림자를 함께 숨쉬듯
        부풀렸다 줄인다 — 버튼 자체가 옅어졌다 진해지는 것과 그림자 번짐이 같이 움직여야
        "깜빡인다"는 게 눈에 뚜렷하게 보인다."""
        button.setStyleSheet(_raised_style(int(70 + phase * 185)))  # 버튼 자체 투명도: 70~255

        effect.setBlurRadius(10 + phase * 22)  # 10~32
        color = QColor(191, 54, 12)
        color.setAlpha(int(90 + phase * 150))  # 90~240
        effect.setColor(color)

    # --- 실행 전/실시간 예상 소요 시간 ------------------------------------------
    def _start_eta(self, stage: str, pre_estimate_sec: float) -> None:
        self._eta_stage = stage
        self._eta_tracker = _EtaTracker(pre_estimate_sec)
        self.eta_label.setVisible(True)
        self._update_eta_label()
        self._eta_timer.start()

    def _stop_eta(self) -> None:
        self._eta_timer.stop()
        self._eta_tracker = None
        self._eta_stage = None
        self.eta_label.setVisible(False)

    def _update_eta_label(self) -> None:
        if self._eta_tracker is None or self._eta_stage is None:
            return
        remaining = self._eta_tracker.remaining()
        elapsed = self._eta_tracker.elapsed()
        stage_label = _STAGE_LABELS.get(self._eta_stage, self._eta_stage)
        remaining_text = "곧 완료..." if remaining <= 0 else f"약 {_format_duration_kr(remaining)} 남음"
        self.eta_label.setText(f"{stage_label} 예상: {remaining_text} (경과 {_format_duration_kr(elapsed)})")

    def _set_selected_path(self, path: Path) -> None:
        if not is_supported(path):
            QMessageBox.warning(self, "지원하지 않는 파일", f"지원하지 않는 확장자입니다: {path.suffix}")
            return
        self._stop_blink("browse", self.browse_btn)
        self._stop_blink("transcribe", self.transcribe_btn)  # 새 파일을 고르면 이전 추출 결과는 무효
        self._stop_blink("export", self.export_btn)  # 이전 전사 결과도 함께 무효
        self._stop_eta()
        self._selected_path = path
        self._wav_path = None
        self._media_info = None
        self.path_edit.setText(str(path))
        self.extract_btn.setEnabled(True)
        self.transcribe_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.merge_suggest_btn.setEnabled(False)
        self._last_segments = []
        self.info_box.clear()
        self.transcript_box.clear()
        self.statusBar().showMessage(f"선택됨: {path.name}")
        # 파일은 골랐지만 아직 추출을 안 했으니, 다음 할 일인 '오디오 추출' 버튼을 깜빡임
        self._start_blink("extract", self.extract_btn)

    def _on_extract(self) -> None:
        if self._selected_path is None:
            return
        self._stop_blink("extract", self.extract_btn)  # 이제 실행되니 깜빡임 중단
        self.extract_btn.setEnabled(False)
        self.progress.setRange(0, 0)  # 추출 단계는 실제 진행률을 얻기 어려워 불확정(스피너) 표시
        self.progress.setVisible(True)
        self.statusBar().showMessage("오디오 추출 중... (ffmpeg)")

        # 추출 자체는 워커 안에서 다시 프로브하지만, 예상 소요 시간을 즉시 보여주려면
        # 메인 스레드에서 가볍게 한 번 더 프로브해서 길이만 미리 알아둔다(ffprobe는
        # 메타데이터만 읽어서 빠름). 프로브가 실패해도(예: 손상된 파일) 워커가 실제
        # 오류를 다시 잡아서 보고하므로, 여기서는 예상 시간 표시만 생략하고 넘어간다.
        try:
            pre_duration = probe_media(self._selected_path).duration_sec
            self._start_eta("extract", estimate_seconds("extract", pre_duration))
        except MediaError:
            pass

        self._worker = ExtractWorker(self._selected_path, DEFAULT_SETTINGS.output_dir)
        self._worker.succeeded.connect(self._on_extract_done)
        self._worker.failed.connect(self._on_extract_failed)
        self._worker.start()

    def _on_extract_done(self, info: MediaInfo, wav_path: Path) -> None:
        self.progress.setVisible(False)
        if self._eta_tracker is not None:
            record_actual("extract", info.duration_sec, self._eta_tracker.elapsed())
        self._stop_eta()
        self.extract_btn.setEnabled(True)
        self._wav_path = wav_path
        self._media_info = info
        self.transcribe_btn.setEnabled(True)
        self.info_box.setPlainText(_format_info(info) + f"\n\n추출된 STT용 오디오: {wav_path}")
        self.statusBar().showMessage("오디오 추출 완료. 전사를 시작할 수 있습니다.")
        # 추출이 끝났고 아직 전사를 안 했으니, 다음 할 일인 '전사 시작' 버튼을 깜빡임
        self._start_blink("transcribe", self.transcribe_btn)

    def _on_extract_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self._stop_eta()
        self.extract_btn.setEnabled(True)
        self.statusBar().showMessage("오류 발생")
        QMessageBox.critical(self, "오류", message)
        # 추출이 실패해서 아직 안 된 상태 그대로이니, 다시 눌러야 함을 알리기 위해 재개
        self._start_blink("extract", self.extract_btn)

    def _on_transcribe(self) -> None:
        if self._wav_path is None:
            return
        s = self._settings

        if s.engine_mode == "api" and s.api_provider == "groq":
            if not get_api_key("groq"):
                QMessageBox.warning(self, "API 키 필요", "설정에서 Groq API 키를 먼저 입력해주세요.")
                return
            if self.diarize_check.isChecked() and not get_api_key("pyannoteai"):
                QMessageBox.warning(
                    self, "API 키 필요", "화자분리를 사용하려면 설정에서 pyannoteAI API 키도 입력해주세요."
                )
                return
        elif s.engine_mode == "api" and not get_api_key("assemblyai"):
            QMessageBox.warning(self, "API 키 필요", "설정에서 AssemblyAI API 키를 먼저 입력해주세요.")
            return

        model_size = MODEL_CHOICES[self.model_combo.currentIndex()][1]

        min_speakers: int | None = None
        max_speakers: int | None = None
        if self.speaker_count_check.isChecked():
            min_speakers = self.min_speakers_spin.value()
            max_speakers = self.max_speakers_spin.value()
            if min_speakers > max_speakers:
                QMessageBox.warning(self, "입력 확인", "최소 화자 수가 최대 화자 수보다 큽니다.")
                return

        # 여기까지가 검증(위에서 return하면 아직 실행 전이므로 버튼/깜빡임/진행바를 건드리지
        # 않는다 — 예전엔 검증보다 먼저 버튼을 비활성화해서, 화자 수 입력이 잘못됐을 때
        # 경고만 뜨고 버튼이 계속 비활성 상태로 남는 문제가 있었음).
        self._stop_blink("transcribe", self.transcribe_btn)  # 이제 실행되니 깜빡임 중단
        self.transcribe_btn.setEnabled(False)
        self.extract_btn.setEnabled(False)
        self.progress.setRange(0, 0)  # 실제 진행률이 들어오기 전까지는 불확정(스피너) 표시
        self.progress.setVisible(True)
        self.transcript_box.clear()
        if s.engine_mode == "local" and model_size == "large-v3":
            self.statusBar().showMessage("전사 중... (large-v3는 최초 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)")
        else:
            self.statusBar().showMessage("전사 중...")

        if s.engine_mode == "api" and s.api_provider == "groq":
            api_keys = {"groq": get_api_key("groq") or "", "pyannoteai": get_api_key("pyannoteai") or ""}
        elif s.engine_mode == "api":
            api_keys = {"assemblyai": get_api_key("assemblyai") or ""}
        else:
            api_keys = {}

        self._current_model_size = model_size
        self._transcribe_worker = TranscribeWorker(
            self._wav_path,
            engine_mode=s.engine_mode,
            model_size=model_size,
            languages=s.languages,
            multilingual_mode=s.multilingual_mode,
            diarize=self.diarize_check.isChecked(),
            api_provider=s.api_provider,
            api_keys=api_keys,
            audio_duration_sec=self._media_info.duration_sec if self._media_info else 0.0,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        self._transcribe_worker.succeeded.connect(self._on_transcribe_done)
        self._transcribe_worker.failed.connect(self._on_transcribe_failed)
        self._transcribe_worker.status.connect(self.statusBar().showMessage)
        self._transcribe_worker.progress.connect(self._on_transcribe_progress)
        self._transcribe_worker.stage_started.connect(self._on_stage_started)
        self._transcribe_worker.stage_done.connect(self._on_stage_done)
        self._transcribe_worker.start()

    def _on_transcribe_progress(self, done: int, total: int) -> None:
        if total <= 0:
            return
        if self.progress.maximum() == 0:  # 아직 불확정 모드면 확정 모드(0~100%)로 전환
            self.progress.setRange(0, 100)
        # pyannote 내부 단계는 배치가 겹쳐서 done이 그 순간의 total을 넘는 경우가 실제로
        # 있었다(예: 192/171) — QProgressBar.setValue()는 범위 밖 값을 그냥 무시해버려서
        # 클램프 없이는 진행률 표시가 깨진 채로 멈추는 걸 확인해서 여기서 직접 클램프한다.
        pct = max(0, min(100, int(done / total * 100)))
        self.progress.setValue(pct)

        # STT 단계(로컬 CPU/GPU "stt"/"stt_gpu", Groq "groq_stt")는 done/total이 뚝뚝
        # 끊기지 않는 단일 카운터(윈도우/조각 처리 수)라 실제 속도로 예상 시간을 다시
        # 계산할 수 있음. 화자분리 내부 단계는 done/total이 단계마다 다시 0부터 시작해
        # 신뢰할 수 없어서, 대신 사전 추정치 기반 카운트다운을 그대로 둔다(_start_eta에서
        # 이미 설정됨) — GPU 화자분리("diarize_gpu")도 로컬 pyannote 구조를 그대로 쓰므로 동일.
        if self._eta_tracker is not None and self._eta_stage in ("stt", "stt_gpu", "groq_stt"):
            self._eta_tracker.on_progress(done, total)
            self._update_eta_label()

    def _on_stage_started(self, stage: str, pre_estimate_sec: float) -> None:
        # 새 단계가 시작될 때마다 진행바를 일단 불확정(스피너) 모드로 되돌린다. 예:
        # Groq STT 단계가 100%로 끝난 뒤 pyannoteAI 화자분리 단계로 넘어가면, 그 단계는
        # 실제 퍼센트를 안 주므로(progress.emit을 호출하지 않음) 리셋을 안 하면 이전
        # 단계의 100%가 그대로 얼어붙은 채 남아 "업로드 중"인데 진행바만 100%로 보이는
        # 문제가 있었다(실사용 중 발견). 실제 진행률이 들어오는 단계(stt/diarize 등)는
        # 곧바로 _on_transcribe_progress가 다시 확정(0~100%) 모드로 바꿔주므로 문제 없다.
        self.progress.setRange(0, 0)
        self._start_eta(stage, pre_estimate_sec)

    def _on_stage_done(self, stage: str, elapsed_sec: float) -> None:
        if self._media_info is not None:
            model_size = self._current_model_size if stage in ("stt", "stt_gpu") else None
            record_actual(stage, self._media_info.duration_sec, elapsed_sec, model_size=model_size)

    def _on_transcribe_done(self, segments: list[TranscriptSegment]) -> None:
        self.progress.setVisible(False)
        self._stop_eta()
        self.transcribe_btn.setEnabled(True)
        self.extract_btn.setEnabled(True)
        self._last_segments = segments
        self.export_btn.setEnabled(bool(segments))
        if segments:
            # 전사가 끝났고 아직 저장을 안 했으니, 다음 할 일인 '파일로 저장' 버튼을 깜빡임
            self._start_blink("export", self.export_btn)
        else:
            self._stop_blink("export", self.export_btn)
        has_speakers = bool(segments) and hasattr(segments[0], "speaker")
        self.merge_suggest_btn.setEnabled(has_speakers and bool(get_provider_candidates()))
        self.transcript_box.setPlainText(_format_transcript(segments))
        self.statusBar().showMessage(f"전사 완료 ({len(segments)}개 구간)")

    def _on_transcribe_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self._stop_eta()
        self.transcribe_btn.setEnabled(True)
        self.extract_btn.setEnabled(True)
        self.statusBar().showMessage("오류 발생")
        QMessageBox.critical(self, "오류", message)
        # 전사가 실패해서 아직 안 된 상태 그대로이니, 다시 눌러야 함을 알리기 위해 재개
        self._start_blink("transcribe", self.transcribe_btn)

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

        self._merge_worker = MergeSuggestWorker(
            self._last_segments, cross_validate=self._settings.cross_validate_merges
        )
        self._merge_worker.succeeded.connect(self._on_merge_suggest_done)
        self._merge_worker.failed.connect(self._on_merge_suggest_failed)
        self._merge_worker.status.connect(self.statusBar().showMessage)
        self._merge_worker.start()

    def _on_merge_suggest_done(
        self, candidates: list, reasoning: str, provider: str, consensus_note: str
    ) -> None:
        self.progress.setVisible(False)
        self.merge_suggest_btn.setEnabled(True)
        self.statusBar().showMessage(f"화자 병합 제안 도착 (1차 제공자: {provider})")

        dialog = MergeReviewDialog(candidates, reasoning, self, consensus_note=consensus_note)
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
        self._stop_blink("export", self.export_btn)  # 저장을 시도하니 일단 멈춤

        label, fmt, file_filter = EXPORT_FORMATS[self.export_format_combo.currentIndex()]
        default_name = (self._selected_path.stem if self._selected_path else "transcript") + f".{fmt}"
        default_dir = str(DEFAULT_SETTINGS.output_dir / default_name)

        save_path, _ = QFileDialog.getSaveFileName(self, f"{label}로 저장", default_dir, file_filter)
        if not save_path:
            self._start_blink("export", self.export_btn)  # 취소했으니 아직 저장 안 된 상태 그대로 재개
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
            self._start_blink("export", self.export_btn)  # 실패했으니 다시 눌러야 함을 알리기 위해 재개
            return

        self.statusBar().showMessage(f"저장 완료: {save_path}")
