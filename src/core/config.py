"""프로그램 전역 설정."""
from dataclasses import dataclass, field
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = APP_ROOT / "models"
OUTPUT_DIR = APP_ROOT / "output"
SETTINGS_PATH = APP_ROOT / "settings.json"

# 지원 확장자 (사용자가 공유한 코덱 표 기준)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

# 언어 선택 화면에서 고를 수 있는 언어 목록 (표시 이름, 코드)
AVAILABLE_LANGUAGES = [
    ("한국어", "ko"),
    ("영어", "en"),
    ("일본어", "ja"),
    ("중국어", "zh"),
    ("스페인어", "es"),
    ("프랑스어", "fr"),
    ("독일어", "de"),
    ("베트남어", "vi"),
]

# "groq"는 화자분리를 자체 지원하지 않는 순수 STT라서, 선택하면 화자분리는 항상
# pyannoteAI(core/engines/pyannoteai_engine.py)와 함께 쓴다(README "10시간 제한/속도
# 대안 조사" 절 참고). 그래서 이 이름 하나가 "Groq STT + pyannoteAI 화자분리" 조합을 뜻한다.
API_PROVIDERS = ["assemblyai", "groq"]


@dataclass
class Settings:
    # 기본 언어: 한국어+영어 혼용 모드. 추후 언어 목록을 확장/전환 가능하도록 리스트로 관리.
    languages: list[str] = field(default_factory=lambda: ["ko", "en"])
    multilingual_mode: bool = False  # True면 언어 제한 없이 전 언어 후보로 감지

    engine_mode: str = "local"  # "local" | "api"
    api_provider: str = "assemblyai"
    whisper_model_size: str = "large-v3"
    diarize_default: bool = True

    # 화자 병합 제안(LLM) 시 다른 벤더 제공자로 독립 재확인해 교집합만 남길지 여부.
    # 기본 켜짐(신뢰성 우선) — 끄면 API 호출이 절반으로 줄지만 단일 모델 판단만 쓰게 됨.
    cross_validate_merges: bool = True

    models_dir: Path = MODELS_DIR
    output_dir: Path = OUTPUT_DIR


DEFAULT_SETTINGS = Settings()
