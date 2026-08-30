"""단계별(오디오 추출/로컬 STT/로컬 화자분리/API 처리) 처리 속도 프로파일.

메인 화면에 "실행 전 예상 소요 시간" + "실행 중 실시간 갱신"을 보여달라는 요청에 대응하는
모듈. 두 가지를 한다:
  1. estimate_seconds(): 오디오 길이와 (필요하면) 모델 크기로 실행 전 예상 시간을 계산.
  2. record_actual(): 실제 실행이 끝난 뒤 관찰된 속도로 프로파일을 갱신(지수이동평균) —
     다음부터는 "이 사용자의 실제 기기"에서 관찰된 속도로 예상하게 된다.

"처리 속도"는 오디오 1초를 처리하는 데 걸리는 실제 처리 시간(초)의 비율, 즉 실시간
배율(RTF, real-time factor)로 표현한다 — RTF=1.0이면 오디오 길이만큼, RTF=0.5면 오디오
길이의 절반만큼 걸린다는 뜻.

기기(CPU)마다, 그리고 같은 기기라도 그 순간의 부하(다른 프로그램이 CPU를 쓰고 있는지)에
따라 실제 처리 속도가 크게 달라진다 — 이 프로젝트에서 같은 15분(900초) 분량의 오디오를
large-v3로 8번 실측했을 때도 STT가 264.9초~1640.4초로 6배 넘게 차이 났다(원인: 검증
세션 중 동시에 돌아가던 다른 백그라운드 작업의 CPU 경합, README "Day 2~4 확장 교차검증"
절 참고). 그래서 초기 시드값은 이 8번의 실측 평균으로 잡되(정밀한 벤치마크가 아니라
대략적인 출발점일 뿐), 실제 사용자 기기에서 실행할 때마다 관찰값으로 계속 자기보정한다.
"""
from __future__ import annotations

import json
import logging

from .config import APP_ROOT

logger = logging.getLogger(__name__)

PROFILE_PATH = APP_ROOT / "perf_profile.json"

# large-v3 기준 실측 RTF 시드값 — 이 프로젝트 검증 중 같은 900초(15분) 분량 오디오를
# large-v3로 8회 실측한 평균(STT: 264.9~1640.4초, 화자분리: 465.2~1090.6초로 편차가 큼,
# 위 모듈 설명 참고). 다른 모델 크기의 초기값을 만드는 시드로만 쓰고, 실제 실행 후에는
# record_actual()이 저장한 관찰값으로 대체된다.
_MEASURED_LARGE_V3_STT_RTF = 1.1136
_MEASURED_DIARIZE_RTF = 0.8938

# whisper 공개 파라미터 수(small=244M, medium=769M, large-v3=1550M) 비율로 만든 대략적인
# 속도 배율 — 정밀한 벤치마크가 아니라 "작은 모델일수록 빠르다"는 방향성만 반영한 기본값.
# 실제 실행 후에는 각 모델 크기별로 독립적으로 자기보정된다.
_MODEL_SIZE_RELATIVE_SPEED = {
    "small": 244 / 1550,
    "medium": 769 / 1550,
    "large-v3": 1.0,
}

# stt를 제외한 단계의 기본 RTF. diarize는 위 실측 평균을 그대로 쓰고, extract/api는
# 실측 데이터가 없어 대략적인 출발점만 잡음.
_DEFAULT_RTF = {
    "diarize": _MEASURED_DIARIZE_RTF,
    "extract": 0.05,  # 오디오만 추출하는 ffmpeg 재인코딩은 보통 실시간보다 훨씬 빠름
    "api": 0.3,  # AssemblyAI 등 클라우드 처리 — 실측 전까지의 대략적 기본값
}

_EMA_ALPHA = 0.3  # 지수이동평균 가중치 — 최근 실행에 더 비중을 두되 과거 값도 반영


def _default_stt_rtf(model_size: str) -> float:
    ratio = _MODEL_SIZE_RELATIVE_SPEED.get(model_size, 1.0)
    return _MEASURED_LARGE_V3_STT_RTF * ratio


def _profile_key(stage: str, model_size: str | None) -> str:
    return f"{stage}:{model_size}" if model_size else stage


def _read_profile() -> dict:
    if not PROFILE_PATH.exists():
        return {}
    try:
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("%s 파싱 실패, 빈 값으로 취급합니다.", PROFILE_PATH)
        return {}


def _write_profile(data: dict) -> None:
    try:
        PROFILE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        logger.warning("%s 저장 실패(예상 시간 기능에는 지장 없음): %s", PROFILE_PATH, e)


def get_rtf(stage: str, model_size: str | None = None) -> float:
    """이 기기에서 실제 관찰된(없으면 기본) 실시간 배율(RTF)을 반환."""
    data = _read_profile()
    entry = data.get(_profile_key(stage, model_size))
    if entry and entry.get("rtf"):
        return float(entry["rtf"])

    if stage == "stt":
        return _default_stt_rtf(model_size or "large-v3")
    return _DEFAULT_RTF.get(stage, 1.0)


def estimate_seconds(stage: str, audio_duration_sec: float, model_size: str | None = None) -> float:
    """실행 전 예상 소요 시간(초). 최소 1초는 보장(0으로 나오면 UI에서 어색해 보임)."""
    return max(1.0, audio_duration_sec * get_rtf(stage, model_size))


def record_actual(
    stage: str, audio_duration_sec: float, elapsed_sec: float, model_size: str | None = None
) -> None:
    """실제 실행이 끝난 뒤 관찰된 처리 속도로 프로파일을 갱신(지수이동평균).

    다음 실행부터는 이 기기에서 실제로 관찰된 속도로 예상 시간을 계산하게 된다.
    audio_duration_sec이 0 이하면(비정상적인 입력) 기록하지 않는다.
    """
    if audio_duration_sec <= 0:
        return
    observed_rtf = elapsed_sec / audio_duration_sec

    data = _read_profile()
    key = _profile_key(stage, model_size)
    entry = data.get(key) or {}
    prev_rtf = entry.get("rtf")
    new_rtf = observed_rtf if prev_rtf is None else (_EMA_ALPHA * observed_rtf + (1 - _EMA_ALPHA) * prev_rtf)

    data[key] = {
        "rtf": new_rtf,
        "samples": entry.get("samples", 0) + 1,
        "last_observed_rtf": observed_rtf,
    }
    _write_profile(data)
    logger.info(
        "[perf_profile] %s 실측 RTF=%.3f (오디오 %.1fs / 처리 %.1fs) -> 갱신된 평균 RTF=%.3f",
        key, observed_rtf, audio_duration_sec, elapsed_sec, new_rtf,
    )
