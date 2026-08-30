"""로컬 GPU 가속(NVIDIA CUDA) 사용 가능 여부 판별.

요청: "GPU 로컬 전용(NVIDIA 상위 모델. 현재 사용가능한 모델을 넘어서 앞으로 출시될
모델까지도 판별)". "상위 모델 이름 목록"을 만들어 이름으로 판별하는 방식은 앞으로
나올(아직 이름조차 없는) 새 모델을 절대 따라잡을 수 없다 — 그래서 이름 목록 대신
**실제 능력을 직접 측정**하는 방식으로 설계했다. 새 NVIDIA GPU가 나와도 코드 변경 없이
자동으로 통과한다: 이름을 몰라도 CUDA를 실제로 쓸 수 있고 VRAM이 충분하면 그걸로 끝.

두 단계로 나눠서 확인한다:
  1. **하드웨어 존재 확인**(정보 표시용): PowerShell(Get-CimInstance Win32_VideoController)로
     NVIDIA 어댑터가 있는지, 화면에 보여줄 이름이 뭔지 확인. 이 프로젝트를 개발한 실제
     기기(Intel Arc GPU)에서 돌려보니, WMI의 AdapterRAM 필드가 카드 자체 이름표에는
     "8GB"라고 적힌 카드를 2GB로 잘못 보고하는 걸 실측으로 확인함(4GB 넘는 최신 카드의
     VRAM을 WMI가 정확히 못 읽는다는 건 잘 알려진 문제) — 그래서 VRAM 용량 판정에는
     이 필드를 아예 쓰지 않는다.
  2. **실제 런타임 능력 확인**(진짜 판정 기준): torch.cuda.is_available()과
     ctranslate2.get_cuda_device_count()로 "지금 설치된 PyTorch/ctranslate2가 실제로
     CUDA를 쓸 수 있는지"를 직접 확인하고, 되면 torch가 드라이버에서 직접 읽어온 VRAM
     용량(WMI보다 훨씬 신뢰할 수 있음)이 문턱값 이상인지 본다.

주의: 이 프로젝트의 requirements.txt는 torch/ctranslate2를 CPU 전용 빌드로 고정해뒀다
(다른 버전 호환성 문제 때문 — README "패키지 버전 관련 참고" 절 참고). 그래서 NVIDIA
하드웨어가 있어도 CUDA 지원 빌드를 따로 설치하지 않으면 이 모듈은 "하드웨어는 있지만
소프트웨어가 없다"고 정확히 알려준다(조용히 CPU로 내려가는 대신) — requirements-gpu.txt 참고.
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# large-v3를 GPU에서 float16으로 돌리는 데 필요한 VRAM은 커뮤니티/공식 벤치마크 기준
# 대략 4.7GB 안팎 + 화자분리(pyannote) 동시 로드분 + 여유분을 감안한 보수적인 문턱값.
# 정밀한 과학적 수치가 아니라 "이 정도는 있어야 안전하게 돌아간다"는 실무적 기준이며,
# 실제로 VRAM이 빠듯한 기기에서 OOM이 보고되면 이 값을 조정하면 된다.
MIN_VRAM_GB_FOR_GPU_MODE = 6.0

_SUBPROCESS_KWARGS = {"stdin": subprocess.DEVNULL}
if os.name == "nt":
    _SUBPROCESS_KWARGS["creationflags"] = subprocess.CREATE_NO_WINDOW


@dataclass(frozen=True)
class GpuCapability:
    gpu_names: list[str] = field(default_factory=list)  # WMI가 보고한 그래픽 어댑터 이름들(표시용)
    has_nvidia_hardware: bool = False  # 이름에 "NVIDIA"가 들어간 어댑터가 하나라도 있는지
    cuda_runtime_available: bool = False  # torch.cuda.is_available() and ctranslate2가 CUDA 인식
    vram_gb: float | None = None  # torch가 드라이버에서 직접 읽은 값(WMI 아님)
    qualifies_for_gpu_mode: bool = False  # 최종 판정: 이 값만 보고 GPU 모드를 켜도 됨
    status_message: str = ""  # 판정 이유를 사람이 읽을 수 있게 설명(설정 화면에 그대로 표시)


def _run_powershell(command: str, timeout: int = 15) -> str:
    """PowerShell 명령을 실행해 표준출력을 반환. 실패해도 예외를 던지지 않고 빈 문자열—
    GPU 감지가 안 되면 그냥 "모르겠다"(=하드웨어 정보 없음)로 취급하면 되지, 앱이 죽으면
    안 되기 때문."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **_SUBPROCESS_KWARGS,
        )
        return result.stdout or ""
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("PowerShell 명령 실행 실패(GPU 하드웨어 감지를 건너뜁니다): %s", e)
        return ""


def get_video_controller_names() -> list[str]:
    """Get-CimInstance Win32_VideoController로 시스템의 그래픽 어댑터 이름 목록을 얻는다.

    표시용/사전 필터용이며, VRAM 용량 판정에는 이 결과의 AdapterRAM을 쓰지 않는다
    (모듈 설명 참고 — 실측으로 부정확함을 확인함).
    """
    output = _run_powershell(
        "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def get_system_memory_summary() -> str:
    """systeminfo에서 총 물리 메모리 줄만 뽑아 온다 — 진단 정보 표시용일 뿐, GPU 판정에는
    전혀 쓰지 않는다.

    실제로 이 한국어 Windows 기기에서 돌려보니 systeminfo.exe는 UTF-8이 아니라 콘솔
    OEM 코드페이지(cp949)로 한국어("총 실제 메모리: ...")를 출력했다 — text=True에
    encoding="utf-8"을 주면 전부 깨진 문자로 나오는 걸 직접 확인함. 그래서 원시 바이트를
    받아 cp949로 먼저 디코딩을 시도하고, 실패하거나(영어 로캘 Windows 등) 라벨을 못 찾으면
    utf-8도 시도한다 — 그래도 안 되면 조용히 빈 문자열을 반환한다(표시용일 뿐이라
    실패해도 앱 동작에는 지장 없음).
    """
    try:
        result = subprocess.run(
            ["systeminfo"], capture_output=True, timeout=30, **_SUBPROCESS_KWARGS
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("systeminfo 실행 실패(진단 정보 표시만 생략됨): %s", e)
        return ""

    raw = result.stdout or b""
    for encoding in ("cp949", "utf-8"):
        text = raw.decode(encoding, errors="replace")
        for line in text.splitlines():
            if "총 실제 메모리" in line or "Total Physical Memory" in line:
                return line.strip()
    return ""


def _probe_cuda_runtime() -> tuple[bool, float | None]:
    """지금 설치된 PyTorch/ctranslate2가 실제로 CUDA를 쓸 수 있는지 직접 확인.

    faster-whisper는 ctranslate2를, 화자분리(로컬 pyannote)는 torch를 GPU 실행에 쓰므로
    둘 다 CUDA를 인식해야 "진짜로 GPU 모드가 동작한다"고 볼 수 있다. import 자체가 실패할
    수도 있어(패키지가 없는 극단적인 경우) 방어적으로 처리.
    """
    try:
        import torch
    except ImportError:
        return False, None

    torch_cuda_ok = False
    vram_gb: float | None = None
    try:
        if torch.cuda.is_available():
            torch_cuda_ok = True
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception as e:  # noqa: BLE001 — 드라이버 조회 중 어떤 예외가 나든 "CUDA 없음"으로 취급
        logger.warning("torch.cuda 조회 실패(GPU 없음으로 처리): %s", e)
        return False, None

    if not torch_cuda_ok:
        return False, None

    try:
        import ctranslate2

        ct2_cuda_ok = ctranslate2.get_cuda_device_count() > 0
    except Exception as e:  # noqa: BLE001
        logger.warning("ctranslate2 CUDA 조회 실패(GPU 없음으로 처리): %s", e)
        return False, None

    return (torch_cuda_ok and ct2_cuda_ok), vram_gb


def probe_gpu_capability() -> GpuCapability:
    """GPU 가속 로컬 모드를 제공해도 되는지 최종 판정. 이 함수 하나만 부르면 된다."""
    gpu_names = get_video_controller_names()
    has_nvidia_hardware = any("nvidia" in name.lower() for name in gpu_names)

    cuda_ok, vram_gb = _probe_cuda_runtime()

    if not gpu_names:
        status = "그래픽 어댑터 정보를 확인하지 못했습니다(하드웨어 감지 실패) — 로컬은 CPU만 제공합니다."
        qualifies = False
    elif not has_nvidia_hardware:
        status = f"NVIDIA GPU가 감지되지 않았습니다(현재: {', '.join(gpu_names)}) — 로컬은 CPU만 제공합니다."
        qualifies = False
    elif not cuda_ok:
        status = (
            f"NVIDIA GPU가 감지됐지만({', '.join(gpu_names)}) CUDA 지원 PyTorch/ctranslate2가 "
            "설치되어 있지 않아 GPU 가속을 쓸 수 없습니다(requirements-gpu.txt로 설치해주세요) "
            "— 로컬은 CPU만 제공합니다."
        )
        qualifies = False
    elif vram_gb is None or vram_gb < MIN_VRAM_GB_FOR_GPU_MODE:
        vram_text = f"{vram_gb:.1f}GB" if vram_gb is not None else "확인 불가"
        status = (
            f"NVIDIA GPU가 감지됐지만({', '.join(gpu_names)}) VRAM이 부족합니다"
            f"({vram_text} < 최소 {MIN_VRAM_GB_FOR_GPU_MODE:.0f}GB) — 로컬은 CPU만 제공합니다."
        )
        qualifies = False
    else:
        status = f"GPU 가속 사용 가능: {', '.join(gpu_names)} ({vram_gb:.1f}GB VRAM)"
        qualifies = True

    return GpuCapability(
        gpu_names=gpu_names,
        has_nvidia_hardware=has_nvidia_hardware,
        cuda_runtime_available=cuda_ok,
        vram_gb=vram_gb,
        qualifies_for_gpu_mode=qualifies,
        status_message=status,
    )


_cached_capability: GpuCapability | None = None


def get_cached_capability(force_refresh: bool = False) -> GpuCapability:
    """probe_gpu_capability()는 PowerShell 프로세스를 띄우는 등 몇백ms가 걸릴 수 있어,
    설정 화면을 열 때마다 다시 부르지 않도록 앱 실행 중 한 번만 계산해서 재사용한다.
    force_refresh=True면(예: 사용자가 방금 GPU 드라이버를 설치했을 때) 다시 계산한다.
    """
    global _cached_capability
    if _cached_capability is None or force_refresh:
        _cached_capability = probe_gpu_capability()
    return _cached_capability


def resolve_stt_device_and_compute_type(capability: GpuCapability) -> tuple[str, str]:
    """GPU 판정 결과에 맞춰 faster-whisper(ctranslate2)에 넘길 (device, compute_type)을 고른다.

    GPU 자격이 없으면 기존과 동일하게 CPU+int8. 자격이 있으면 이 기기의 ctranslate2가
    실제로 지원하는 CUDA compute type 목록을 직접 물어보고(하드코딩 안 함 — ctranslate2
    버전/GPU에 따라 지원 목록이 달라질 수 있어서) 그중 속도/메모리 균형이 좋은 순서로 고른다.
    """
    if not capability.qualifies_for_gpu_mode:
        return "cpu", "int8"

    try:
        import ctranslate2

        supported = ctranslate2.get_supported_compute_types("cuda")
    except Exception as e:  # noqa: BLE001
        logger.warning("ctranslate2 CUDA compute type 조회 실패, float32로 폴백: %s", e)
        return "cuda", "float32"

    for preferred in ("float16", "int8_float16", "float32"):
        if preferred in supported:
            return "cuda", preferred
    return "cuda", "float32"
