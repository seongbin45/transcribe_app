"""영상/오디오 파일에서 STT용 오디오(16kHz mono wav)를 추출하고 미디어 정보를 조회."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import AUDIO_EXTENSIONS, SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS

STT_SAMPLE_RATE = 16000

# QThread(백그라운드 스레드)에서 ffmpeg/ffprobe를 subprocess.run()으로 호출하면
# Windows에서 자식 프로세스가 콘솔/표준입력을 상속받으려다 멈추는 경우가 있어,
# 콘솔 창을 만들지 않고 stdin도 완전히 끊어서 실행한다.
# (공개 이름으로 둠 — core/audio_chunk.py도 같은 ffmpeg 호출 방식을 재사용함)
SUBPROCESS_KWARGS = {"stdin": subprocess.DEVNULL}
if os.name == "nt":
    SUBPROCESS_KWARGS["creationflags"] = subprocess.CREATE_NO_WINDOW


class MediaError(RuntimeError):
    pass


@dataclass
class MediaInfo:
    path: Path
    is_video: bool
    duration_sec: float
    sample_rate: int | None
    channels: int | None
    codec: str | None


def find_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(
            f"'{name}' 실행 파일을 찾을 수 없습니다. ffmpeg가 설치되어 PATH에 등록되어 있는지 확인해주세요."
        )
    return path


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def probe_media(path: Path) -> MediaInfo:
    """ffprobe로 파일의 길이/코덱/샘플레이트 등을 조회."""
    if not path.exists():
        raise MediaError(f"파일을 찾을 수 없습니다: {path}")
    if not is_supported(path):
        raise MediaError(f"지원하지 않는 확장자입니다: {path.suffix}")

    ffprobe = find_binary("ffprobe")
    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **SUBPROCESS_KWARGS
    )
    if result.returncode != 0:
        raise MediaError(f"ffprobe 실행 실패: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    has_video_stream = any(s.get("codec_type") == "video" for s in streams)

    duration = float(fmt.get("duration") or (audio_stream or {}).get("duration") or 0.0)
    sample_rate = int(audio_stream["sample_rate"]) if audio_stream and audio_stream.get("sample_rate") else None
    channels = audio_stream.get("channels") if audio_stream else None
    codec = audio_stream.get("codec_name") if audio_stream else None

    return MediaInfo(
        path=path,
        is_video=path.suffix.lower() in VIDEO_EXTENSIONS or has_video_stream,
        duration_sec=duration,
        sample_rate=sample_rate,
        channels=channels,
        codec=codec,
    )


def extract_audio(path: Path, output_dir: Path, sample_rate: int = STT_SAMPLE_RATE) -> Path:
    """STT 엔진 입력용으로 16kHz mono wav를 추출해 output_dir에 저장하고 경로를 반환."""
    if not is_supported(path):
        raise MediaError(f"지원하지 않는 확장자입니다: {path.suffix}")

    ffmpeg = find_binary("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    # 원본 확장자/경로가 달라도 stem이 같으면 같은 출력 파일로 충돌할 수 있어
    # 원본 절대경로 해시를 붙여 구분한다.
    path_hash = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    out_path = output_dir / f"{path.stem}_{path_hash}.wav"

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **SUBPROCESS_KWARGS
    )
    if result.returncode != 0:
        raise MediaError(f"오디오 추출 실패: {result.stderr.strip()[-1000:]}")

    return out_path
