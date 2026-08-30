"""긴 오디오를 API 요청 크기 제한에 맞게 여러 조각으로 나누는 유틸리티.

Groq처럼 요청당 파일 크기 상한(무료 티어 25MB)이 있는 API에 아주 긴 오디오(수 시간)를
보내려면 미리 잘라서 여러 번 나눠 보내야 한다. STT는(화자분리와 달리) 각 조각을 독립적으로
전사한 뒤 결과 타임스탬프에 "조각이 원본에서 시작하는 시각"만 더해 이어붙이면 되므로,
겹침(overlap) 없이 단순 분할 방식을 쓴다 — 조각 경계에서 문장이 살짝 끊길 수 있다는 건
알려진 한계로 문서화(README 참고).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .audio_extract import SUBPROCESS_KWARGS, MediaError, find_binary, probe_media

# Whisper류 STT는 고음질이 필요 없고, 파일을 작게 유지해야 API 요청당 크기 제한(예: Groq
# 무료 티어 25MB)을 안전하게 지킬 수 있다. 48kbps mono mp3 ≈ 분당 360KB이므로, 45분 조각은
# 약 16.2MB로 25MB에 여유 있게 들어간다(인코딩 오버헤드 감안해도 안전).
DEFAULT_CHUNK_SECONDS = 45 * 60
DEFAULT_BITRATE_KBPS = 48

# ffmpeg segment muxer가 원본 길이가 청크 길이의 배수에 딱 안 맞으면(거의 항상 그럼)
# 경계에서 1초도 안 되는 사실상 빈 조각을 마지막에 더 만들어내는 걸 실제로 확인함
# (3분짜리 파일을 60초 단위로 자르면 3개가 아니라 4개가 나오고, 4번째가 0.7KB였음).
# 내용이 없는 API 호출을 낭비하지 않도록 이보다 짧은 조각은 버린다.
MIN_CHUNK_DURATION_SEC = 1.0


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    offset_sec: float  # 원본 오디오에서 이 조각이 시작하는 시각(초)


def split_audio(
    wav_path: Path,
    output_dir: Path,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    bitrate_kbps: int = DEFAULT_BITRATE_KBPS,
) -> list[AudioChunk]:
    """wav_path를 chunk_seconds 길이의 mono mp3 조각들로 나눠 output_dir에 저장.

    ffmpeg의 segment muxer를 사용 — 원본을 한 번만 훑으면서 순서대로 잘라내므로,
    직접 여러 번 -ss/-t로 잘라내는 것보다 빠르고 조각 순서가 항상 보장된다.
    """
    ffmpeg = find_binary("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "chunk_%04d.mp3"

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(wav_path),
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-ac", "1",
        "-b:a", f"{bitrate_kbps}k",
        "-reset_timestamps", "1",
        str(pattern),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **SUBPROCESS_KWARGS
    )
    if result.returncode != 0:
        raise MediaError(f"오디오 분할 실패: {result.stderr.strip()[-1000:]}")

    chunk_files = sorted(output_dir.glob("chunk_*.mp3"))
    if not chunk_files:
        raise MediaError("오디오 분할 결과 조각이 생성되지 않았습니다.")

    chunks = [AudioChunk(path=f, offset_sec=i * chunk_seconds) for i, f in enumerate(chunk_files)]

    # 경계에서 생기는 거의 빈 마지막 조각(위 상수 설명 참고) 제거. 마지막 조각만 이 문제가
    # 생기는 걸 실제로 확인했지만, 혹시 몰라 전체를 검사한다(비용은 짧은 mp3 프로브라 무시할 만함).
    valid_chunks = [c for c in chunks if probe_media(c.path).duration_sec >= MIN_CHUNK_DURATION_SEC]
    if not valid_chunks:
        raise MediaError("오디오 분할 결과 유효한 조각이 없습니다(원본이 너무 짧을 수 있음).")
    return valid_chunks
