"""시간 표기 포맷 헬퍼 (문서/자막 출력에서 공통으로 사용)."""
from __future__ import annotations


def fmt_hhmmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_ms(seconds: float) -> tuple[int, int, int, int]:
    total_ms = round(max(seconds, 0.0) * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return h, m, s, ms


def fmt_srt_timestamp(seconds: float) -> str:
    h, m, s, ms = _fmt_ms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_vtt_timestamp(seconds: float) -> str:
    h, m, s, ms = _fmt_ms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
