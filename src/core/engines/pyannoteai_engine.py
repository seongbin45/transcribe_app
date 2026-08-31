"""pyannoteAI(클라우드 GPU 호스팅) 기반 화자분리 엔진.

로컬에서 쓰는 pyannote.audio(core/engines/local_pyannote.py)와 같은 개발사(pyannoteAI)의
클라우드 API — 같은 min/max 화자 수 힌트를 그대로 지원해서, 로컬에서 실측으로 검증된
"화자 수 힌트" 기능(README "화자 수 힌트 기능" 절 참고)을 이 엔진에서도 그대로 쓸 수 있다.
GPU에서 돌아가므로(pyannoteAI 공식 문서: "self-hosted도 GPU 필수, CPU는 사실상 불가능")
로컬 CPU보다 훨씬 빠를 것으로 기대 — 실제 배율은 README에 실측치로 기록 예정.

동작 방식(REST API, 비동기 잡):
  1) POST /v1/media/input 으로 업로드용 사전 서명 URL을 받아 오디오를 PUT으로 업로드
  2) POST /v1/diarize 에 그 media URL과 화자 수 힌트를 담아 잡을 생성
  3) GET /v1/jobs/{jobId} 로 잡이 끝날 때까지 폴링
  4) 결과(화자 라벨/시작/끝 목록)를 SpeakerSegment로 변환
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable

import requests

from .diarization_base import DiarizationEngine, SpeakerSegment

BASE_URL = "https://api.pyannote.ai"
POLL_INTERVAL_SEC = 3
POLL_TIMEOUT_SEC = 3600
UPLOAD_TIMEOUT_SEC = 900
JOB_POLL_REQUEST_TIMEOUT_SEC = 45  # 실사용 중 30초에서 read timeout이 실제로 발생해 여유를 둠(아래 참고)

# 실사용 중(2026-08-31) 10시간 넘는 파일을 여러 시간 폴링하다가 도중에 단 한 번의
# 네트워크 순간 지연(Read timed out)만으로 전체 작업이 실패로 끝나는 걸 실제로 겪음.
# 폴링은 최대 1200번(3600초 / 3초)까지 반복 호출되므로, 그 긴 시간 동안 일시적인 지연이
# 한 번도 없을 거라고 가정하는 게 오히려 비현실적 — 그래서 연결 오류/타임아웃처럼 재시도로
# 해결될 수 있는 예외만 지수 백오프로 몇 번 더 시도한 뒤에도 안 되면 그때 포기한다
# (서버가 4xx/5xx를 명시적으로 준 경우는 재시도해도 소용없으므로 그대로 즉시 실패 처리).
MAX_TRANSIENT_RETRIES = 5
RETRY_BACKOFF_BASE_SEC = 3


class PyannoteAIError(RuntimeError):
    pass


def _request_with_retries(
    method: str,
    url: str,
    *,
    progress_callback: Callable[[str, int, int], None] | None = None,
    max_retries: int = MAX_TRANSIENT_RETRIES,
    **kwargs,
) -> requests.Response:
    """일시적 네트워크 오류(타임아웃/연결 끊김)에 한해서만 지수 백오프로 재시도.

    서버가 실제로 응답은 했지만 4xx/5xx인 경우는 재시도해도 결과가 바뀌지 않으므로
    여기서 건드리지 않고 그대로 반환한다 — 상태 코드 판정은 호출부에서 기존처럼 처리.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return requests.request(method, url, **kwargs)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt == max_retries:
                break
            wait = RETRY_BACKOFF_BASE_SEC * attempt
            if progress_callback:
                progress_callback(f"네트워크 오류, {wait}초 후 재시도 중 ({attempt}/{max_retries}회)", 0, 1)
            time.sleep(wait)
    raise PyannoteAIError(
        f"pyannoteAI 요청이 네트워크 오류로 {max_retries}회 재시도 후에도 실패했습니다: {last_exc}"
    ) from last_exc


class PyannoteAIEngine(DiarizationEngine):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("pyannoteAI API 키가 필요합니다. 설정 화면에서 입력해주세요.")
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def diarize(
        self,
        wav_path: Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[SpeakerSegment]:
        if progress_callback:
            progress_callback("업로드 중", 0, 1)
        media_url = self._upload(wav_path, progress_callback)

        if progress_callback:
            progress_callback("화자분리 요청 중", 0, 1)

        # 실제로 API를 호출해보니(2026-08-31), minSpeakers/maxSpeakers/numSpeakers는
        # model을 명시적으로 "precision-2"로 지정해야만 받아준다 — 기본 모델(community-1)로
        # 두면 400 오류("available only when model is 'precision-2'")가 남. precision-2가
        # 정확도도 더 높다고 조사됐으니(Precision-1 대비 +14%, 로컬 오픈소스 대비 +28%),
        # 화자 수 힌트를 쓰든 안 쓰든 이 엔진은 항상 precision-2를 명시한다.
        payload: dict = {"url": media_url, "model": "precision-2"}
        # num_speakers(정확한 인원 수)가 있으면 그걸 우선 쓰고, 없으면 min/max 범위 힌트만
        # 있는 대로 넣는다 — 로컬 pyannote와 동일한 우선순위 규칙(core/local_pyannote.py 참고).
        if num_speakers is not None:
            payload["numSpeakers"] = num_speakers
        else:
            if min_speakers is not None:
                payload["minSpeakers"] = min_speakers
            if max_speakers is not None:
                payload["maxSpeakers"] = max_speakers

        resp = _request_with_retries(
            "post",
            f"{BASE_URL}/v1/diarize",
            headers=self._headers,
            json=payload,
            timeout=60,
            progress_callback=progress_callback,
        )
        if resp.status_code >= 400:
            raise PyannoteAIError(f"pyannoteAI 화자분리 요청 실패({resp.status_code}): {resp.text[:500]}")

        body = resp.json()
        job_id = body.get("jobId")
        if not job_id:
            raise PyannoteAIError(f"pyannoteAI 응답에 jobId가 없습니다: {body}")

        output = self._poll(job_id, progress_callback)
        return [
            SpeakerSegment(start=float(seg["start"]), end=float(seg["end"]), speaker=str(seg["speaker"]))
            for seg in output
        ]

    def _upload(
        self, wav_path: Path, progress_callback: Callable[[str, int, int], None] | None = None
    ) -> str:
        # media:// 스킴의 임의 고유 키 — pyannoteAI가 내부적으로 이 키에 대응하는
        # 사전 서명 업로드 URL을 발급해준다(48시간 뒤 자동 삭제).
        media_key = f"media://transcribe-app-{uuid.uuid4().hex}"
        resp = _request_with_retries(
            "post",
            f"{BASE_URL}/v1/media/input",
            headers=self._headers,
            json={"url": media_key},
            timeout=30,
            progress_callback=progress_callback,
        )
        if resp.status_code >= 400:
            raise PyannoteAIError(f"pyannoteAI 업로드 URL 요청 실패({resp.status_code}): {resp.text[:500]}")

        presigned_url = resp.json().get("url")
        if not presigned_url:
            raise PyannoteAIError("pyannoteAI가 업로드용 URL을 내려주지 않았습니다.")

        # PUT은 파일 핸들을 스트리밍으로 소비하므로(재시도 시 새로 열어야 함) 위
        # _request_with_retries 헬퍼를 그대로 못 쓰고 직접 재시도 루프를 돈다.
        last_exc: Exception | None = None
        for attempt in range(1, MAX_TRANSIENT_RETRIES + 1):
            try:
                with open(wav_path, "rb") as f:
                    put_resp = requests.put(presigned_url, data=f, timeout=UPLOAD_TIMEOUT_SEC)
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exc = e
                if attempt == MAX_TRANSIENT_RETRIES:
                    raise PyannoteAIError(
                        f"pyannoteAI로 오디오 업로드가 네트워크 오류로 {MAX_TRANSIENT_RETRIES}회 재시도 후에도 "
                        f"실패했습니다: {e}"
                    ) from e
                wait = RETRY_BACKOFF_BASE_SEC * attempt
                if progress_callback:
                    progress_callback(f"업로드 네트워크 오류, {wait}초 후 재시도 중 ({attempt}/{MAX_TRANSIENT_RETRIES}회)", 0, 1)
                time.sleep(wait)
        if put_resp.status_code >= 400:
            raise PyannoteAIError(f"pyannoteAI로 오디오 업로드 실패({put_resp.status_code})")

        return media_key

    def _poll(self, job_id: str, progress_callback: Callable[[str, int, int], None] | None = None) -> list[dict]:
        elapsed = 0
        while elapsed < POLL_TIMEOUT_SEC:
            # 폴링은 (최대 1시간 / 3초 간격 =) 최대 1200번까지 반복 호출되므로, 그 동안
            # 일시적인 네트워크 지연이 한 번쯤 있는 게 오히려 자연스럽다 — 실사용 중
            # 실제로 read timeout 한 번에 여러 시간 걸린 작업 전체가 실패하는 걸 겪어서
            # (아래 참고), 연결 오류/타임아웃만 재시도로 흡수한다.
            resp = _request_with_retries(
                "get",
                f"{BASE_URL}/v1/jobs/{job_id}",
                headers=self._headers,
                timeout=JOB_POLL_REQUEST_TIMEOUT_SEC,
                progress_callback=progress_callback,
            )
            if resp.status_code >= 400:
                raise PyannoteAIError(f"pyannoteAI 작업 조회 실패({resp.status_code}): {resp.text[:500]}")

            data = resp.json()
            status = data.get("status")
            if status == "succeeded":
                # 실제로 호출해보니(2026-08-31) 공개 문서 예시와 달리 output이 세그먼트
                # 리스트를 바로 담고 있지 않고 {"diarization": [...]}로 한 겹 더 감싸져
                # 있었음 — 실물 응답을 직접 찍어보고 확인한 뒤 고침.
                output = data.get("output") or {}
                return output.get("diarization") or []
            if status in ("failed", "canceled"):
                raise PyannoteAIError(f"pyannoteAI 화자분리 실패: {data.get('warning') or status}")

            if progress_callback:
                progress_callback(f"처리 중 ({status})", elapsed, POLL_TIMEOUT_SEC)
            time.sleep(POLL_INTERVAL_SEC)
            elapsed += POLL_INTERVAL_SEC

        raise PyannoteAIError("화자분리 대기 시간이 초과되었습니다.")
