"""pyannote.audio 기반 로컬 화자분리 엔진.

pyannote/speaker-diarization-3.1, pyannote/segmentation-3.0 두 모델 모두
Hugging Face에서 라이선스 동의가 되어 있어야 하고, Read 권한 토큰이 필요하다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from pyannote.audio import Pipeline

from .diarization_base import DiarizationEngine, SpeakerSegment

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"


class LocalPyannoteEngine(DiarizationEngine):
    def __init__(self, hf_token: str, model: str = DEFAULT_MODEL):
        if not hf_token:
            raise ValueError(
                "Hugging Face 액세스 토큰이 필요합니다. transcribe_app/.env 의 HF_TOKEN 값을 확인해주세요."
            )
        self.pipeline = Pipeline.from_pretrained(model, use_auth_token=hf_token)

    def diarize(
        self,
        wav_path: Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[SpeakerSegment]:
        """progress_callback(내부 단계 이름, 완료 개수, 전체 개수)를 pyannote의 배치 진행마다 호출.

        pyannote.audio Pipeline은 segmentation/embeddings 등 내부 단계별로 hook(step_name,
        step_artifact, file, total, completed)을 호출하는 기능을 지원한다(콘솔에 진행률
        바를 그리는 ProgressHook이 기본 예시). 콘솔 출력 대신 이 콜백으로 GUI에 진행률을
        전달하기 위해 직접 hook 함수를 만들어 넘긴다. GUI의 "로딩 진행률 표시" 요청 대응.
        """
        def _hook(step_name, step_artifact, file=None, total=None, completed=None):
            if progress_callback is None:
                return
            if completed is None:
                completed = total = 1
            progress_callback(step_name, completed, total)

        output = self.pipeline(
            str(wav_path),
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hook=_hook if progress_callback else None,
        )
        # pyannote.audio 4.x는 DiarizeOutput(speaker_diarization=Annotation, ...)을 반환하지만,
        # 이 프로젝트는 (gated 모델 추가 없이 동작하는) 3.x 계열을 사용하므로 Annotation을 그대로 받는다.
        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", output)

        segments = [
            SpeakerSegment(start=turn.start, end=turn.end, speaker=speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        return segments
