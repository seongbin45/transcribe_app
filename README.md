# 화자별 전사(논의록) 프로그램

영상/오디오 파일을 입력하면 오디오를 전사하고 화자별로 분류된 전사/논의록 문서를 생성하는 프로그램.
기본 언어는 한국어+영어 혼용이며, 이후 다른 언어 선택 기능을 추가할 예정.

## 진행 단계

- [x] 1단계: 프로젝트 스캐폴딩 + 오디오 추출(ffmpeg) + 최소 GUI (파일 선택 → 정보 확인)
- [x] 2단계: 로컬 STT(faster-whisper) 연동 + 세그먼트 단위 언어 판정
- [x] 3단계: 화자분리(pyannote.audio) 연동 + STT 결과 정렬
- [x] 4단계: 문서 출력 (docx / md·txt / srt·vtt)
- [x] 5단계: 설정 화면 (언어 선택, 로컬/API 엔진 전환, API 키 입력)

## 지원 확장자

- 영상: `.mp4` `.mov` `.mkv` `.avi` `.webm`
- 오디오: `.mp3` `.wav` `.flac` `.m4a` `.aac` `.ogg`

## 사전 준비

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) 설치 후 PATH 등록 (이 PC에는 이미 설치되어 있음)
- (화자분리용) Hugging Face 계정 + `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0` 두 모델 라이선스 동의 + Read 권한 액세스 토큰. 발급받은 토큰은 `transcribe_app/.env`에 아래처럼 저장 (git에 커밋되지 않음):
  ```
  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```

## 설치 및 실행

```bash
pip install -r requirements.txt
python src/main.py
```

파일을 선택(또는 드래그) → 오디오 추출 → 전사 시작 순서로 진행합니다. `.env`에 `HF_TOKEN`이 있으면 "화자분리 포함" 체크박스가 기본으로 켜져서 전사와 화자분리를 함께 실행합니다.

### 패키지 버전 관련 참고

`requirements.txt`의 torch/torchaudio/huggingface_hub/pyannote.audio 버전은 서로 맞물려 있어 임의로 올리면 깨집니다 (직접 겪은 문제들: pyannote.audio 4.x는 추가 gated 모델을 요구, torchaudio 2.9+는 pyannote가 쓰는 구버전 오디오 API 제거, torch 2.6+는 체크포인트 로드 방식 변경, huggingface_hub 1.x는 `use_auth_token` 인자 제거). 자세한 내용은 requirements.txt 주석 참고.

## 언어 감지 정확도 전략

- **구현됨(2단계)**: `src/core/engines/local_whisper.py`
  1. VAD(Silero, faster-whisper 내장)로 발화 구간을 찾고, 약 25초 단위 윈도우로 묶어서 윈도우마다 언어를 다시 판정 (파일 전체 1회 판정 대신 → 코드스위칭 대응)
  2. 기본 모드(`multilingual_mode=False`)에서는 `all_language_probs`를 한국어/영어 두 후보로만 재정규화해 오판(일본어/중국어 등과 혼동) 감소
  3. `condition_on_previous_text=False`로 윈도우 간 문맥/언어 오염 방지
  4. 세그먼트별 언어 확신도(`language_probability`)를 함께 저장 → GUI에 `(ko 97%)` 형태로 표시
- **검증**: Windows SAPI로 합성한 영어→한국어→영어 혼합 음성으로 테스트 → 언어 전환 지점을 정확히 감지, 두 언어 모두 정확히 전사됨을 확인
- **예정(3단계 이후)**: 화자 턴과 결합해 화자별 주 언어 추정, 신뢰도 낮은 구간 GUI 하이라이트

### GPU 관련 참고

이 PC는 Intel Arc 130V GPU를 사용 중인데, `faster-whisper`(CTranslate2 백엔드)와 `pyannote.audio`(torch 백엔드) 모두 NVIDIA CUDA/CPU만 지원해서 Arc는 활용할 수 없습니다. 현재는 CPU로 동작하며, 추후 Arc 가속이 필요하면 OpenVINO 기반 구현으로 백엔드를 교체하는 방향을 검토할 수 있습니다(STT/화자분리 모두 `STTEngine`/`DiarizationEngine` 인터페이스로 분리되어 있어 교체 가능).

### 문서 출력 (4단계)

- `core/document.py`: 같은 화자가 연속으로 말한 세그먼트를 하나의 문단(턴)으로 묶음 (회의록처럼 읽기 좋게)
- `core/exporters/docx_exporter.py`, `markdown_exporter.py`: 화자 턴 단위로 Word/Markdown/텍스트 출력
- `core/exporters/subtitle_exporter.py`: srt/vtt는 턴으로 묶지 않고 세그먼트 단위 그대로 출력 (영상과 타이밍을 맞추기 위함)
- GUI 하단에서 형식을 골라 "파일로 저장" 가능

### 설정 화면 (5단계)

- 메인 화면 우측 상단 "설정" 버튼으로 열림 ([gui/widgets/settings_dialog.py](transcribe_app/src/gui/widgets/settings_dialog.py))
- **언어**: 기본 언어 목록(한국어/영어/일본어/중국어/스페인어/프랑스어/독일어/베트남어) 중 선택, 또는 "모든 언어 자동 감지" 체크 시 다국어 모드
- **엔진**: 로컬(faster-whisper + pyannote, 무료) ↔ API(AssemblyAI, 유료, 화자분리까지 API 한 번으로 처리) 전환
- **API 키**: 평문 파일이 아니라 OS 자격 증명 저장소(Windows Credential Manager)에 `keyring` 라이브러리로 저장 — 코드나 설정 파일에 키가 남지 않음
- 설정은 `transcribe_app/settings.json`에 저장(언어/엔진/모델 크기 등, git에 커밋되지 않음). API 키는 여기에 저장되지 않고 keyring에만 저장됨

**AssemblyAI 연동 검증**: [core/engines/assemblyai_engine.py](transcribe_app/src/core/engines/assemblyai_engine.py)는 실제 API 키로 검증 완료 (2026-08-29). Day1 자기소개 세션 3분 클립으로 `transcribe()`(paragraphs), `transcribe_with_diarization()`(utterances) 두 경로 모두 실제 응답 스키마와 일치함을 확인. API 키는 설정 화면(keyring) 또는 `.env`의 `ASSEMBLYAI_API_KEY`/`ASSEMBLY_AI_API_KEY`로 인식됨.
- `speech_models: ["universal-3-5-pro", "universal-2"]`를 명시적으로 지정(생략 시 계정 기본값이 구버전일 수 있어서). 다만 실제 응답의 `speech_model_used` 필드를 확인해보니 이 계정에서는 아직 `universal-2`로 폴백되고 있음 — `universal-3-5-pro`가 이 계정에 활성화되지 않은 것으로 보임(요청 자체는 정상). 나중에 계정에 플래그십이 열리면 코드 변경 없이 자동으로 전환됨.
- 로컬 엔진과 달리 언어 감지는 파일 전체 기준 1회로, 코드스위칭 구간별 재판정은 하지 않음
- **화자분리 비교**: 같은 3분 클립에서 로컬 pyannote는 진행자(MC)를 처음부터 끝까지 하나의 화자로 일관되게 유지한 반면, AssemblyAI는 같은 진행자로 보이는 구간을 화자 A/B 두 개로 나누는 차이가 관찰됨 — 실제 음성을 직접 들어봐야 어느 쪽이 맞는지 확정할 수 있어 참고만.

### 심층 교차검증 (2026-08-29)

전체 파이프라인을 실제 데이터와 엣지케이스로 다시 훑으면서 발견/수정한 것들:

- **[버그 수정] 오디오 캐시 파일명 충돌**: `extract_audio()`가 확장자를 뺀 파일명(`stem`)만으로 출력 wav 이름을 만들어서, 확장자만 다른 동명 파일(`meeting.mp4`, `meeting.wav` 등)을 연달아 처리하면 이전 추출 결과가 조용히 덮어써지는 문제 발견. 원본 절대경로 해시를 파일명에 추가해 해결.
- **[버그 수정] 한글 콘솔/경로 환경에서 ffmpeg 출력 디코딩 오류**: `subprocess.run(..., text=True)`가 인코딩을 명시하지 않으면 시스템 로케일(이 PC는 cp949)로 디코딩을 시도하는데, ffmpeg의 UTF-8 출력과 맞지 않아 특정 상황에서 `UnicodeDecodeError`로 조용히 실패(`stdout`이 `None`이 되어 알 수 없는 오류로 표시됨). `encoding="utf-8", errors="replace"`를 명시해 해결.
- **컨테이너 포맷**: 실제 보유 데이터에 없는 `.mp4`/`.mov`/`.avi`를 합성 생성해 확인 — 모두 정상 동작
- **내보내기 엣지케이스**: "화자 미상" 라벨, 0초 길이 세그먼트(자막 최소 표시시간 0.2초 적용됨), 10분 넘는 연속 발화 문단 등 실제 데이터의 지저분한 케이스로 5개 포맷 전부 재검증 — 문제 없음
- **다국어 모드 vs 제한 모드**: 실제 15분 클립 2개(개인녹음/F9ln)를 `multilingual_mode=True`(전체 언어 후보)로 재실행해 비교.
  - 개인녹음(잡음 많은 등록현장음): 애매한 한 단어 발화("Amém.", 00:02:53)가 제한 모드에서는 영어(65%)로 분류됐는데, 다국어 모드에서는 포르투갈어(33%)로 오판 — 언어 후보를 ko/en으로 제한하는 설계가 실제로 오판을 줄여준다는 걸 데이터로 확인
  - F9ln(자기소개, 발음이 명확한 클립): 다국어 모드에서도 52개 세그먼트 전부 한국어로 정확히 감지 — 오판은 신호 품질이 안 좋은 애매한 구간에서만 발생하고, 명확한 발화에서는 언어 제한 여부와 무관하게 정확함을 확인
- **GUI 배선(QTest)**: 실제 버튼 클릭(`QTest.mouseClick`)으로 파일선택→추출→전사→내보내기 전체 흐름 검증, 통과. 테스트 중 이 환경에서 `QTest.qWait()`가 워커 스레드의 신호 전달을 지연/차단시키는 현상을 발견했는데, 실제 앱은 `QTest.qWait()`를 전혀 쓰지 않고 `app.exec()`의 정상 이벤트 루프로 동작하므로 **제품 코드에는 영향 없는 테스트 도구 한정 이슈**로 확인.
- **AssemblyAI**: 위쪽 섹션 참고 — 실제 키로 두 경로 모두 검증 완료.

### Day 2~4 확장 교차검증 (2026-08-29)

Day 1과 같은 방식으로 Day 2~4의 개인녹음/제공파일(각 15분, 총 6개 클립·약 90분 분량)을 large-v3 + pyannote로 전부 처리. 결과:

| 구간 | 화자 수 | 세그먼트 | 언어분포 |
|---|---|---|---|
| Day2 개인녹음 | 3 | 202 | ko 100% |
| Day2 제공파일 | 2 | 170 | ko 100% |
| Day3 개인녹음 | 2 | 84 | ko 100% |
| Day3 제공파일 | 2 | 59 | ko 100% |
| Day4 개인녹음 | 3 | 214 | ko 100% |
| Day4 제공파일 | 3 | 161 | ko 100% |

- 6개 클립 모두 에러 없이 처리 완료, 클립 경계 부근 환각 문구 재발 없음(필터 정상 동작)
- Day3 제공파일은 초반 3분이 무음(마이크 테스트 전 대기시간)이었는데, 실제 결과에서도 "마이크 테스트"가 00:03:36에 정확히 등장 — 별도로 진행한 AssemblyAI 3분 프로브(음량 -91dB로 무음 확인)와 일치
- AssemblyAI 실API로 각 Day의 3분 프로브도 함께 확인: 강사 혼자 계속 말하는 구간은 화자 1명으로, 참가자 간 대화가 있는 구간(Day2 개인녹음)은 화자 3명으로 올바르게 구분 — 로컬 엔진 결과와 방향성 일치

### 화자 수 힌트 기능 (2026-08-29, 사용자 리포트로 발견)

사용자가 Day1 개인녹음 **전체 4시간53분**을 힌트 없이 AssemblyAI(API 엔진)로 처리했더니 하루 종일 화자가 **3명(A/B/C)**으로만 잡히는 문제를 리포트. 실제로는 자기소개 세션에만 8명이 참여했으므로 명백한 과소 분류(under-clustering)였음.

- **원인**: 길고 음향 조건이 다양한 오디오를 화자 수 힌트 없이 완전 자동으로 클러스터링하면, 서로 다른 화자를 같은 라벨로 뭉치는 경향이 있음
- **조치**: 로컬(pyannote `min_speakers`/`max_speakers`, 기존에 엔진은 지원했으나 GUI 미연결)과 API(AssemblyAI `speaker_options.min_speakers_expected`/`max_speakers_expected`, 공식 API 레퍼런스로 정확한 파라미터명 확인 후 신규 추가) 양쪽에 화자 수 힌트를 연결하고, 메인 화면에 "화자 수(대략)" 입력을 추가
- **실측 검증**: 같은 4시간53분 파일을 `min_speakers=10, max_speakers=25`로 재처리 → **화자 3명 → 11명**으로 개선, 처리 425.8초. 예시로 "최성민입니다. 저기 숫자판으로 저분 좀 봐주세요."로 한 화자에 뭉쳐 있던 게 자기소개하는 사람과 안내하는 사람 2명으로 정확히 분리됨을 확인
- 15분 정도의 짧은 클립에서는 pyannote가 힌트 없이도 이미 합리적인 화자 수를 찾는 경우가 많아 힌트 효과가 뚜렷하지 않았음(같은 15분 클립으로 baseline과 min4~max10 힌트를 비교했을 때 결과가 완전히 동일) — 이 문제는 **긴 녹음(1시간 이상)에서 두드러짐**
- **화자 라벨은 실행 단위로만 유효**함(다른 파일/다른 실행 간에 같은 사람이 같은 번호로 매칭되지 않음)도 함께 확인 — 목소리로 신원을 인식/매칭하는 기능(voiceprint recognition)은 지원 대상이 아니므로, 이 부분은 UI 툴팁으로 안내만 하고 별도로 고치지 않음

### 화자분리 검증 관련 참고

3단계 통합(STT + 화자분리 + 정렬)은 Windows 내장 TTS로 만든 합성 음성으로 파이프라인 자체(에러 없이 동작, 시간/텍스트/화자 라벨이 올바르게 병합되는지)는 확인했습니다. 다만 두 합성 음성(둘 다 여성 목소리)을 pyannote가 같은 화자로 묶는 경우가 있었는데, 이는 합성 음성이 실제 사람 목소리보다 음색 차이가 작고 클립이 짧아서(~20초) 생긴 현상으로 보입니다. **실제 화자분리 정확도는 실제 사람 목소리가 담긴 파일로 직접 확인이 필요**합니다.
