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

### LLM 화자 문맥 보정 (실험적, 2026-08-29~30)

사용자 피드백: "화자분리가 음향(목소리)만 보고 문맥/상황을 전혀 이해하지 못한다." 맞는 지적이라, "제 이름은 OOO입니다" 같은 문맥 단서로 화자 병합을 제안하는 후처리 기능을 추가했습니다.

- **동작 방식**: 화자분리 결과를 LLM에 보내 "문맥상 같은 사람인데 다른 화자 번호로 나뉜 경우"의 병합만 제안받음 (전체 텍스트를 다시 쓰게 하지 않고 라벨 매핑만 요청 — 환각 위험 최소화)
- **⚠️ 자동 적용 아님, 사람이 검토 후 선택 적용**: 처음 검증 때 AssemblyAI 계정에서 접근 가능한 유일한 모델(`qwen3.5-4b-32k-fast`, 소형 모델)이 (1) 명백히 다른 두 사람(등록 안내 스태프 vs 기술 얘기하는 참가자)을 병합하자고 제안하거나 (2) 입력에 없는 화자 라벨을 지어내는 것을 실제로 확인했습니다. 그래서:
  - 응답에서 실제 존재하지 않는 화자 라벨이 섞인 병합은 자동으로 걸러냄(`suggest_merges`)
  - 남은 제안도 자동 적용하지 않고, [gui/widgets/merge_review_dialog.py](transcribe_app/src/gui/widgets/merge_review_dialog.py)에서 각 제안마다 두 화자의 발화 예시를 보여주고 사람이 체크한 것만 적용

**정책 변경(2026-08-31): 이 기능에는 AssemblyAI를 더 이상 사용하지 않음.** Gemini로 재검증한 결과 qwen보다 훨씬 신중한 판단을 확인했고(아래), 이후로는 Gemini만 사용합니다(`core/llm_refine.py`). AssemblyAI LLM Gateway 관련 코드는 이 파일에서 제거했습니다 — AssemblyAI 자체(STT/화자분리 엔진)는 계속 별도 기능으로 사용 가능합니다.

- **Gemini 실데이터 재검증(2026-08-30)**: qwen보다 훨씬 신중한 판단을 확인.
  - Day1 개인녹음 15분 클립: qwen이 잘못 병합한 화자4/화자6을 Gemini는 "역할이 다르다"며 정확히 병합하지 않음. "화자 미상"과 화자1의 발화가 문장상 비슷해도 역할(안내 vs 기술대화)이 다르다는 이유로 병합 안 함 — qwen이 놓쳤던 문맥적 판단을 실제로 수행함
  - Day1 개인녹음 전체(4h53m, 화자 수 힌트로 11명 분리된 결과): 오히려 "일부 화자 라벨(A/B/C/E/F/H)이 자기소개하는 여러 사람을 이미 한 라벨로 묶고 있는 것 같다"는 **과소 분리** 문제를 스스로 짚어내고, 이런 불확실한 라벨을 더 병합하는 건 위험하다고 판단해 병합 안 함
  - 다만 현재 기능은 "병합"만 제안하므로, 이렇게 한 라벨 안에 여러 사람이 섞인 **과소 분리 문제는 고치지 못함** (라벨을 쪼개는 기능은 별도 구현 필요, 아직 없음)
- **무료/일반 Gemini 키 이중화 + 재시도(2026-08-31)**: `.env`에 `GEMINI_FREE_KEY`(무료 키, 우선 사용)와 `GEMINI_API_KEY`(일반 키, 롤백용) 둘 다 넣어두면, 무료 키를 먼저 최대 25회까지 재시도(지수 백오프, 2~15초 간격 — 무료 키는 사용자가 많아 429/503 같은 일시적 오류가 잦기 때문)하고, 그래도 실패하면 자동으로 일반 키로 전환합니다 (`get_provider_candidates()`, `suggest_merges()`).
  - 429/500/502/503/504처럼 재시도해서 해결될 수 있는 오류만 재시도하고, 400처럼 요청 자체가 잘못된 오류는 즉시 다음 키로 넘어감(재시도 로직은 모킹으로, 실제 재시도/롤백 동작은 실제 키로 각각 검증 완료)
- GUI: 전사 완료 후(화자분리 결과 + Gemini 키 있을 때) "화자 병합 제안 받기 (LLM, 실험적)" 버튼 활성화
- 매우 긴 녹음은 전사록이 6만자를 넘으면 앞부분만 잘라서 판단 (`MAX_TRANSCRIPT_CHARS`)

### LLM 모델/제공자: 하드코딩 제거 + xai/openai/gemini/claude 전체 폴백 체인 (2026-08-31)

**겪은 문제**: `gemini-2.5-flash`를 코드에 문자열로 박아뒀는데, 어느 순간 무료 키에서 "신규 사용자에게 더 이상 제공되지 않음"이라는 404 오류가 남(API 응답 메시지가 직접 다음 모델을 안내). 모델명이 이렇게 자주 바뀌는데 코드에 박아두면 바뀔 때마다 코드를 고쳐야 함.

**해결(1차, Gemini만)**: `C:\...\Automating_automatic_message_sending\src\aam\catalog.py`(다른 프로젝트, xai/openai/gemini/claude 여러 제공자를 지원하는 범용 CLI 도구)의 설계를 참고해 [core/llm_catalog.py](transcribe_app/src/core/llm_catalog.py)를 만듦.

**해결(2차, 전체 제공자로 확장)**: "API 키 연결 로직을 모두 반영하라"는 요청에 따라, `aam`의 `settings.py`(제공자 레지스트리)까지 통째로 이식해서 xai/openai/claude도 폴백 후보로 추가:

- [core/llm_providers.py](transcribe_app/src/core/llm_providers.py): `aam/settings.py`를 그대로 이식 — `PROVIDERS` 레지스트리(제공자별 env var 이름: `XAI_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`/`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`), `resolve_provider()`, `configured_providers()`
- [core/llm_call.py](transcribe_app/src/core/llm_call.py): 제공자별로 다른 실제 채팅 API 포맷(OpenAI 호환 Chat Completions / Claude Messages API / Gemini generateContent)을 `call_llm()` 하나로 통일
- [core/llm_catalog.py](transcribe_app/src/core/llm_catalog.py): 모델 목록 조회도 제공자별로 분기(`fetch_openai_compatible`, `fetch_claude`, `fetch_gemini_models`) → `fetch_models()`
- **폴백 체인** (`core/llm_refine.py`의 `get_provider_candidates()`): `gemini_free`(이 앱 전용, 25회 재시도) → `gemini` → `claude` → `openai` → `xai` 순. `.env`에 키가 없는 제공자는 자동으로 건너뜀
- 모델 선택은 슬롯(`gemini_free`/`gemini`/`claude`/`openai`/`xai`)별로 `llm_selection.json`에 저장 — 같은 Gemini라도 무료 키/일반 키가 접근 가능한 모델이 다를 수 있어서 따로 저장

**실제 키로 4개 제공자 전부 검증(2026-08-31)**:
- **openai**: 모델 124개 조회, `gpt-4.1-mini` 자동 선택 및 실제 채팅 호출 성공
- **gemini / gemini_free**: 위와 동일하게 정상 동작
- **claude**: 모델 목록(10개)은 정상 조회되지만 실제 채팅 호출은 "크레딧 부족"(400, 계정 과금 문제 — 코드 버그 아님)으로 실패 → 폴백 체인이 정상적으로 다음 제공자로 넘어감을 확인
- **xai**: 모델 목록 조회 자체가 403 "팀에 크레딧/라이선스 없음"으로 실패(계정 문제) → 마찬가지로 정상적으로 건너뜀
- **실제로 겪은 버그 2개, 모두 수정**:
  1. **자동 선택 키워드 버그**: 저렴한 모델을 우선 고르려고 `"mini"`를 키워드로 썼는데, "gemini"라는 단어 자체에 `"mini"`가 들어있어서(ge**mini**) 사실상 모든 Gemini 모델이 걸리는 문제 발견 → `"-mini"`처럼 하이픈을 붙여서 수정. 이미지/음성 전용 모델(`-image`, `-tts` 등)도 자동 선택 후보에서 제외하도록 추가
  2. **Gemini "thinking" 토큰이 응답 예산을 다 먹는 문제**: `gemini-2.5-flash`가 내부적으로 생각하는 토큰을 먼저 쓰고 남는 걸로 답을 만드는데, `maxOutputTokens`가 부족해서 JSON 응답이 중간에 잘리는 걸(`Unterminated string` 파싱 오류) 실제로 겪음 → `thinkingConfig.thinkingBudget: 0`으로 비활성화 + `max_tokens`를 700→1500으로 상향해서 해결

### 설정 화면: 실시간 LLM 모델 선택 UI + 화자 수 UI 개선 (2026-08-31)

**요청**: "지금은 자동으로만 고르고 있는 걸, 설정 화면에서 사용자가 실시간 목록을 보고 직접 모델을 고를 수 있게 만드는 UI를 만들어보자." + "메인 화면의 화자 수 선택 UI가 직관적이지 않아. 고쳐줘."

**LLM 모델 선택 UI**: [gui/widgets/llm_model_dialog.py](transcribe_app/src/gui/widgets/llm_model_dialog.py)를 새로 만들어 [설정] → "LLM 모델 선택..." 버튼으로 열림.
- `.env`에 키가 있는 슬롯(`gemini_free`/`gemini`/`claude`/`openai`/`xai`)만 드롭다운에 표시 (`core/llm_providers.configured_slots()`)
- "실시간 모델 목록 불러오기"로 `core/llm_catalog.fetch_models()`를 백그라운드 스레드(QThread)에서 호출 — GUI가 멈추지 않음
- 목록에서 모델을 골라 "선택한 모델로 저장"을 누르면 `core/llm_catalog.select_model()`이 실제 라이브 카탈로그에 있는지 확인 후 `llm_selection.json`에 저장(이것도 스레드에서 실행)
- "자동 선택으로 초기화"로 언제든 자동 선택 로직으로 되돌릴 수 있음(`clear_selected_model`)
- 이 작업 중에 `core/llm_providers.py`에 `resolve_slot()`/`SLOT_IDS`/`SLOT_LABELS`/`configured_slots()`를 추가해 `llm_refine.py`와 이 다이얼로그가 "gemini_free는 레지스트리에 없는 이 앱 전용 슬롯" 특수 처리를 중복 구현하지 않도록 정리함. 리팩터링 중 `get_provider_candidates()`가 실수로 `SLOT_IDS`(레지스트리 나열 순서, xai가 gemini보다 먼저 옴)를 폴백 우선순위로 쓸 뻔한 걸 테스트 전에 발견해 명시적 `FALLBACK_ORDER` 상수로 되돌림 — 모델 *선택 UI*의 표시 순서(`SLOT_IDS`)와 실패 시 *폴백 순서*(`FALLBACK_ORDER`)는 서로 다른 목적이라 분리 유지
- **실제 Gemini 무료 키로 3개 흐름 전부 검증**: 목록 조회(39개 모델 실시간 확인) → 저장(카탈로그 대조 후 `llm_selection.json` 갱신) → 자동 선택으로 초기화, 모두 정상 동작 확인

**화자 수 UI 개선**: [gui/main_window.py](transcribe_app/src/gui/main_window.py)에서 기존에 "화자분리 포함" 체크박스와 최소/최대 화자 수 스핀박스(`0=자동`이라는 텍스트로만 설명)가 한 줄에 뒤섞여 있어 관계가 불명확했음. 개선:
- "화자분리" `QGroupBox`로 묶고, 그 안에 "화자분리 포함" 체크박스 아래에 "예상 화자 수를 알고 있어요" 체크박스를 별도로 둠(껐을 때는 스핀박스가 비활성화됨 — `0=자동` 같은 암묵적 규칙 대신 체크박스로 명시)
- 체크하면 "최소 [2명] ~ 최대 [8명]"로 값을 직접 입력 — 화자 수 힌트 기능(위 "화자 수 힌트 기능" 절 참고)이 실제로 긴 녹음에서 정확도를 크게 올린다는 걸 이미 확인했으므로, 툴팁에 그 근거를 그대로 안내
- 최소 > 최대로 입력하면 전사 시작 전에 경고 후 중단
- 헤드리스 스모크 테스트로 토글 시 스핀박스 활성/비활성 전환이 올바른지 확인 완료

### 화자 병합 제안: 증거 기반(evidence-grounded) 검증 추가 (2026-08-31)

**요청**: "AI에게 권한을 넘겨주거나 모든 것을 의지하지 않게, 환각 방지를 위해 다른 사람들이 AI 환각 교차검증 문제를 어떻게 해결했는지 논문/에세이를 찾아 반영하자."

**리서치 방법**: 개별 논문 1000편을 일일이 나열하는 건 비현실적이고 신뢰할 수 없는 결과물(가짜 인용)로 이어지기 쉬워서, 대신 각각 수백 편(예: 300+ 편을 6개 범주로 분류한 MDPI 서베이)을 이미 종합한 **서베이 논문들 + 실무 에세이**를 폭넓게 검색해 이 앱의 실제 유스케이스(좁은 도메인의 구조화된 JSON 출력, 자유 서술 생성 아님)에 맞는 기법만 추렸다.

**핵심 발견 및 반영**:
- **LLM-as-judge는 구조적으로 과신(overconfident)한다** — self-reported confidence("90% 확신")는 실제 정확도와 잘 맞지 않는다는 연구 다수(예: [Overconfidence in LLM-as-a-Judge](https://arxiv.org/html/2508.06225v2)). 기존 프롬프트가 "확신 90% 미만이면 병합 금지"라고 시켰지만, 이건 모델이 "90%"라고 스스로 말하는 것 자체를 신뢰하는 셈이라 근본적 한계가 있음.
- **증거 기반 접근(evidence/citation grounding)이 더 신뢰할 수 있다** — 답변이 원문의 특정 구간(span)에 매여 있으면 검증·탈락이 기계적으로 가능하다는 게 grounded-citation 계열 연구의 결론(예: [Learning Fine-Grained Grounded Citations](https://arxiv.org/pdf/2408.04568)). → **각 병합 제안마다 "원문에서 그대로 복사한 인용문"을 화자별로 요구**하도록 [core/llm_refine.py](transcribe_app/src/core/llm_refine.py)의 `SYSTEM_PROMPT`를 변경.
- **제약된 출력(constrained output)은 그 자체로 환각의 한 유형을 원천 차단한다** — 이미 있던 "유효 화자 라벨만 허용" 필터를 넘어, 이제 **인용문도 원문과 문자열 대조로 검증**(`_verify_quote`/`_verify_and_filter_merges`)해서, 인용문 자체가 지어낸 것으로 확인되면(=판단 근거가 환각) 그 병합 제안 전체를 사람에게 보여주지도 않고 자동 폐기한다.
- **Human-in-the-loop는 그 자체로 충분하지 않다** — automation bias 연구([Nature Scientific Reports](https://www.nature.com/articles/s41598-026-34983-y) 등)에 따르면 사람도 "그럴듯해 보이면" 별 검토 없이 승인해버리는 경향이 있고, 이를 줄이려면 실제 근거를 눈에 보이게 제시하는 "cognitive forcing function"이 필요하다는 게 중론. → [gui/widgets/merge_review_dialog.py](transcribe_app/src/gui/widgets/merge_review_dialog.py)가 이제 임의의 "화자별 첫 발화 예시"(기존 `_sample_text`, 실제 판단 근거와 무관할 수 있었음) 대신 **원문 검증을 통과한 실제 근거 인용문**을 보여줌.
- **entity/coreference 병합 결정은 precision-over-recall이 정설** — 기존 설계(병합 누락이 오병합보다 낫다)가 문헌으로 재확인됨. 변경 없음, 그대로 유지.

**구현 변경 사항**:
- `merges` JSON 스키마를 `{"화자 F": "화자 A"}` 딕셔너리에서 `[{"from": ..., "to": ..., "rule": 1|2, "quote_src": "...", "quote_dst": "..."}]` 리스트로 변경 — `rule`은 기존 병합 기준 1(자기소개/호명)·2(문장 끊김) 중 어느 것에 해당하는지 명시하게 해서 판단을 더 구체적으로 강제함(mini chain-of-verification 성격).
- `suggest_merges()`가 이제 `dict[str, str]` 대신 검증된 `MergeCandidate` 객체 리스트를 반환 — `apply_merges()`, `MergeReviewDialog`, `gui/main_window.py`의 `MergeSuggestWorker` 시그널까지 전부 이 타입으로 정리.
- **실제 Gemini 무료 키로 Day1 15분 전체 데이터(203개 세그먼트) 재검증**: 새 스키마로 실제 응답을 받아 2건의 병합 제안이 나왔고, 둘 다 인용문이 원문과 정확히 일치함을 확인 — 특히 "화자 6: '맞아요 채팅을 보낸 게' → 화자 1: '툴입니다 라고'"는 실제로 한 문장이 화자분리 경계에서 끊긴 진짜 사례였음(`output/validation/day1/personal_15min_result.txt` 200~201행에서 직접 대조 확인).
- 단위 테스트로 정상/인용문 조작(환각)/존재하지 않는 라벨/너무 짧은 인용문/형식 오류/공백 차이 6가지 케이스를 모두 확인.

### 화자 병합 제안: 교차 제공자 컨센서스 추가 (2026-08-31)

**요청**: "교차 제공자 컨센서스도 진행해줘" — 증거 기반 검증(위 절)에 이어, 리서치에서 찾은 두 번째 기법을 반영.

**근거**: multi-agent debate/self-consistency 계열 연구는 서로 독립적인 모델이 같은 결론에 도달했을 때 그 결론을 더 신뢰할 수 있다고 본다 — 예: ChatGPT와 Bard가 각각 따로는 틀렸던 답이 교차 검증(cross-model debate) 후 정답으로 수렴한 사례가 보고됨. 다만 "모델 다양성이 debate 성공의 핵심 변수"라는 지적도 있어([Can LLM Agents Really Debate?](https://arxiv.org/html/2511.07784)), `gemini_free`/`gemini`처럼 같은 벤더의 키 두 개로는 상관된 오류를 낼 수 있다고 보고 제외했다.

**동작 방식** (`core/llm_refine.py`):
1. 1차 제공자(폴백 체인 순서)가 응답하면, 원문 대조 검증(위 절)을 통과한 제안들을 얻는다.
2. 제안이 하나라도 있으면, **1차와 다른 벤더**(`ResolvedProvider.id` 기준)의 제공자를 찾아 같은 전사록을 독립적으로 다시 보여준다(`_consensus_checker_candidates`) — 이 응답도 동일하게 원문 대조 검증을 거친다.
3. 두 제공자가 **모두 제안한(같은 화자 라벨 쌍)** 병합만 최종 후보로 남긴다(교집합). 한쪽만 제안한 건 자동 제외.
4. 다른 벤더 키가 아예 없거나, 있어도 호출이 전부 실패하면(예: 크레딧 부족) 조용히 1차 결과만 쓰는 것으로 후퇴하되, **이 사실을 항상 `consensus_note`로 사용자에게 알림** — [gui/widgets/merge_review_dialog.py](transcribe_app/src/gui/widgets/merge_review_dialog.py)에 굵은 글씨로 표시됨. "AI 판단을 조용히 신뢰하지 않는다"는 원칙을 UI에서도 지킴.

**실제 키로 검증(2026-08-31, Day1 15분 데이터 203세그먼트)**:
- 1차(gemini_free)가 2건 제안 → 교차검증 후보로 claude를 먼저 시도했으나 **실제로 크레딧 부족 오류로 실패**(기존에 알려진 계정 문제) → 자동으로 다음 후보 openai로 넘어가 `gpt-4.1-mini`로 성공.
- 결과: **openai가 독립적으로 재검토한 결과 2건 중 0건이 일치** → 최종 후보 0건으로 자동 축소됨. openai의 reasoning을 직접 확인해보니, gemini_free가 제안한 "화자 1 ↔ 화자 6" 병합의 근거(문장이 이어짐)를 openai도 **텍스트로는 인지했지만**, "자기소개/명칭 일치로 명확히 확인되지 않는다"며 더 엄격하게 병합을 보류했음 — 실제 모델 간 판단 차이가 존재함을 확인.
- 이건 버그가 아니라 설계대로 동작한 것: 단일 모델이 그럴듯한 근거(원문에 실제로 있는 인용문)를 대도, 독립적인 재확인에서 동의를 못 받으면 최종적으로는 보수적으로 병합하지 않는 쪽을 택함(병합 누락이 오병합보다 낫다는 기존 철학과 일관).
- 목(mock)으로 완전 일치/부분 불일치/다른 벤더 키 없음 3가지 분기의 메시지 포매팅도 별도 검증.

**한계**: 호출 비용이 최대 2배(1차 + 교차검증)로 늘어남. 다른 벤더 키가 하나도 없으면(예: Gemini 키만 있는 환경) 컨센서스 없이 기존과 동일하게 동작 — 이 경우도 `consensus_note`로 명시됨.

### 설정 화면에 교차검증 켜기/끄기 토글 추가 (2026-08-31)

**요청**: "설정 화면에 교차검증 켜기/끄기 토글 추가해줘" — 위 컨센서스 기능이 API 호출을 최대 2배로 늘리는 트레이드오프가 있어서, 사용자가 직접 켜고 끌 수 있게 해달라는 후속 요청.

- `core/config.py`의 `Settings`에 `cross_validate_merges: bool = True`(기본 켜짐 — 신뢰성 우선) 추가, `core/settings_store.py`의 `_PERSISTED_FIELDS`에도 반영해 `settings.json`에 저장.
- `core/llm_refine.py`의 `suggest_merges()`에 `cross_validate: bool = True` 매개변수 추가 — `False`면 1차 응답 직후 교차검증을 아예 시도하지 않고 바로 반환(호출 자체가 안 나가므로 진짜로 절반으로 줄어듦), `consensus_note`에 "교차검증 꺼짐(설정에서 비활성화됨)"이라고 명시해 어떤 모드로 나온 결과인지 항상 알 수 있게 함.
- [gui/widgets/settings_dialog.py](transcribe_app/src/gui/widgets/settings_dialog.py)의 "LLM 모델 (화자 병합 제안)" 그룹에 "다른 제공자로 교차검증 (권장)" 체크박스 추가, 툴팁으로 트레이드오프(호출 2배 vs 신뢰도) 설명.
- `gui/main_window.py`의 `MergeSuggestWorker`가 `self._settings.cross_validate_merges`를 읽어 `suggest_merges()`에 전달.
- **검증**: 설정 다이얼로그 체크박스 ↔ `Settings.cross_validate_merges` ↔ `settings.json` 저장/재로드 왕복 확인, 실제 Gemini 무료 키로 `cross_validate=False` 호출 시 교차검증 관련 상태 메시지가 하나도 발생하지 않음(=호출 자체를 안 함)을 직접 확인.

### 메인 화면: 파일 선택 버튼 깜빡임 + 실제 로딩 진행률 표시 (2026-08-31)

**요청**: "메인 화면이 너무 복잡하므로, 사용자가 알기 쉽도록 파일 경로가 선택이 안되어있는 경우 '파일선택' 버튼이 깜빡이도록 해주시고. 로딩 진행률도 표시하도록 해주세요."

**"다음에 뭘 눌러야 하는지" 버튼 깜빡임** ([gui/main_window.py](transcribe_app/src/gui/main_window.py)):
- 조건(예: 파일 미선택)이 갖춰지지 않았거나 그 단계가 아직 실행되지 않은 동안, 다음에 눌러야 할 버튼을 500ms 간격 `QTimer`로 깜빡임 — `_start_blink(key, button)`/`_stop_blink(key, button)`/`_toggle_blink(key, button)`으로 여러 버튼에 재사용 가능한 범용 메커니즘.
- "파일 선택" 버튼: 파일이 선택 안 되어 있는 동안 깜빡임(`self._selected_path is None`). 파일을 고르면(`_set_selected_path`) 멈춤.
- **"오디오 추출 및 정보 확인" 버튼(후속 요청, 2026-08-31)**: "경로가 선택되어 있고(조건 충족) + 아직 추출을 실행하지 않았을 때"만 깜빡이도록 추가. 파일을 고르면 시작(`_set_selected_path`에서 `_start_blink("extract", ...)`), 버튼을 눌러 추출이 시작되면 즉시 멈춤(`_on_extract`), 추출이 실패하면 다시 눌러야 하므로 재개(`_on_extract_failed`), 추출이 성공하면(다음 단계는 전사이므로) 다시 켜지 않음.
- **"전사 시작" 버튼(추가 후속 요청, 2026-08-31)**: "추출이 끝났고(조건 충족) + 아직 전사를 실행하지 않았을 때"만 깜빡이도록 추가. 추출이 성공하면 시작(`_on_extract_done`), 전사가 시작되면 멈춤, 실패하면 재개, 성공하면 재개 안 함. 새 파일을 선택하면(이전 추출 결과가 무효가 되므로) `_set_selected_path`에서 멈춤.
  - **부수적으로 발견해 수정한 버그**: 이 작업을 하다가 `_on_transcribe()`가 화자 수(최소>최대) 입력 검증보다 *먼저* 버튼 비활성화·진행바 표시를 해버리는 순서 문제를 발견함 — 잘못된 입력으로 경고창만 뜨고 반환되면 "전사 시작"/"오디오 추출" 버튼이 계속 비활성화된 채로, 진행바도 계속 떠 있는 채로 멈춰 있었음(실제로 아무것도 실행되지 않았는데도). 검증을 모든 부수효과보다 먼저 수행하도록 순서를 바꿔서 해결.
- 헤드리스 스모크 테스트로 세 버튼의 전체 상태 전이(시작→선택→클릭→실패 재개→성공 시 미재개→새 파일 선택 시 초기화)와 화자 수 검증 실패 시 버튼/진행바가 실제로 건드려지지 않는지까지 모두 확인. 테스트 중 `QMessageBox.critical()`/`warning()`이 오프스크린 환경에서 이벤트 루프 없이 모달로 블로킹되어 테스트가 멈추는 현상을 발견했는데, 이는 실제 앱 동작(모달이 사용자 클릭을 기다리는 건 정상)이 아니라 무한 대기형 헤드리스 테스트 스크립트 자체의 한계라 테스트에서만 무력화해서 확인함.

**깜빡임을 입체(키캡) 디자인 + 실제 그림자로 업그레이드(후속 요청, 2026-08-31)**: "단순히 색상만 깜빡이지 말고, 튀어나오는 듯한 입체 버튼(옵션 키) 디자인과 그림자를 반영해달라"는 요청. Qt 스타일시트(QSS)는 CSS의 `box-shadow`를 지원하지 않아서(공식적으로 없는 속성), 그림자는 `QGraphicsDropShadowEffect`를 버튼에 직접 붙여서 구현해야 함:
- "켜짐" 프레임: `_BLINK_RAISED_STYLE`(위는 밝고 아래는 진한 주황 그라디언트 + 아래쪽이 더 두꺼운 어두운 테두리로 볼록한 키캡처럼 보이게) + `QGraphicsDropShadowEffect`(blur 20, offset (0,5), 반투명 검정)를 함께 적용.
- "꺼짐" 프레임: 스타일시트와 그림자 이펙트를 모두 제거해 원래의 평평한 기본 버튼으로 되돌림.
- 500ms마다 이 두 프레임을 오가며, 단순 색상 점멸보다 훨씬 뚜렷하게 "이게 지금 눌러야 할 것"임을 전달.
- **시각적 검증**: 오프스크린 렌더링으로 실제 창을 그린 뒤 버튼 영역만 크롭해 확대한 스크린샷으로 "켜짐"(그라디언트+테두리+그림자가 뚜렷이 보임) vs "꺼짐"(평평한 기본 버튼) 상태를 직접 비교 확인. 이 과정에서 처음엔 크롭 좌표 계산 실수(`QWidget.mapTo()`에 이미 부모-상대 좌표인 `geometry().topLeft()`를 넘겨서 좌표가 중복 오프셋되는 버그)로 크롭이 안 되고 전체 창이 나오는 문제가 있었는데, 버튼의 로컬 원점 `QPoint(0,0)`을 넘기도록 고쳐서 해결. 스타일 변경 후에도 기존 깜빡임 상태 전이 테스트(3개 버튼) 전부 재확인해 로직 회귀가 없음을 확인.

**깜빡임을 부드러운 애니메이션으로 전환(추가 후속 요청, 2026-08-31)**: "깜빡임 동작을 조금 더 부드럽게 해줘." 기존엔 500ms `QTimer`로 평평한 상태 ↔ 입체(키캡) 상태를 뚝뚝 끊어서 전환했는데, 이 하드컷 자체가 부드럽지 않다고 판단해 애니메이션 방식으로 바꿈:
- 버튼은 깜빡이는 동안 **항상 입체(키캡) 스타일을 유지**(더 이상 평평한 상태로 뚝뚝 끊기지 않음). 대신 `QGraphicsDropShadowEffect`의 흐림 정도(blurRadius 10~32)와 투명도(alpha 90~240)를 `QVariantAnimation`(구간 0→1→0, `QEasingCurve.InOutSine`, 900ms, 무한 반복)으로 연속적으로 오가게 해서 그림자가 "숨쉬듯" 부풀었다 줄어드는 펄스로 만듦.
- QSS 자체는 애니메이션 대상 프로퍼티가 아니라서(문자열이라 보간 불가), 대신 애니메이션 가능한 `QGraphicsDropShadowEffect`의 `blurRadius`/`color` 프로퍼티만 매 프레임 갱신하는 방식을 택함 — 버튼의 입체 모양(그라디언트/테두리)은 고정해두고 그림자만 살아있는 느낌을 주는 절충.
- **시각적 검증**: 애니메이션을 특정 위상(phase 0.0/0.5/1.0)에 고정시켜 각각 스크린샷을 찍어 비교 — 0.0(흐림 10, 옅음) → 0.5(흐림 21, 중간) → 1.0(흐림 32, 진하고 넓게 번짐)으로 그림자가 실제로 매끄럽게 커지는 것을 직접 확인. (검증 중 살아있는 애니메이션이 이미 돌고 있는 상태에서 수동으로 위상을 덮어써서 값이 거의 안 바뀌어 보이는 착시가 있었는데, 애니메이션을 먼저 멈추고 나서 수동 위상을 넣어야 한다는 걸 깨닫고 고쳐서 정확히 확인함.) 기존 3개 버튼의 상태 전이(시작→선택→클릭→실패 재개→성공 시 미재개) 회귀 테스트도 애니메이션 상태(`QAbstractAnimation.State.Running`) 기준으로 다시 작성해 전부 통과 확인.

**실제 로딩 진행률**: 기존에는 모든 단계가 "돌아가는 중"만 보여주는 불확정(indeterminate) 진행바였음. 로컬 엔진(기본값)의 두 단계에 실제 진행률을 연결:
- [core/engines/local_whisper.py](transcribe_app/src/core/engines/local_whisper.py)의 `transcribe()`에 `progress_callback(완료 윈도우 수, 전체 윈도우 수)` 추가 — VAD로 나눈 윈도우를 순차 처리하므로 "지금까지 처리한 윈도우 / 전체 윈도우"가 실제 진행률의 합리적인 근사치.
- [core/engines/local_pyannote.py](transcribe_app/src/core/engines/local_pyannote.py)의 `diarize()`에 `progress_callback(내부 단계 이름, 완료, 전체)` 추가 — pyannote.audio Pipeline이 지원하는 `hook` 파라미터(콘솔에 진행률 바를 그리는 `ProgressHook`과 같은 메커니즘)를 그대로 활용해, 콘솔 대신 이 콜백으로 GUI에 전달.
- `gui/main_window.py`의 `TranscribeWorker`에 `progress = Signal(int, int)` 추가, 두 콜백을 여기 연결. `MainWindow._on_transcribe_progress()`가 처음 신호를 받으면 진행바를 불확정 모드에서 0~100% 확정 모드로 전환.
- 추출(ffmpeg)·API(AssemblyAI) 단계는 실제 진행률을 얻기 어려워 기존대로 불확정 표시 유지 — 억지로 가짜 퍼센트를 만들지 않음.

**실제 발견한 버그(수정 완료)**: 실제 3분 클립(`output/validation/day1/personal_probe.wav`)으로 STT+화자분리 전체를 실제로 돌려 진행률 신호를 관찰하던 중, pyannote의 내부 배치가 겹쳐서 `done`이 그 순간의 `total`을 넘는 경우(예: 192/171)가 실제로 발생함을 확인. `QProgressBar.setValue()`는 범위를 벗어난 값을 조용히 무시하고 진행바를 `-1`(비정상 상태)로 남겨둔다는 것도 함께 확인 — `_on_transcribe_progress()`에서 퍼센트를 0~100으로 직접 클램프하도록 고쳐서 해결.

**검증**: 헤드리스 스모크 테스트(불확정→확정 전환, `done>total` 클램프)에 더해, `TranscribeWorker`를 실제 오디오로 직접 실행해(threading 없이 `run()` 동기 호출) 실제 STT 7단계·화자분리 segmentation/embeddings 단계별 진행 신호가 순서대로 발생하고 상태 텍스트("전사 중... (5/7 구간)", "화자분리 중: embeddings (17/17)")가 올바르게 표시되는 것을 직접 확인.

### 화자분리 검증 관련 참고

3단계 통합(STT + 화자분리 + 정렬)은 Windows 내장 TTS로 만든 합성 음성으로 파이프라인 자체(에러 없이 동작, 시간/텍스트/화자 라벨이 올바르게 병합되는지)는 확인했습니다. 다만 두 합성 음성(둘 다 여성 목소리)을 pyannote가 같은 화자로 묶는 경우가 있었는데, 이는 합성 음성이 실제 사람 목소리보다 음색 차이가 작고 클립이 짧아서(~20초) 생긴 현상으로 보입니다. **실제 화자분리 정확도는 실제 사람 목소리가 담긴 파일로 직접 확인이 필요**합니다.
