# Phase 5 LLM-as-Judge — 실수 + 학습 기록

비용 손실 / 시간 낭비를 야기한 실수들. 향후 동일 실험 또는 다른 프로젝트에서 같은 실수 반복 방지용.

---

## 💰 비용 손실 (총 ~$350)

### 1. Anthropic Opus 4.7 — Padding 폭주 (-$245.71)

**증상**:
- Q3 + Q4 partial 진행 중 14,400 → 19,355건 처리에서 -$245 잔액
- chunk별 cache_creation_input_tokens 35.7M (1건당 ~7,150 tok)

**원인**:
- Opus 4.7 cache_min = 4,096 tok 제약 맞추려고 `_PAD_SYSTEM` 4,744 tok 강제 padding
- items 정렬을 candidate-major로 → 같은 (qid, metric) prefix가 흩어져서 cache 재사용 비효율
- 매 metric 변경마다 cache_write 발생

**교훈**:
- Anthropic Opus 같은 무거운 모델은 **padding 강제 금지** (cache 못 걸려도 raw input이 더 쌈)
- Sonnet 4.6는 cache_min 1024라 padding 없이 자연 도달 → Sonnet 권장
- items 정렬은 (qid, metric)-major (같은 prefix가 candidate 수만큼 reuse)

### 2. Gemini Cache Storage — 사용 안 한 cache의 storage (-$103)

**증상**:
- Anthropic 청구와 별도로 Gemini ₩143,421 (~$103) "Cached text storage token hours"
- 14,400 errored / cancelled batches인데도 storage만 청구

**원인**:
- v1 batch (cache 1,200 + ttl='1h') → 14,400 errored (TTL 형식 오류)
- v2 batch (cache 1,200 + ttl='6h') → 5h queue stuck → cancel
- Cache는 batch 결과와 무관하게 만든 시점부터 만료까지 **storage 비용 누적**
- TTL 6h로 늘렸더니 6h 동안 1,200 × 5K tok × $4.5/M/h = ~$162 청구

**교훈**:
- **Cache는 batch가 즉시 작동 확정 후에만 생성**
- Errored batch도 cache는 storage 청구 발생
- TTL은 짧게 (1h 권장), batch 끝나면 명시적 delete
- cancel 시 cache 자동 cleanup 안 됨 — 직접 delete 필수

---

## 🐛 API/모델 호환성 실수

### 3. `gpt-5.4-pro` reasoning_effort='low' 미지원

**증상**:
- 파일럿 batch 1,200/1,200 모두 failed
- 에러: `"Unsupported value: 'low' is not supported with the 'gpt-5.4-pro' model. Supported: 'medium', 'high', 'xhigh'."`

**교훈**:
- 모델별 reasoning_effort 지원 값이 다름
- gpt-5.5는 `'none'` 지원 (가장 비용 효율적)
- gpt-5.4-pro는 `'medium'`이 최소 → output token 폭증 우려
- **단일 호출 검증 필수** (batch 제출 전 1-call dry-run)

### 4. Gemini cache `ttl='1h'` 형식 오류

**증상**:
- 1,200 cache 생성 시도 직후 모두 400 INVALID_ARGUMENT
- `"Field 'ttl', Illegal duration format; duration must end with 's'"`

**교훈**:
- Gemini API는 protobuf Duration 형식 → `'3600s'` (초 단위 + s 접미사)
- `'1h'` 형식 미지원

### 5. Gemini Pro `thinking_budget=0` 미지원

**증상**:
- v2 batch (1,200 cache + thinking_budget=0) → 14,400 errored
- 에러: `"Budget 0 is invalid. This model only works in thinking mode."`

**교훈**:
- gemini-3.1-pro는 thinking 강제 모델 → thinking_budget 최소값 필요
- 다른 judge와 통일하려면 `thinking_budget=128` (최소 thinking)

---

## 🔧 환경/설정 실수

### 6. solar-open-100b chat template 누락

**증상**:
- Judge 모든 응답이 비정수 → 모든 candidate 0% accuracy
- ollama Modelfile에 `TEMPLATE {{ .Prompt }}` 한 줄만 정의됨 (raw prompt 전달)

**교훈**:
- Custom GGUF 모델은 chat template 명시적 정의 필요
- GLM4MOE architecture라면 GLM 또는 ChatML 형식
- judge 1-call 검증 (1자 정수 응답인지) 필수

### 7. 환경변수 우선순위 (`load_dotenv` override 문제)

**증상**:
- `.env`에 `ANTHROPIC_API_KEY=sk-ant-***REDACTED***` 있는데 401 인증 실패
- shell env에 다른 키 (`sk-ant-***REDACTED***`)가 export되어 있어 우선

**교훈**:
- `load_dotenv()` 기본은 기존 환경변수 덮어쓰지 않음
- 모든 entry point에 `load_dotenv(override=True)` 강제

### 8. SSH key 미등록 (서버 간)

**증상**:
- HP Z2 Mini → Desktop rsync 시 `Permission denied (publickey,password)`
- 처음에 Mac을 중계로 사용하려다 16GB 두 번 transfer 시도

**교훈**:
- 다중 서버 환경 — 처음부터 모든 pair에 ssh key 등록
- `cat ~/.ssh/id_ed25519.pub | ssh remote 'cat >> ~/.ssh/authorized_keys'` 한 번에

---

## 📊 데이터/스크립트 실수

### 9. Items 정렬 candidate-major (cache 재사용 비효율)

**증상**:
- 같은 (qid, metric) prefix가 candidate 12개 사이에 분산 → cache hit 거의 0%
- Opus padding 비용이 candidate별로 누적

**교훈**:
- (qid, metric)-major 정렬 → 같은 prefix 12회 연속 reuse
- `for qid: for metric: for cand:` 순서 (candidate 가장 안쪽)

### 10. Local 12 LLM `qid` None (positional 매핑 문제)

**증상**:
- `auto_judge_pipeline.py`가 `r.get("qid")` → None → `qid_to_target.get(None)` skip
- 14,400 items 중 3,600 candidates (local 12) 통째로 누락

**교훈**:
- 폴백 명시: `qid = r.get("qid") or f"q{idx:03d}"`
- 다양한 qid 형식 정규화 (`gen::q005`, `gen::model::q200` 등) — regex `q\d{3}` 추출

### 11. Custom_id 패턴 위반

**증상**:
- Anthropic batch 400: `String should match pattern '^[a-zA-Z0-9_-]{1,64}$'`
- model_id에 `.`, `:`, `/` 포함 → 미허용

**교훈**:
- Anthropic / OpenAI batch custom_id 정규화 필수
- `re.sub(r'[^a-zA-Z0-9_-]', '_', raw)[:64]`

### 12. Anthropic batch payload 413 Payload Too Large

**증상**:
- 34,792 requests를 단일 batch로 보내다 cloudflare 413
- request 1건당 ~30KB × 34K = ~1GB → 256MB 한도 초과

**교훈**:
- chunk 5,000건씩 분할 제출 (단일 batch ~150MB 안전 영역)
- Anthropic batch SLA: requests 한도 100K, payload 한도 256MB

---

## ⏱️ 운영 실수

### 13. Gemini Batch Queue Stuck (RPD 한도)

**증상**:
- v2 batch 5h+ JOB_STATE_RUNNING 유지, start_time None
- 사실상 RPD 250 (Pro) 한도에 막혀 처리 안 됨

**교훈**:
- Gemini RPD 한도 모델별 큰 차이: Pro 250, Flash 10K, Flash Lite 150K
- Batch 자체도 RPD 카운트에 포함될 수 있음
- 큰 작업 (14K calls)은 RPD 안에서만 가능

### 14. HP Z2 Mini 시스템 RAM vs VRAM 혼동

**증상**:
- `free -h`로 본 RAM 31GB → "100B 모델 못 들어감" 결론
- 실제 HP Z2 Mini (AI 395+) 통합 메모리 96GB가 GPU에 할당됨

**교훈**:
- AMD APU 시스템에서 BIOS가 GPU에 할당하는 메모리는 OS의 `free` 출력에 안 보임
- ROCm `rocm-smi` 또는 `lspci` + `mem_info_vram_total` 확인

### 15. ollama 디스크 공간 부족

**증상**:
- HP Z2 Mini에서 `ollama create` 중 `no space left on device`
- root partition 91% 사용 중 (43GB 남음, 100B 모델 62GB 안 들어감)

**교훈**:
- ollama 모델 위치 (`/usr/share/ollama/.ollama/models/blobs/`)는 root 디스크 사용
- 큰 모델은 `OLLAMA_MODELS` 환경변수로 별도 위치 지정
- 또는 ollama 우회하고 llama.cpp 직접 사용 (Modelfile 만들 필요 X)

### 16. Desktop GPU 80B 모델 partial CPU offload 25min/call

**증상**:
- qwen3-next:80b (50GB Q4_K_M) → Desktop 48GB GPU에 14GB CPU offload
- call당 ~25분, 1,200 calls/candidate × 12 cand = 비현실적 (수개월)

**교훈**:
- 모델 사이즈 > GPU VRAM이면 partial offload는 ✗ (PCIe RAM access가 token마다 발생 → memory bound)
- 양자화 변경 또는 더 작은 모델 사용 필요

---

## 🛠️ 권장 워크플로우 (Lessons Applied)

1. **단일 호출 검증** (batch 제출 전 1-call dry-run으로 모델/effort/format 작동 확인)
2. **작은 파일럿** (100 calls 수준) → 비용/속도 측정 → 본 진행
3. **API quota 사전 확인** (RPD/RPM/TPM 모델별 차이)
4. **Cache는 사용 확정 후에만 생성** (errored batch에도 storage 청구)
5. **chunk 분할** (Anthropic 5K, OpenAI 50K 한도 안에서 안전 영역)
6. **items 정렬 (qid, metric)-major** (cache reuse 극대화)
7. **dotenv override 명시** (`load_dotenv(override=True)`)
8. **SSH key 사전 등록** (multi-server 환경)
9. **디스크/메모리 사양 확인** (ollama 위치, ROCm 통합 메모리, GPU vs RAM)
