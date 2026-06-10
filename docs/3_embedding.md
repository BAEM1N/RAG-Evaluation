# Embedding — 한국어 RAG에 적합한 임베딩 모델 선정

> **데이터셋**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs)
>
> **고정 조건**: parser=pymupdf4llm, chunking=500/100, FAISS, top-k=5 (Stage 2 winner 확정 전 측정 — 이후 stage들과 다른 베이스라인)
>
> **측정**: MRR / Hit@1 / Hit@5 / File@5 (page-match)
>
> **방법론 통일 (raw)**: 전 모델 동일 청크·동일 300 gt·cosine·**raw 입력(프롬프트/instruction/query·document 무구분)**. 각 모델은 자기 **올바른 구성**(네이티브 pooling, 올바른 라이브러리 버전, 정확한 dtype)으로 측정 — 데이터·metric은 100% 동일, 구현 디테일만 모델별 최적.

> ⚠️ **재현·교정 이력 (2026-06-04)**: 초기 측정은 멀티모델 게이트웨이(llama.cpp)에서 **pooling 미지정(기본 mean)**으로 일부 모델이 과소평가됐었음. 전 모델을 **올바른 네이티브 pooling + 올바른 버전**으로 raw 재측정해 교정. 주요 교정: `qwen3-embed-8b` 0.527→**0.650**(+0.123, last-pooling), `qwen3-embed-4b` +0.066, `bge-m3` +0.060, `labse` 0.047→**0.332**(깨진 변환→정상), `jina-v5-nano` 0.179→**0.524**. 상용 API는 query/document task 차별 없이 raw로 통일(대부분 기존값과 동일, upstage·cohere만 소폭 하락). 자체 파인튜닝 모델은 본 표에서 제외.

## 39 모델 leaderboard (로컬 31 + 상용 API 8, 전부 raw)

| 순위 | 모델 | dim | MRR | Hit@1 | Hit@5 | File@5 | 비고 |
|---:|---|---:|---:|---:|---:|---:|---|
| 🥇 | **voyage-3-large** | 1024 | 0.6873 | 62.3% | 76.3% | 87.3% | **API**·Voyage |
| 🥈 | **kure-v1** | 1024 | 0.6687 | 61.3% | 73.7% | 85.3% | 한국어 특화(BGE-M3 기반), 오픈 1위 |
| 🥉 | pixie-rune-v1 | 1024 | 0.6581 | 59.7% | 74.3% | 85.0% | 한국어 |
| 4 | **gemini-embed-001** | 3072 | 0.6518 | 58.7% | 74.0% | 87.0% | **API**·Google |
| 5 | snowflake-arctic-ko | 1024 | 0.6516 | 59.0% | 73.7% | 85.0% | 한국어 튜닝 |
| 6 | **cohere-embed-v4** | 1536 | 0.6516 | 58.0% | 75.3% | 86.0% | **API**·Cohere |
| 7 | qwen3-embed-4b | 4096 | 0.6512 | 58.3% | 74.0% | 86.3% | pooling=last |
| 8 | qwen3-embed-8b | 4096 | 0.6502 | 57.7% | 75.0% | 85.7% | pooling=last (교정 +0.123) |
| 9 | **voyage-multilingual-2** | 1024 | 0.6436 | 58.3% | 73.0% | 86.0% | **API**·Voyage |
| 10 | koe5 | 1024 | 0.6422 | 57.3% | 73.7% | 84.3% | 한국어(E5 기반) |
| 11 | jina-v4-retrieval | 4096 | 0.6359 | 56.3% | 73.0% | 86.3% | VL(Qwen2.5-VL) |
| 12 | **gemma-embed-300m** | 768 | 0.6350 | 56.0% | 71.7% | 84.0% | 최고의 소형 모델(운영) |
| 13 | **upstage-solar-large** | 4096 | 0.6325 | 55.7% | 73.3% | 84.7% | **API**·Upstage |
| 14 | bge-m3 | 1024 | 0.6227 | 54.7% | 73.3% | 85.0% | pooling=cls |
| 15 | **gemini-embed-2** | 3072 | 0.6210 | 54.7% | 71.0% | 87.0% | **API**·Google (001보다 낮음) |
| 16 | snowflake-arctic-l-v2 | 1024 | 0.6204 | 55.0% | 70.3% | 84.0% |  |
| 17 | harrier-27b | 5376 | 0.6177 | 55.0% | 70.3% | 84.7% | Gemma3-27B·last-pooling. **구 0.017은 fp16-NaN 버그** — bf16/fp32 재측정 0.618 (↓ 정밀도·양자화 절) |
| 18 | qwen3-embed-0.6b | 1024 | 0.6163 | 53.3% | 73.7% | 83.7% |  |
| 19 | me5-large | 1024 | 0.6120 | 53.3% | 71.7% | 83.7% | Multilingual-E5 |
| 20 | nomic-embed-v2-moe | 768 | 0.6027 | 53.7% | 70.7% | 82.3% | MoE |
| 21 | **openai-embed-3-large** | 3072 | 0.6008 | 52.3% | 71.7% | 84.0% | **API**·OpenAI |
| 22 | jina-v3 | 1024 | 0.5824 | 50.3% | 69.7% | 84.0% |  |
| 23 | harrier-0.6b | 1024 | 0.5810 | 50.7% | 66.3% | 81.0% | pooling=last |
| 24 | bge-mult-gemma2 | 3584 | 0.5684 | 49.7% | 66.3% | 81.7% |  |
| 25 | granite-278m | 768 | 0.5657 | 48.7% | 66.0% | 79.3% | IBM |
| 26 | me5-large-instruct | 1024 | 0.5576 | 48.3% | 67.0% | 83.3% |  |
| 27 | jina-v4-code | 4096 | 0.5438 | 46.0% | 65.0% | 82.0% | 코드 특화 |
| 28 | **openai-embed-3-small** | 1536 | 0.5417 | 45.3% | 66.0% | 77.7% | **API**·OpenAI |
| 29 | granite-107m | 768 | 0.5330 | 44.7% | 66.0% | 79.3% |  |
| 30 | jina-v5-nano-matching | 512 | 0.5238 | 43.3% | 62.7% | 81.0% | matching 튜닝 |
| 31 | harrier-270m | 640 | 0.5052 | 42.3% | 61.3% | 81.3% | pooling=last |
| 32 | ko-sroberta | 768 | 0.4935 | 41.0% | 61.3% | 82.7% |  |
| 33 | kosimcse-roberta | 768 | 0.4483 | 35.3% | 56.0% | 77.7% |  |
| 34 | nemotron-embed-8b | 4096 | 0.4404 | 35.3% | 56.0% | 80.3% | bidirectional (=llama-embed-nemotron-8b, 동일 가중치 — 중복 1종 제거) |
| 35 | jina-v5-small-retrieval | 1024 | 0.4370 | 37.0% | 53.7% | 71.3% |  |
| 36 | jina-code-1.5b | 1024 | 0.3621 | 26.7% | 49.0% | 71.3% | 코드 특화 |
| 37 | labse | 768 | 0.3320 | 25.0% | 45.7% | 68.7% | BERT 다국어 |
| 38 | e5-mistral-7b | 4096 | 0.2016 | 14.3% | 28.7% | 63.3% | instruction 모델(raw 부적합) |
| 39 | mxbai-embed-large | 1024 | 0.1533 | 12.0% | 20.7% | 46.7% | prompt 의존(영어) |

## raw vs prompted (프롬프트 방식 비교)

위 리더보드는 **raw**(질문·문서를 동일하게, 프롬프트/instruction/query·document 무구분) 기준이다. 비교를 위해 각 모델의 **권장 프롬프트 방식**(query/document 구분 — e5 `query:`/`passage:`, qwen3 instruct, mxbai/embeddinggemma 전용 프롬프트, API는 input_type query/document)도 동일 데이터로 측정했다.

| 모델 | raw | prompted | Δ |
|---|---:|---:|---:|
| me5-large-instruct | 0.5576 | 0.6156 | **+0.058** |
| upstage-solar-large | 0.6325 | 0.6771 | **+0.045** |
| voyage-multilingual-2 | 0.6436 | 0.6604 | +0.017 |
| cohere-embed-v4 | 0.6516 | 0.6674 | +0.016 |
| nomic-embed-v2-moe | 0.6027 | 0.6167 | +0.014 |
| qwen3-embed-0.6b | 0.6163 | 0.6295 | +0.013 |
| snowflake-arctic-l-v2 | 0.6204 | 0.6330 | +0.013 |
| snowflake-arctic-ko | 0.6516 | 0.6620 | +0.010 |
| qwen3-embed-4b | 0.6512 | 0.6611 | +0.010 |
| qwen3-embed-8b | 0.6502 | 0.6541 | +0.004 |
| voyage-3-large | 0.6873 | 0.6899 | +0.003 |
| koe5 | 0.6422 | 0.6437 | +0.002 |
| jina-v4-retrieval | 0.6359 | 0.6434 | +0.008 |
| mxbai-embed-large | 0.1533 | 0.1639 | +0.011 |
| e5-mistral-7b | 0.2016 | 0.2111 | +0.010 |
| gemini-embed-001 | 0.6518 | 0.6443 | **-0.008** |
| gemma-embed-300m | 0.6350 | 0.6247 | **-0.010** |
| jina-v4-code | 0.5438 | 0.4968 | **-0.047** |

> 대칭(symmetric) 모델 — BERT 계열(bge-m3·kure·pixie·granite·labse·snowflake-arctic-ko 일부·ko-sroberta·kosimcse·harrier·jina-v5·nemotron 등) — 은 query/document 프롬프트 개념이 없어 prompted = raw.

**해석**:
- **프롬프트가 대체로 소폭(+0.01~0.06) 도움** — 특히 instruction 튜닝 모델(`me5-large-instruct` +0.058)과 query/passage 모델 분리형 API(`upstage` +0.045)에서 효과가 크다.
- **그러나 보편적이지 않다 — 오히려 손해 보는 모델도 있다**: `gemma-embed-300m`(-0.010)·`gemini-embedding-001`(-0.008)은 권장 프롬프트가 이 한국어 문서검색 코퍼스엔 부적합. `jina-v4-code`는 **-0.047** — code-task 프롬프트가 일반 문서검색에 맞지 않아 크게 하락(같은 모델의 retrieval-task는 +0.008).
- **순위 영향 제한적**: 상위권은 raw·prompted 모두 voyage/kure/snowflake-ko/cohere/qwen3·upstage(prompted 시) 군집. prompt 의존이 큰 `e5-mistral`·`mxbai`는 프롬프트를 줘도 여전히 한국어에서 바닥(0.21·0.16).
- **결론**: raw는 전 모델을 동일 footing에 올리는 **공정한 공통 기준선**, prompted는 각 모델의 "권장 사용 시 상한"을 보여준다. 실제 운영에선 모델별 권장 프롬프트를 쓰는 게 보통 미세하게 유리하나, **모델·도메인에 따라 역효과도 나므로 검증 후 적용**해야 한다.

## 정밀도(dtype) · 양자화(정수) 강건성

리더보드는 모델별 **네이티브 정밀도**(대부분 fp32/bf16) raw MRR이다. 운영 배포 시 메모리·속도를 위해 정밀도를 낮추는데, 두 종류를 구분해야 한다 — **부동소수점 dtype 축소**(fp32→bf16→fp16)와 **정수 양자화**(Q8~Q2). 동일 데이터·page-match로 측정.

### A. dtype 정밀도 — 사실상 무손실 (단 Gemma fp16 주의)

전 로컬 모델 fp32/bf16/fp16 측정 (대표 12종, 26종 전체 중 발췌):

| 모델 | 계열 | fp32 | bf16 | fp16 |
|---|---|---:|---:|---:|
| kure-v1 | BERT | 0.6687 | 0.6656 | 0.6687 |
| qwen3-embed-4b | Qwen | 0.6514 | 0.6512 | 0.6514 |
| snowflake-arctic-ko | BERT | 0.6516 | 0.6531 | 0.6505 |
| bge-m3 | BERT | 0.6227 | 0.6299 | 0.6227 |
| koe5 | E5 | 0.6422 | 0.6417 | 0.6422 |
| me5-large | E5 | 0.6120 | 0.6158 | 0.6103 |
| nomic-v2-moe | MoE | 0.6027 | 0.6007 | 0.6027 |
| **gemma-embed-300m** | 🔴 Gemma | 0.6350 | 0.6360 | **NaN** |
| **gemma-ft-v1** | 🔴 Gemma | 0.6453 | 0.6436 | **NaN** |
| **harrier-27b** | 🔴 Gemma | 0.6177 | 0.6180 | **NaN** |
| **harrier-270m** | 🔴 Gemma | 0.5014 | 0.4979 | **NaN** |
| harrier-0.6b | Gemma* | 0.5806 | 0.5810 | 0.5806 |

<sub>* harrier-0.6b는 Gemma 계열이나 activation이 작아 fp16 통과(경계선). bf16−fp32 차이는 전 행 ±0.007 내.</sub>

- **bf16 ≈ fp32 — 무손실** (26종 평균차 −0.0008, 최대 |0.007|). **27B(harrier-27b)에서도 fp32 0.6177 ≈ bf16 0.618 (Δ −0.0003)** 확인. → 운영 기본은 **bf16** 권장(메모리 절반·동일 품질).
- **fp16 — 비-Gemma 무손실**(평균 −0.0003)이나 **Gemma 계열은 NaN**. fp16 NaN 5종 전부 Gemma3 기반: `gemma-embed-300m`·gemma-ft×3·`harrier-270m`·`harrier-27b`. Gemma의 큰 activation이 fp16 동적범위(±65504)를 초과해 오버플로.
- **harrier-27b 0.017 미스터리 해결**: 구 측정값은 GGUF/fp16 경로의 NaN 버그였고, **bf16/fp32 정상 측정 시 0.618**(리더보드 #17). 모델 결함이 아니라 정밀도 처리 버그.

> dtype은 부동소수점 스케일만 바꾸므로 **품질 손실은 당연히 ~0** (fp16-NaN은 손실이 아니라 수치 오버플로 사고). "양자화 성능"의 본질은 아래 정수 양자화에서 드러난다.

### B. 정수 양자화 — Q4 스윗스팟, Q3 아래 절벽 (공식 GGUF)

대표 3종에 **모델 제작사/커뮤니티 공식 GGUF**(우리가 양자화하지 않음)를 적용, llama.cpp page-match MRR:

| 모델 | fp | Q8 | Q6 | Q5 | Q4 | Q3 | Q2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **qwen3-embed-8b** (8B) | 0.654 | 0.654 | 0.652 | 0.656 | **0.654** | — | — |
| **nomic-v2-moe** (305m MoE) | 0.606 | 0.605 | — | 0.608 | **0.607** | 0.586 | **0.319** |
| **embeddinggemma-300m** (Google QAT) | 0.635 | 0.639 | — | — | **0.629** | — | — |

- **Q4가 보편적 스윗스팟**: 공식 K-quant/QAT 기준 fp 대비 손실 ~0이면서 크기 **1/3~1/4**(8B를 15GB→4.7GB). 임베딩 검색은 4-bit 정수 양자화에 매우 강건.
- **손실 절벽은 Q4 *아래***: nomic Q3 −0.020, **Q2_K 0.319로 붕괴**(절반). 운영에서 Q3 이하는 위험.
- **모델이 클수록 강건**: qwen-8B는 Q4도 손실 0인데, gemma-300m 공식 **QAT-int4는 −0.006**. 파라미터 여유가 양자화 오차를 흡수 → 엣지용 소형 모델일수록 저비트에 민감.
- **8-bit(Q8)은 전 모델 무손실** (dtype 결론과 일치).

### C. 비트 선택 가이드

| 정밀도 | 품질 손실 | 판정 | 용도 |
|---|---|:--:|---|
| fp32 / bf16 | ~0 | ✅ | 기본 권장 — bf16은 메모리 절반·동일 품질 |
| Q8 / int8 | 미미 (±0.006) | ✅ | 안전, 크기 1/2 |
| Q4 (K-quant·QAT) | ~0 | ✅ | 권장 압축점 — 크기 1/3~1/4 (8B→4.7GB) |
| Q3 | −2pp~ | ⚠️ | 주의 — 소형 모델일수록 위험 |
| Q2 | 붕괴 (절반) | ❌ | 금지 |
| fp16 (Gemma 계열) | NaN | ❌ | 금지 — activation 오버플로 |

**운영 권고**: 기본 **bf16**, 메모리 압박 시 **Q8(무손실)→Q4(거의 무손실)**, **Q3 이하·Gemma fp16 금지**. 8B급은 Q4가 가장 공격적이면서 안전.

> 측정: `scripts/bench_phase4_parallel.py`(ST, fp32/bf16/fp16) + 공식 GGUF(llama.cpp `--embedding`, Q8~Q2). 전 26종 dtype 표·곡선 raw는 결과 산출물 참조.

## 핵심 관찰

1. **상위권은 오픈/API 혼전**: 1위 `voyage-3-large`(0.687, API)와 2위 `kure-v1`(0.669, 오픈)을 필두로 pixie·gemini-001·snowflake-ko·cohere·qwen3-4b/8b가 **0.65대에 군집**. 차원·오픈/클로즈보다 한국어 정렬이 지배적.
2. **pooling 교정의 충격**: 초기 게이트웨이가 pooling을 미지정(mean)해 `qwen3-embed-8b`가 0.527로 바닥권이었으나, 올바른 last-pooling raw 재측정 시 **0.650(+0.123)**으로 상위권 진입. `qwen3-4b`·`bge-m3`도 +0.06. **측정 설정이 모델 순위를 좌우할 수 있음**을 보여주는 사례.
3. **한국어 특화 강세**: kure-v1·pixie-rune-v1·snowflake-arctic-ko·koe5·gemma-300m 모두 상위권(0.63~0.67).
4. **상용 API ≈ 오픈**: voyage-3-large는 최상위지만, gemini-001·cohere-v4·upstage·openai-3-large는 한국어 오픈 모델과 동급이거나 낮음. 차원·브랜드보다 한국어 정렬이 우세. 최신 `gemini-embedding-2`(0.621)가 구버전 `001`(0.652)보다 낮은 현상 지속.
5. **raw 기준의 한계 = 모델 특성**: instruction/prompt 의존 모델(`e5-mistral-7b` 0.202, `mxbai-embed-large` 0.153)은 raw에서 약함. 이는 측정 오류가 아니라 "프롬프트 없이는 약한" 모델 특성.
6. **중복 모델**: `nemotron-embed-8b` = `llama-embed-nemotron-8b` (동일 가중치, MRR 0.4404).
7. **harrier-27b 교정(0.017→0.618)**: 구 0.017은 GGUF/fp16 경로의 **NaN 버그**였고, bf16/fp32 정상 측정 시 **0.6177**(#17). Gemma3-27B의 fp16-NaN을 확증한 사례 (↑ 정밀도·양자화 절).

## 방법론 통일 작업 (2026-06-04)

| 항목 | 영향 |
|---|---|
| 전 모델 raw 통일 (프롬프트/query·doc 무구분) | 오픈·API 동일 footing |
| 게이트웨이 pooling 미지정 → 네이티브 pooling 교정 | qwen3-8b +0.123, qwen3-4b +0.066, bge-m3 +0.060 |
| 512-context 모델 max_seq_length 절단 | granite/me5/mxbai 등 정상화 |
| 모델별 올바른 라이브러리 버전 | gemma(양방향) / jina-v4(task adapter) 정확 로드 |
| 상용 API query/doc task 제거(raw) | upstage -0.045, cohere -0.017, 나머지는 동일 |

## 결론

- **최고 정확도(비용 무관)**: `voyage-3-large` (API, 0.687)
- **오픈 1순위 / 메인 권장**: `kure-v1` (0.669) 또는 `pixie-rune-v1`(0.658)·`snowflake-arctic-ko`(0.652)
- **소형 인프라 / 본 벤치 운영**: `gemma-embed-300m` (768d, 0.635, 빠른 latency) — Stage 4 이후 모든 stage에서 사용
- **대형 오픈 경쟁력 확인**: `qwen3-embed-8b/4b`(0.65)는 올바른 pooling 시 상용 API와 동급 — 초기 과소평가는 측정 설정 문제였음
- **상용 API**: voyage-3-large는 최상위, 그 외 gemini/cohere/upstage/openai는 한국어 오픈 모델과 동급 — 비용·외부 의존을 감수할 만한 압도적 우위는 없음

## 레퍼런스

- KoE5 — [HF](https://huggingface.co/nlpai-lab/KoE5) (Jang, Son, Lee 2024)
- KURE-v1 — [HF](https://huggingface.co/nlpai-lab/KURE-v1)
- EmbeddingGemma 300m — [HF](https://huggingface.co/google/embeddinggemma-300m)
- Snowflake Arctic Embed L v2 — [HF](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)
- BGE-M3 — Chen et al. 2024 [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
- Jina embeddings v4 — [HF](https://huggingface.co/jinaai/jina-embeddings-v4)
- Qwen3 Embedding — [HF](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
- Voyage / Cohere / Upstage / OpenAI / Gemini embeddings — 각 공식 API
- 평가: 로컬 `scripts/bench_phase4_parallel.py`, raw 재측정 `scripts/bench_embedding_api_raw.py` (API) + sentence-transformers(로컬), 결과 `results/phase4_embedding_final/`
