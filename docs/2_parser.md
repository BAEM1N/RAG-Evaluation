# Parser — text → chunk 분할

> **데이터셋**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs)
>
> **고정**: loader=pymupdf (Stage 1 winner), embedding=`google/embeddinggemma-300m`, top-k=5
>
> **측정**: MRR / Hit@1 / Hit@5 / File@5

Loader가 추출한 raw text를 임베딩 단위 chunk로 자르는 단계. LangChain `TextSplitter`, LlamaIndex `NodeParser`, Chonkie `Chunker` 등 라이브러리에 해당.

**총 42종 chunker 비교**. 측정 환경이 두 그룹으로 나뉨:
- **그룹 A (Char-based 32종)**: Dense retrieval only — 라이브러리 × chunk_size 매트릭스
- **그룹 B (Semantic + LLM 10종)**: Hybrid 3:7 retrieval (Stage 4 winner) — Slumber, semantic chunker 등 비싼 chunker

(두 그룹은 retrieval 베이스라인이 달라 직접 비교 불가. 그룹 B 결과는 그룹 A의 LC Recursive 300/50 hybrid 기준값 0.7171과 비교.)

---

## 1. 그룹 A — Char-based 32종 (Dense baseline)

라이브러리 × chunk size 두 축. dense-only retrieval로 chunker 효과만 측정.

### Top 10 (MRR 기준)

| 순위 | Chunker | size | Chunks | Parse(s) | MRR | Hit@1 | Hit@5 | File@5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 🥇 | **Chonkie Fast** | 800 | 3,398 | 3.0 | **0.6903** | 60.7% | 80.7% | 90.7% |
| 🥈 | Chonkie Recursive | 300 | 4,310 | 3.5 | 0.6885 | 61.3% | 80.0% | 91.7% |
| 🥉 | Chonkie Sentence | 300/50 | 4,542 | 3.7 | 0.6881 | 60.3% | 80.3% | 91.7% |
| 4 | Chonkie Fast | 300 | 8,773 | 3.1 | 0.6831 | 61.0% | 79.7% | 92.0% |
| 5 | LC Recursive | 300/50 | 4,556 | 4.6 | 0.6816 | 59.0% | **81.3%** | 91.7% |
| 6 | LC Token | 256/50 tok | 5,091 | 4.9 | 0.6798 | 60.3% | 78.7% | **92.0%** |
| 7 | LC Character | 300/50 | 4,559 | 3.4 | 0.6779 | 58.7% | 81.0% | 91.7% |
| 8 | Chonkie Fast | 500 | 5,186 | 4.6 | 0.6774 | 59.3% | 79.7% | **92.0%** |
| 9 | LlamaIndex Sentence | 500/100 | 2,858 | 5.2 | 0.6774 | 60.7% | 79.0% | 90.7% |
| 10 | Chonkie Fast | 1000 | 2,835 | 3.0 | 0.6754 | 60.0% | 79.7% | 90.0% |

### chunk_size 효과 (단일 chunker grid)

| chunker | 300 | 500 | 800 | 1000 | 1500 |
|---|---:|---:|---:|---:|---:|
| **LC Recursive** | **0.6816** | 0.6573 | 0.6318 | 0.6334 | 0.6432 |
| **LC Character** | **0.6779** | 0.6604 | 0.6319 | 0.6332 | 0.6432 |
| **Chonkie Recursive** | **0.6885** | 0.6614 | 0.6416 | 0.6279 | 0.6438 |
| **Chonkie Sentence** | **0.6881** | 0.6538 | 0.6382 | 0.6326 | 0.6431 |
| **Chonkie Fast** | 0.6831 | 0.6774 | **0.6903** | 0.6754 | 0.6486 |
| **LlamaIndex Sentence** | 0.6721 | **0.6774** | 0.6563 | 0.6409 | 0.6381 |

→ **chunker마다 sweet spot 다름**:
- LC/Chonkie Recursive·Sentence·Character: **300/50 최적**
- Chonkie Fast: **800 최적** (Fast splitter의 특수 알고리즘이 큰 chunk에서도 의미 유지)
- LlamaIndex Sentence: **500/100 최적**

### Token 단위 chunker — Tokenizer 차이가 결정적

| Chunker | Tokenizer | Chunks | MRR |
|---|---|---:|---:|
| LC Token 256/50 | tiktoken cl100k_base | 5,091 | **0.6798** |
| Chonkie Token 256/50 | gpt2 | 2,377 | **0.4193** ❌ |

같은 파라미터인데 결과 정반대. gpt2 tokenizer가 한국어를 byte-level로 잘게 쪼개 청크가 짧고 의미 단편이 됨. **한국어에선 cl100k 계열 필수**.

### 그룹 A 결론

- **Chonkie Fast 800** 1위 (0.6903), **Chonkie Recursive 300** 2위 (0.6885), **LC Recursive 300/50** 5위 (0.6816)
- 격차는 작음 (1-9위 0.6754~0.6903 범위, 약 1.5pp)
- **chunk size > chunker library** 효과: 300/50 vs 1500/200 격차 7.6pp (LC Recursive 기준)
- Parse 시간은 모두 ~3-5초로 운영상 차이 없음

---

## 2. 그룹 B — Semantic + LLM-based 10종 (Hybrid 3:7 baseline)

embedding 호출 또는 LLM 호출이 필요한 비싼 chunker. Stage 4 winner(Hybrid 3:7)와 결합해 측정.

| 순위 | Chunker | Chunks | Parse(s) | MRR | Hit@1 | Hit@5 | File@5 |
|---:|---|---:|---:|---:|---:|---:|---:|
| (기준) | **LC Recursive 300/50** (그룹 A winner, hybrid 재측정) | 4,556 | 4.6 | **0.7171** | 65.3% | 80.3% | 91.7% |
| 🥇 | **Chonkie Slumber (gpt-5.4)** | 3,690 | 5,608 | **0.7112** | 65.0% | 80.3% | 91.0% |
| 🥈 | LlamaIndex SemanticSplitter (percentile=95) | 2,018 | 213 | 0.7076 | 64.0% | 79.7% | 90.3% |
| 🥉 | Kiwi + Recursive 500/100 | 2,704 | 9 | 0.7026 | 63.3% | 80.3% | 91.0% |
| 4 | LC SemanticChunker (percentile=95) | 2,026 | 156 | 0.7016 | 63.0% | 79.7% | 90.0% |
| 5 | Chonkie NeuralChunker | 2,816 | 74 | 0.6994 | 63.3% | 80.0% | 91.3% |
| 6 | Chonkie Semantic (text-embedding-3-large) | 6,044 | 2,191 | 0.6948 | 63.0% | 79.7% | **92.7%** |
| 7 | LC SemanticChunker (stdev=1.5) | 1,939 | 122 | 0.6945 | 62.0% | 79.7% | 90.3% |
| 8 | Chonkie SemanticChunker (gemma-300m, 500) | 5,976 | 396 | 0.6939 | 62.3% | 80.3% | **92.3%** |
| 9 | Chonkie Semantic (text-embedding-3-small) | 6,073 | 2,348 | 0.6909 | 62.0% | 79.3% | 92.0% |
| 10 | LC MarkdownHeaderTextSplitter (pymupdf4llm) | 2,754 | 1,083 | 0.6574 | 59.0% | 76.3% | 89.0% |
| ❌ | KSS + Recursive | — | timeout | — | — | — | — |

### 그룹 B 결론

- **모든 semantic/LLM chunker가 LC Recursive 300/50 (hybrid 0.7171)보다 낮음** — 비용 추가하고 성능 하락
- 그룹 B 안에선 Chonkie Slumber (LLM-based, gpt-5.4) 1위 (0.7112) — 의미 경계 LLM 결정의 가치
- **Chonkie Slumber vs LC Recursive 300/50: −0.59pp / +5,603초 parse / ~$2 LLM 비용** — 비용 대비 효과 없음
- LC MarkdownHeaderTextSplitter (0.6574) — pymupdf4llm 마크다운에 헤더 단서 부족
- KSS + Recursive — pecab 백엔드 시간초과 (mecab 권장, 본 벤치 timeout)

---

## 3. 통합 결론

| 항목 | 결과 |
|---|---|
| **Stage 2 winner** | **`RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)`** |
| **MRR (Dense baseline)** | 0.6816 |
| **MRR (Hybrid 3:7)** | 0.7171 (Stage 4 winner와 결합) |
| **차상위** | Chonkie Fast 800 / Chonkie Recursive 300 (격차 ~1pp) |
| **Semantic chunker** | 모두 character-based에 미달 |
| **LLM-based (Slumber)** | 그룹 B 1위지만 character-based 절대 winner에 못 미침 |

### 핵심 발견

1. **단순한 character-level recursive split이 가장 강함**: 한국어 단답형 RAG에서 chunker 정교화는 효과 없거나 비용 대비 손해
2. **chunk size > chunker library** 효과 (7.6pp vs 1.5pp)
3. **300/50이 대부분 chunker에서 최적** (Chonkie Fast 800, LlamaIndex Sentence 500/100 예외)
4. **Chonkie Slumber 의외성**: LLM 기반 chunking이 시간만 5,608초 + 비용 ~$2 추가했지만 −0.59pp
5. **LLM-기반 chunking이 본 데이터셋에선 비효율**: 단답형 factoid 질문에 의미 경계 별로 안 중요

### 다른 데이터셋에선 다를 수 있음

- **긴 서술형 / 멀티홉 질문**: 의미 경계 chunking 효과 클 수 있음
- **표/이미지 비중 큰 문서**: markdown header splitter 또는 docling 유효
- **단일 주제 (소설, 백과사전)**: 큰 chunk (1500+) 유리
- 본 결론은 **allganize 300 Q&A 단답형 dataset 범위**에만 유효

---

## 4. 다음 단계 고정 값

- **Loader**: `pymupdf` (PyMuPDFLoader)
- **Parser**: `RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)`
- (Stage 3 이후 stages는 위 chunker 고정)

---

## 5. 레퍼런스

### Char-based
- LC RecursiveCharacterTextSplitter — [API ref](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)
- LC CharacterTextSplitter — [API ref](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.CharacterTextSplitter.html)
- LC TokenTextSplitter (tiktoken) — [API ref](https://python.langchain.com/api_reference/text_splitters/base/langchain_text_splitters.base.TokenTextSplitter.html)
- Chonkie (TokenChunker / RecursiveChunker / SentenceChunker / FastChunker) — [github.com/chonkie-inc/chonkie](https://github.com/chonkie-inc/chonkie)
- LlamaIndex SentenceSplitter — [LlamaIndex docs](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/)

### Semantic / LLM-based
- LC SemanticChunker — Greg Kamradt's "5 Levels of Text Splitting" [참고](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb) · LangChain [API ref](https://api.python.langchain.com/en/latest/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html)
- LC MarkdownHeaderTextSplitter — [LangChain docs](https://python.langchain.com/api_reference/text_splitters/markdown/langchain_text_splitters.markdown.MarkdownHeaderTextSplitter.html)
- Chonkie SemanticChunker / NeuralChunker / **SlumberChunker** — [chonkie-inc/chonkie](https://github.com/chonkie-inc/chonkie) (Slumber: LumberChunker 변형, LLM이 chunk 경계 결정)
- LlamaIndex SemanticSplitterNodeParser — [LlamaIndex docs](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/semantic_splitter/)
- Kiwi 한국어 형태소/문장 분리 — [bab2min/Kiwi](https://github.com/bab2min/Kiwi)
- KSS — [hyunwoongko/kss](https://github.com/hyunwoongko/kss) (mecab backend 권장)

### 임베딩 / API
- text-embedding-3-small / 3-large (OpenAI) — [OpenAI docs](https://platform.openai.com/docs/guides/embeddings)
- google/embeddinggemma-300m — [HuggingFace](https://huggingface.co/google/embeddinggemma-300m)
- gpt-5.4 (Slumber genie) — OpenAI-compatible endpoint, json_schema structured output + reasoning_effort=none
