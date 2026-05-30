# Loader — PDF → text 변환

> **데이터셋**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs)

LangChain 분류 기준의 **Document Loader** 단계. PDF에서 raw text를 얼마나 잘 뽑아내는지가 retrieval 품질의 출발점이다.

## 실험 설정

| 항목 | 값 |
|---|---|
| 원본 PDF | 58 (1,337 페이지) |
| Chunking | 1000 / 200 (RecursiveCharacterTextSplitter, baseline 고정) |
| Embedding | `google/embeddinggemma-300m` (768d) |
| 검색 | cosine similarity, top-k = 5 |
| 측정 | MRR / Hit@1 / Hit@5 / File@5 (file 단위 매칭) |

page-level 매칭(`Hit@1/5`, `MRR`) + file-level 매칭(`File@5`) 모두 측정. docling과 opendataloader는 페이지 단위 export 후 평가.

## 7종 비교 결과

| Loader | PDFs | Chunks | Parse(s) | MRR | Hit@1 | Hit@5 | File@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **pymupdf** | 58 | 1,263 | **3.1** | **0.6486** | 57.0% | 76.3% | 90.0% |
| pdfplumber | 58 | 1,185 | 108.8 | 0.6468 | 56.3% | 77.0% | 88.7% |
| pymupdf4llm | 58 | 1,920 | 547.5 | 0.6388 | 54.7% | 77.3% | 90.3% |
| pdfminer | 58 | 1,949 | 144.9 | 0.6301 | 54.7% | 75.3% | 91.0% |
| docling | 57 | 1,667 | 1,162.5 | 0.6241 | 54.7% | 73.7% | 85.0% |
| pypdf | 58 | 1,224 | 32.9 | 0.6203 | 53.3% | 74.7% | 88.3% |
| **opendataloader** | 58 | 1,811 | 169.3 | 0.5993 | 50.0% | 75.3% | **91.7%** |

**선택: `pymupdf` (PyMuPDFLoader)** — 최고 MRR + 압도적 최고 속도 (다른 후보 대비 30~370배 빠름).

## 핵심 관찰

1. **단순한 평문 추출이 한국어 RAG에서 가장 강함**. `pymupdf` 평문이 `pymupdf4llm` markdown 보다 살짝 우위(MRR +1.0pp). 300M 작은 임베딩 모델에서는 markdown 마커가 오히려 noise로 작동할 가능성.
2. **MRR 1위~7위 격차 약 5.0pp** (0.6486 → 0.5993). 한국어 평문 추출 정확도가 어느 정도 평준화됨.
3. **속도 격차는 매우 큼**:
   - pymupdf: 3.1s
   - pypdf: 33s
   - pdfplumber: 109s
   - pdfminer: 145s
   - opendataloader: 169s
   - pymupdf4llm: 548s (markdown 변환 비용)
   - docling: 1,163s (OCR + IBM layout 모델, ~20분)
4. **File@5 기준 (loose 매칭)**:
   - opendataloader 91.7% 1위
   - pdfminer 91.0%
   - pymupdf4llm 90.3%
   - pymupdf 90.0%
   - 같은 file에 들어가는 비율은 비슷하지만 **정확한 page까지 맞추는 능력**에서 pymupdf가 우위.
5. **docling은 비용 vs 효과 불균형**: 1163s × 57 PDFs로 가장 느린데 MRR은 중하위. 표/이미지 비중 큰 문서에서는 장점일 수 있으나 이 데이터셋에서는 효과 미미. multimodal candidate 평가 시에 재검토 가치 있음.
6. **opendataloader는 file-level 최고지만 MRR 6위**: 페이지 분리가 약간 거칠어 retrieval 시 정확한 page 매칭에 손해.

## 다음 단계 고정 값

- **Loader: `pymupdf` (PyMuPDFLoader)**

> 기존 RESULTS.md에서는 pymupdf4llm을 winner로 표기했었으나, 본 재실험(공정한 page-level 평가 + embeddinggemma-300m 기준)에서는 **pymupdf 평문 추출이 가장 정확하고 빠름**으로 결과가 갱신되었다.

## LangChain wrapper

| Loader | LangChain class |
|---|---|
| pypdf | `PyPDFLoader` |
| pymupdf | `PyMuPDFLoader` |
| pymupdf4llm | `PyMuPDF4LLMLoader` |
| pdfplumber | `PDFPlumberLoader` |
| pdfminer | `PDFMinerLoader` |
| docling | `DoclingLoader` (`langchain-docling`) |
| opendataloader | `OpenDataLoaderPDFLoader` (`langchain-opendataloader-pdf`) |

## 추가 후보 (현재 미실험)

- **Vision OCR**: `Mineru`, `Marker` — 표/이미지 콘텐츠 추출에 강점
- **Multi-engine**: `Unstructured` (여러 backend 자동 선택)
- **Closed API**: `Document Intelligence`, `AWS Textract`
