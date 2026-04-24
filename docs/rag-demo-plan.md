# RAG 데모 웹 서비스 계획

**상태**: 설계 완료, 실험 완주 후 착수 대기
**서브도메인 후보**: `rag.baeum.ai.kr` 또는 `demo.baeum.ai.kr`

---

## 1. 목적

Phase 1~5 RAG 벤치마크 결과로 **가장 성능 좋은 로컬 조합**을 공개 웹에서 직접 체험 가능하게 제공.

- 브랜딩: baeum.ai.kr 시리즈 확장
- SEO/GEO: 실험 블로그 3편 + 데모 → 교차 유입
- 데이터 수집: 실사용자 👍/👎 피드백 → Phase 6 재료

---

## 2. 아키텍처

```
[사용자 브라우저]
      │ HTTPS
      ▼
[Cloudflare Tunnel] — rag.baeum.ai.kr
      │
      ▼
[Mac Mini 192.168.50.240]           [AI-395 192.168.50.245]
  ┌─────────────────────┐           ┌─────────────────────┐
  │ FastAPI + SSE       │ ── HTTP ▶ │ llama-server :9011  │
  │ FAISS 인덱스 (로컬) │           │   LLM 추론          │
  │ SQLite 로그         │ ── HTTP ▶ │ llama-server :9012  │
  │ Next.js / Gradio UI │           │   Embedding (KoE5)  │
  │ PM2 관리            │           │ VRAM 96GB           │
  └─────────────────────┘           └─────────────────────┘
```

- **Mac Mini 240 = 공개 프론트** (기존 baeum.ai.kr/ddok.ai 호스팅 인프라 재활용)
- **AI-395 245 = 모델 전용 서버** (외부 미노출, Mac Mini가 reverse proxy)

---

## 3. 모델 선정 (실험 결과 반영)

| 컴포넌트 | 선택 | 근거 |
|---|---|---|
| Parser | pymupdf4llm | Phase 1 MRR 0.4715 1위 |
| Chunking | `chunk_size=500, overlap=100` | Phase 2 MRR 0.5315 1위 |
| VectorStore | FAISS | Phase 3 p95 0.74ms (200배 빠름) |
| Embedding | **KoE5** (1024dim, 600M) | Phase 4 MRR 0.6871 1위 |
| LLM (생성) | **미정 — Phase 5 Judge 최종 결과 대기** | 후보: Qwen3.5-27B Q4/Q8, gpt-oss-120b, deepseek-r1:70b |
| top-k | 5 | allganize 기본 |

최종 리더보드 확정 후 LLM 1개 확정.

---

## 4. 사용자 흐름

1. 질문 입력
2. KoE5로 질의 임베딩 (AI-395 :9012)
3. Mac Mini 로컬 FAISS에서 top-5 청크 검색
4. 프롬프트 구성 (시스템 + top-5 컨텍스트 + 질문)
5. LLM 스트리밍 응답 (AI-395 :9011 → Mac Mini → 브라우저 SSE)
6. 참조 문서(top-5) 메타 동시 표시
7. 👍/👎 피드백 버튼 + GitHub 이슈 아카이브 옵션

---

## 5. 보안 & 운영

| 항목 | 설정 |
|---|---|
| Rate limit | Cloudflare WAF, IP당 10 req/hour |
| 응답 캐시 | 질문 해시 → 10분 캐시 |
| 로그 | SQLite: `question / answer / hits / latency / feedback` (일일 rotate) |
| AI-395 노출 | 외부 차단, Mac Mini에서만 접근 |
| 도메인 | Cloudflare Tunnel (기존 세팅 참고) |

---

## 6. 구현 단계

| # | 단계 | 소요 | 전제 |
|---|---|---|---|
| 1 | Phase 5 Judge 완주, 최종 LLM 결정 | 2-3일 | 진행 중 |
| 2 | AI-395: `llama-server` (LLM + Embedding) 상시 기동, systemd 등록 | 30분 | Phase 5 종료 |
| 3 | FAISS 인덱스 precompute, Mac Mini로 rsync | 1시간 | Phase 4 산출물 재활용 |
| 4 | Mac Mini: FastAPI + SSE 스트리밍 백엔드 | 4-6시간 | |
| 5 | UI (Next.js + shadcn 또는 Gradio) | 반나절 | |
| 6 | Cloudflare Tunnel + 서브도메인 | 30분 | 기존 baeum.ai.kr 설정 참고 |
| 7 | 로그·rate limit·피드백 | 2-3시간 | |
| **합계** | | **1-2일 작업** | |

---

## 7. 추가 가치

- 데모 페이지에 **실험 리더보드 탭** (Phase 4 임베딩 MRR 테이블 interactive)
- 블로그 3편 `rag-*-comparison.md` 과 상호 링크 → SEO 교차 유입
- 👍/👎 피드백으로 "실사용자 평가" 데이터셋 구축 (Phase 6 씨앗)

---

## 8. GitHub Issue 아카이브 (보조)

- 웹에서 "좋은 Q&A"를 GitHub 이슈로 영구 보존하는 옵션 버튼
- 제목 = 질문, 본문 = 답변 + 참조, 라벨 `rag-answer`
- 검색엔진 색인 → Long-tail SEO

---

## 9. 미결 항목

- [ ] 서브도메인 결정: `rag.baeum.ai.kr` vs `demo.baeum.ai.kr`
- [ ] UI 선택: Next.js (풀 커스텀) vs Gradio (빠른 배포)
- [ ] Phase 5 최종 결과에 따른 LLM 확정
- [ ] 피드백 수집 데이터 활용 방침 (공개/비공개)
