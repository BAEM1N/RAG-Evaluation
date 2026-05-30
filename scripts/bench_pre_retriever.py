#!/usr/bin/env python3
"""Pre-retriever 비교 — Stage 4-1 (LangChain + Azure GPT-5.4).

고정 파이프라인 (Stage 1~4 winner):
  Loader   : PyMuPDFLoader (LangChain)
  Parser   : RecursiveCharacterTextSplitter(300, 50)
  Embedding: HuggingFaceEmbeddings(google/embeddinggemma-300m)
  Retriever: EnsembleRetriever(FAISS dense + BM25Retriever[KIWI], weights=[0.3, 0.7])

LLM (Pre-retriever 변형용): Azure GPT-5.4 (AI Foundry endpoint), via langchain_openai.ChatOpenAI

10 전략:
  baseline             — 원 query 그대로
  hyde                 — LLM 가상답안 → dense 쿼리 (BM25는 원 query)
  hyde_rrf             — 원 query + HyDE 답안, 둘 다 hybrid 검색 후 RRF
  query2doc            — 원 query + LLM 가상문서 concat → hybrid
  multi_query_para     — LLM 3 paraphrase → 각 hybrid → RRF
  multi_query_angle    — LLM 3 각도(구체/추상/어휘) 변형 → 각 hybrid → RRF
  step_back            — LLM 추상화 → 원본+추상 둘 다 검색 RRF
  decompose            — LLM sub-question 분해 → 각 검색 RRF
  query_expansion      — LLM 키워드 추가 → concat → hybrid
  query_rewrite        — LLM 단일 명확화 쿼리 → hybrid (원 query 대체)

Usage:
  python scripts/bench_pre_retriever.py --strategies all --dry-run
  python scripts/bench_pre_retriever.py --strategies all --pilot
  python scripts/bench_pre_retriever.py --strategies all
"""
import argparse, json, time, sys, os, re
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

LLM_WORKERS = int(os.environ.get("LLM_WORKERS", "20"))

# .env 로드 (Azure 자격증명)
ROOT = Path(__file__).parent.parent
for ln in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(ROOT / "scripts"))
from eval_utils import parse_pdf_pymupdf, load_ground_truth

PDF_DIR = ROOT / "data/pdfs"
OUT_DIR = ROOT / "results/phase4_1_pre_retriever"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LLM_CACHE_DIR = OUT_DIR / "_llm_cache_gpt54"
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "google/embeddinggemma-300m"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RRF_K = 60
TOP_K = 5

LLM_MODEL = "gpt-5.4"  # Azure deployment name


# ── chunk / index setup ────────────────────────────────────────────────

def find_pdf(filename):
    for root, _dirs, files in os.walk(PDF_DIR):
        if filename in files:
            return Path(root) / filename
    return None


def build_chunks(gt_full):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks_text, meta = [], []
    for fname in sorted({g["target_file_name"] for g in gt_full}):
        path = find_pdf(fname)
        if not path:
            continue
        for p in parse_pdf_pymupdf(str(path)):
            for part in splitter.split_text(p["text"]):
                part = part.strip()
                if len(part) >= 30:
                    chunks_text.append(part)
                    meta.append({"file": fname, "page": p["page"]})
    return chunks_text, meta


def kiwi_tokenize(text):
    if not hasattr(kiwi_tokenize, "_k"):
        from kiwipiepy import Kiwi
        kiwi_tokenize._k = Kiwi()
    keep = ("N", "V", "X", "S", "M")
    return [t.form for t in kiwi_tokenize._k.tokenize(text) if t.tag.startswith(keep)]


# ── Azure GPT-5.4 via LangChain ────────────────────────────────────────

_llm = None


def get_llm():
    """ChatOpenAI(base_url=AI Foundry endpoint) — Azure GPT-5.4."""
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        project = os.environ["AZURE_OPENAI_PROJECT"]
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            base_url=f"{endpoint}/api/projects/{project}/openai/v1",
            temperature=0.3,
            max_tokens=400,
            timeout=60,
            max_retries=3,
        )
    return _llm


def cache_call(strategy, qid, prompt_str, temperature=0.3, max_tokens=400):
    p = LLM_CACHE_DIR / f"{strategy}_{qid}.json"
    if p.exists():
        return json.load(open(p))["output"]
    from langchain_core.messages import HumanMessage
    llm = get_llm()
    out = llm.bind(temperature=temperature, max_tokens=max_tokens).invoke([HumanMessage(content=prompt_str)]).content.strip()
    json.dump({"prompt": prompt_str, "output": out}, open(p, "w"), ensure_ascii=False, indent=2)
    return out


def warm_cache_parallel(jobs, label=""):
    """jobs: list of (strategy, qid, prompt_str, temperature, max_tokens). Fills disk cache concurrently."""
    todo = [j for j in jobs if not (LLM_CACHE_DIR / f"{j[0]}_{j[1]}.json").exists()]
    if not todo:
        return
    print(f"  warming cache: {len(todo)} LLM calls (workers={LLM_WORKERS}) {label}", flush=True)
    t0 = time.time()
    done = 0
    def _do(job):
        s, qid, prompt, temp, mx = job
        try:
            cache_call(s, qid, prompt, temperature=temp, max_tokens=mx)
            return True
        except Exception as e:
            print(f"    err {s}/{qid}: {e}")
            return False
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        futures = {ex.submit(_do, j): j for j in todo}
        for fut in as_completed(futures):
            done += 1
            if done % 50 == 0:
                rate = done / (time.time() - t0)
                print(f"    {done}/{len(todo)}  ({time.time()-t0:.0f}s, {rate:.1f}/s)", flush=True)
    print(f"  cache warmed: {done}/{len(todo)} in {time.time()-t0:.1f}s", flush=True)


# ── prompts (동일, GPT-5.4용으로 미세 보강) ─────────────────────────

HYDE_PROMPT = """다음 한국어 질문에 대해, 정답이 적혀있을 법한 보고서의 **한 문단**을 한국어로 작성하세요.
- 실제 사실 여부와 무관하게, 답을 담은 문서 본문 같은 톤
- 2-4문장, 핵심 개체와 숫자 포함
- 도입부 없이 본문만

질문: {q}

답안 문단:"""

QUERY2DOC_PROMPT = """다음 질문에 대한 **간단한 검색용 의사 문서**를 한국어로 생성하세요.
- 정답 키워드를 자연스럽게 포함한 1-2문장
- "다음과 같습니다" 같은 메타 표현 금지

질문: {q}

의사 문서:"""

MULTIQ_PARA_PROMPT = """아래 한국어 검색 질문을 의미는 유지한 채 **3가지** 다른 표현으로 바꿔주세요.
- 동의어/조사/어순/관점을 다양화
- 각 줄에 하나씩, 줄번호나 글머리표 없이
- 정확히 3줄

원 질문: {q}

변형 질문:"""

MULTIQ_ANGLE_PROMPT = """아래 한국어 검색 질문을 **3가지 각도**로 변형하세요. 각 줄에 하나씩, 라벨이나 글머리표 없이:
- 첫째 줄: 더 **구체적인** 질문 (특정 개체/조건 추가)
- 둘째 줄: 더 **추상적이고 일반적인** 질문 (상위 개념으로)
- 셋째 줄: **다른 어휘**로 같은 의도 (전문 용어 ↔ 일반어)

원 질문: {q}

변형:"""

STEPBACK_PROMPT = """다음 한국어 질문이 다루는 **상위 개념의 추상적 질문**을 한 문장으로 작성하세요.
- 구체적 개체·숫자·날짜를 제거하고 본질만 남기기
- 한 문장으로

원 질문: {q}

추상 질문:"""

DECOMPOSE_PROMPT = """다음 한국어 질문을 **검색을 더 쉽게 만들 sub-question 2~3개**로 분해하세요.
- 각 sub-question은 짧고 독립적
- 한 줄에 하나, 줄번호나 글머리표 없이
- 최대 3줄

원 질문: {q}

분해:"""

EXPAND_PROMPT = """다음 한국어 질문의 검색 품질을 높이기 위한 **관련 키워드 5개**를 콤마로 구분해서 출력하세요.
- 동의어, 상위/하위 개념, 정답에 등장할 만한 용어
- 키워드만, 설명·따옴표·번호 없이

원 질문: {q}

키워드:"""

REWRITE_PROMPT = """다음 한국어 검색 질문을 **검색 엔진에 더 잘 맞는 한 문장**으로 재작성하세요.
- 모호한 표현 명확화 (대명사·시제·범위 구체화)
- 정답에 등장할 법한 핵심 용어 우선 사용
- 한 문장, 도입부·따옴표 없이 재작성문만

원 질문: {q}

재작성:"""


def parse_lines(text, n, fallback):
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in lines]
    lines = [ln for ln in lines if len(ln) > 2][:n]
    while len(lines) < n:
        lines.append(fallback)
    return lines


def parse_keywords(text):
    raw = text.replace("\n", ",").split(",")
    kws = [w.strip() for w in raw if w.strip()]
    kws = [re.sub(r"^[\-\*\d\.\)\s]+", "", w).strip() for w in kws]
    return [w for w in kws if 1 < len(w) < 30][:5]


# ── retrieval primitives (numpy fast-path, 결과 동일) ──────────────────

_dense_model = None
_chunk_emb = None
_bm25 = None


def get_dense_model():
    from sentence_transformers import SentenceTransformer
    global _dense_model
    if _dense_model is None:
        print(f"  loading dense model {EMBED_MODEL}...", flush=True)
        _dense_model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    return _dense_model


def precompute_chunks(chunks_text):
    global _chunk_emb, _bm25
    from rank_bm25 import BM25Okapi
    model = get_dense_model()
    print(f"  embedding {len(chunks_text)} chunks...", flush=True)
    _chunk_emb = model.encode(chunks_text, batch_size=16, show_progress_bar=True,
                              normalize_embeddings=True, convert_to_numpy=True)
    print(f"  tokenizing for BM25-KIWI...", flush=True)
    _bm25 = BM25Okapi([kiwi_tokenize(t) for t in chunks_text])


def hybrid_scores(q_dense_text, q_bm25_text):
    model = get_dense_model()
    q_emb = model.encode([q_dense_text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    sims = (q_emb @ _chunk_emb.T)[0]
    dense_rank = np.argsort(-sims)
    bm25_scores = _bm25.get_scores(kiwi_tokenize(q_bm25_text))
    bm25_rank = np.argsort(-bm25_scores)
    n_c = len(sims)
    rank_d = np.empty(n_c, dtype=np.int64); rank_b = np.empty(n_c, dtype=np.int64)
    rank_d[dense_rank] = np.arange(n_c); rank_b[bm25_rank] = np.arange(n_c)
    return 0.3 / (RRF_K + rank_d + 1) + 0.7 / (RRF_K + rank_b + 1)


def fuse_and_topk(score_list, top_k=TOP_K):
    fused = np.zeros_like(score_list[0])
    for s in score_list:
        fused += s
    top = np.argpartition(-fused, top_k)[:top_k]
    return top[np.argsort(-fused[top])]


# ── strategy execution ────────────────────────────────────────────────

def run_strategy(strategy, gt):
    n_q = len(gt)
    topk = np.zeros((n_q, TOP_K), dtype=np.int64)
    t0 = time.time()
    llm_seconds = 0.0; llm_calls = 0

    for i, g in enumerate(gt):
        q = g["question"]
        qid = f"q{i:04d}"

        if strategy == "baseline":
            topk[i] = fuse_and_topk([hybrid_scores(q, q)])

        elif strategy == "hyde":
            tll = time.time()
            ans = cache_call("hyde", qid, HYDE_PROMPT.format(q=q), max_tokens=300)
            llm_seconds += time.time() - tll; llm_calls += 1
            topk[i] = fuse_and_topk([hybrid_scores(ans, q)])

        elif strategy == "hyde_rrf":
            tll = time.time()
            ans = cache_call("hyde", qid, HYDE_PROMPT.format(q=q), max_tokens=300)
            llm_seconds += time.time() - tll; llm_calls += 1
            topk[i] = fuse_and_topk([hybrid_scores(q, q), hybrid_scores(ans, q)])

        elif strategy == "query2doc":
            tll = time.time()
            doc = cache_call("query2doc", qid, QUERY2DOC_PROMPT.format(q=q), max_tokens=200)
            llm_seconds += time.time() - tll; llm_calls += 1
            combined = f"{q} {doc}"
            topk[i] = fuse_and_topk([hybrid_scores(combined, combined)])

        elif strategy == "multi_query_para":
            tll = time.time()
            out = cache_call("multiq_para", qid, MULTIQ_PARA_PROMPT.format(q=q), max_tokens=400, temperature=0.5)
            llm_seconds += time.time() - tll; llm_calls += 1
            variants = parse_lines(out, 3, q)
            topk[i] = fuse_and_topk([hybrid_scores(q, q)] + [hybrid_scores(v, v) for v in variants])

        elif strategy == "multi_query_angle":
            tll = time.time()
            out = cache_call("multiq_angle", qid, MULTIQ_ANGLE_PROMPT.format(q=q), max_tokens=400, temperature=0.5)
            llm_seconds += time.time() - tll; llm_calls += 1
            variants = parse_lines(out, 3, q)
            topk[i] = fuse_and_topk([hybrid_scores(q, q)] + [hybrid_scores(v, v) for v in variants])

        elif strategy == "step_back":
            tll = time.time()
            abs_q = cache_call("stepback", qid, STEPBACK_PROMPT.format(q=q), max_tokens=150)
            llm_seconds += time.time() - tll; llm_calls += 1
            topk[i] = fuse_and_topk([hybrid_scores(q, q), hybrid_scores(abs_q, abs_q)])

        elif strategy == "decompose":
            tll = time.time()
            out = cache_call("decompose", qid, DECOMPOSE_PROMPT.format(q=q), max_tokens=300)
            llm_seconds += time.time() - tll; llm_calls += 1
            subs = parse_lines(out, 3, q)
            topk[i] = fuse_and_topk([hybrid_scores(q, q)] + [hybrid_scores(s, s) for s in subs])

        elif strategy == "query_expansion":
            tll = time.time()
            out = cache_call("expand", qid, EXPAND_PROMPT.format(q=q), max_tokens=200)
            llm_seconds += time.time() - tll; llm_calls += 1
            kws = parse_keywords(out)
            expanded = f"{q} {' '.join(kws)}"
            topk[i] = fuse_and_topk([hybrid_scores(expanded, expanded)])

        elif strategy == "query_rewrite":
            tll = time.time()
            rewritten = cache_call("rewrite", qid, REWRITE_PROMPT.format(q=q), max_tokens=200)
            llm_seconds += time.time() - tll; llm_calls += 1
            rewritten = rewritten.split("\n")[0].strip() or q
            topk[i] = fuse_and_topk([hybrid_scores(rewritten, rewritten)])

        else:
            raise ValueError(strategy)

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_q}  ({time.time()-t0:.1f}s)", flush=True)

    return topk, time.time() - t0, llm_seconds, llm_calls


# ── metrics ─────────────────────────────────────────────────────────────

def compute_metrics(topk_idx, meta, gt):
    hit1 = hit5 = file_hit5 = 0
    mrr = 0.0
    for i, g in enumerate(gt):
        tgt_f = g["target_file_name"]
        try:
            tgt_p = int(str(g["target_page_no"]).strip().split(",")[0].strip())
        except (ValueError, AttributeError):
            tgt_p = None
        found = None; file_found = False
        for rank, idx in enumerate(topk_idx[i]):
            m = meta[idx]
            if m["file"] == tgt_f:
                file_found = True
                if tgt_p is None or m["page"] == tgt_p:
                    if found is None:
                        found = rank + 1
        if found == 1: hit1 += 1
        if found and found <= 5: hit5 += 1
        if file_found: file_hit5 += 1
        if found: mrr += 1.0 / found
    n = len(gt)
    return {"MRR": mrr / n, "Hit@1": hit1 / n, "Hit@5": hit5 / n, "File@5": file_hit5 / n}


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=["all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    gt_full = load_ground_truth()
    if args.dry_run:
        gt = gt_full[:1]; tag = "_dry"
    elif args.pilot:
        gt = gt_full[:100]; tag = "_pilot"
    else:
        gt = gt_full; tag = ""
    print(f"GT: {len(gt)} Q&A{tag} (chunks built over {len(gt_full)} files)")
    print(f"LLM: Azure {LLM_MODEL}")

    print("\n=== building chunks ===")
    chunks_text, meta = build_chunks(gt_full)
    precompute_chunks(chunks_text)

    ALL = ["baseline", "hyde", "hyde_rrf", "query2doc",
           "multi_query_para", "multi_query_angle", "step_back",
           "decompose", "query_expansion", "query_rewrite"]
    strategies = ALL if args.strategies == ["all"] else args.strategies

    # ── pre-warm LLM cache in parallel ─────────────────────────────────
    prompt_specs = {
        "hyde":         (HYDE_PROMPT,        300, 0.3),
        "query2doc":    (QUERY2DOC_PROMPT,   200, 0.3),
        "multiq_para":  (MULTIQ_PARA_PROMPT, 400, 0.5),
        "multiq_angle": (MULTIQ_ANGLE_PROMPT,400, 0.5),
        "stepback":     (STEPBACK_PROMPT,    150, 0.3),
        "decompose":    (DECOMPOSE_PROMPT,   300, 0.3),
        "expand":       (EXPAND_PROMPT,      200, 0.3),
        "rewrite":      (REWRITE_PROMPT,     200, 0.3),
    }
    strategy_to_prompt = {
        "hyde":             "hyde",
        "hyde_rrf":         "hyde",  # shared cache
        "query2doc":        "query2doc",
        "multi_query_para": "multiq_para",
        "multi_query_angle":"multiq_angle",
        "step_back":        "stepback",
        "decompose":        "decompose",
        "query_expansion":  "expand",
        "query_rewrite":    "rewrite",
    }
    jobs = []
    for s in strategies:
        if s == "baseline": continue
        pk = strategy_to_prompt[s]
        if pk not in prompt_specs: continue
        tmpl, mx, temp = prompt_specs[pk]
        for i, g in enumerate(gt):
            qid = f"q{i:04d}"
            jobs.append((pk, qid, tmpl.format(q=g["question"]), temp, mx))
    # dedupe (hyde + hyde_rrf share pk='hyde')
    seen = set(); uniq_jobs = []
    for j in jobs:
        key = (j[0], j[1])
        if key in seen: continue
        seen.add(key); uniq_jobs.append(j)
    warm_cache_parallel(uniq_jobs, label=f"({len(uniq_jobs)} unique prompts)")

    results = {}
    for s in strategies:
        out_path = OUT_DIR / f"{s}{tag}.json"
        if args.skip_existing and out_path.exists():
            print(f"SKIP {s}{tag} (cached)")
            results[s] = json.load(open(out_path)); continue
        print(f"\n=== {s} ===")
        topk, elapsed, llm_secs, llm_calls = run_strategy(s, gt)
        m = compute_metrics(topk, meta, gt)
        r = {"strategy": s, "llm_model": LLM_MODEL, "llm_provider": "azure_ai_foundry",
             "n_q": len(gt),
             "elapsed_sec": elapsed, "llm_seconds": llm_secs, "llm_calls": llm_calls, **m}
        results[s] = r
        json.dump(r, open(out_path, "w"), ensure_ascii=False, indent=2)
        print(f"  MRR={m['MRR']:.4f} Hit@1={m['Hit@1']:.3f} Hit@5={m['Hit@5']:.3f} File@5={m['File@5']:.3f}  "
              f"({elapsed:.1f}s, llm {llm_secs:.1f}s / {llm_calls} calls)")

    print("\n" + "=" * 100)
    print(f"{'Strategy':<22} {'MRR':>7} {'Hit@1':>7} {'Hit@5':>7} {'File@5':>7} {'LLM(s)':>8} {'Calls':>6}")
    print("-" * 100)
    for s in sorted(results.keys(), key=lambda x: results[x]["MRR"], reverse=True):
        r = results[s]
        print(f"{s:<22} {r['MRR']:>7.4f} {r['Hit@1']:>7.3f} {r['Hit@5']:>7.3f} {r['File@5']:>7.3f} "
              f"{r.get('llm_seconds',0):>8.1f} {r.get('llm_calls',0):>6}")


if __name__ == "__main__":
    main()
