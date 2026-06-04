#!/usr/bin/env python3
"""API 임베딩 RAW 재측정 — 로컬 FINAL(raw)과 방법론 통일.
raw = query/document 구분 없음. input_type 필수 API는 양쪽 동일 타입(차별 제거).
동일 prepared_chunks(3166) + 300 gt + page-match(evaluate_model). 출력: phase4_embedding_final/<alias>.json
"""
import os, sys, json, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_utils import load_ground_truth
from bench_phase4_parallel import evaluate_model

CHUNKS = ROOT / "data" / "prepared_chunks.json"
OUT = ROOT / "results" / "phase4_embedding_final"
OUT.mkdir(parents=True, exist_ok=True)

# alias -> (provider, model_id, dim)
MODELS = {
    "voyage-3-large":        ("voyage", "voyage-3-large", 1024),
    "voyage-multilingual-2": ("voyage", "voyage-multilingual-2", 1024),
    "cohere-embed-v4":       ("cohere", "embed-v4.0", 1536),
    "upstage-solar-large":   ("upstage", "embedding-passage", 4096),
    "gemini-embed-001":      ("gemini", "gemini-embedding-001", 3072),
    "gemini-embed-2":        ("gemini", "gemini-embedding-2", 3072),
    "openai-embed-3-large":  ("openai", "text-embedding-3-large", 3072),
    "openai-embed-3-small":  ("openai", "text-embedding-3-small", 1536),
}


def embed_voyage(model, texts, dim):
    import voyageai
    c = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    out = []
    for i in range(0, len(texts), 128):
        out += c.embed(texts[i:i+128], model=model).embeddings  # input_type 생략 = raw
        print(f"    voyage {min(i+128,len(texts))}/{len(texts)}", flush=True)
    return out


def embed_cohere(model, texts, dim):
    import cohere
    c = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY") or os.environ["CO_API_KEY"])
    out = []
    B=48
    i=0
    while i < len(texts):
        for attempt in range(8):
            try:
                r = c.embed(texts=texts[i:i+B], model=model, input_type="search_document",
                            embedding_types=["float"], output_dimension=dim)
                out += [list(v) for v in r.embeddings.float]
                break
            except Exception as e:
                if "TooMany" in type(e).__name__ or "429" in str(e):
                    w=min(60, 5*(attempt+1)); print(f"    cohere 429 backoff {w}s", flush=True); time.sleep(w)
                else: raise
        else:
            raise RuntimeError("cohere rate-limit exhausted")
        print(f"    cohere {min(i+B,len(texts))}/{len(texts)}", flush=True)
        i+=B; time.sleep(1.2)
    return out


def embed_upstage(model, texts, dim):
    from openai import OpenAI
    c = OpenAI(api_key=os.environ["UPSTAGE_API_KEY"], base_url="https://api.upstage.ai/v1")
    out = []
    for i in range(0, len(texts), 100):
        r = c.embeddings.create(model=model, input=texts[i:i+100])  # 양쪽 동일 'embedding-passage' = raw 등가
        out += [d.embedding for d in r.data]
        print(f"    upstage {min(i+100,len(texts))}/{len(texts)}", flush=True)
    return out


def embed_gemini(model, texts, dim):
    from google import genai
    from google.genai import types
    c = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])
    cfg = types.EmbedContentConfig(output_dimensionality=dim)
    def call(contents):
        for attempt in range(6):
            try:
                return c.models.embed_content(model=model, contents=contents, config=cfg).embeddings
            except Exception as e:
                msg=str(e)
                if any(x in msg for x in ("503","429","500","UNAVAILABLE","RESOURCE_EXHAUSTED","deadline")):
                    w=min(30,3*(attempt+1)); time.sleep(w); continue
                raise
        raise RuntimeError("gemini retry exhausted")
    out=[]; i=0
    while i < len(texts):
        batch=texts[i:i+100]
        embs=call(batch)
        if len(embs)!=len(batch):
            embs=[call([t])[0] for t in batch]  # 배치 미지원 모델 단건 폴백
        out += [list(e.values) for e in embs]
        print(f"    gemini {min(i+100,len(texts))}/{len(texts)}", flush=True)
        i+=100
    return out


def embed_openai(model, texts, dim):
    from openai import OpenAI
    c = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out = []
    for i in range(0, len(texts), 256):
        r = c.embeddings.create(model=model, input=texts[i:i+256], dimensions=dim)  # raw
        out += [d.embedding for d in r.data]
        print(f"    openai {min(i+256,len(texts))}/{len(texts)}", flush=True)
    return out


EMB = {"voyage": embed_voyage, "cohere": embed_cohere, "upstage": embed_upstage,
       "gemini": embed_gemini, "openai": embed_openai}


def main():
    sel = sys.argv[1].split(",") if len(sys.argv) > 1 else list(MODELS)
    gt = load_ground_truth()
    chunks = json.load(open(CHUNKS))
    texts = [c["text"] for c in chunks]
    qs = [g["question"] for g in gt]
    print(f"청크 {len(chunks)} · 질문 {len(gt)}", flush=True)
    for alias in sel:
        if (OUT / f"{alias}.json").exists():
            print(f"[skip] {alias}", flush=True); continue
        prov, mid, dim = MODELS[alias]
        print(f"\n=== {alias} ({prov}:{mid}, raw) ===", flush=True)
        try:
            t0 = time.time()
            fn = EMB[prov]
            ce = fn(mid, texts, dim)
            qe = fn(mid, qs, dim)
            res = evaluate_model(alias, {"dim": dim, "vram": 0.0}, gt, chunks,
                                 {"chunk_embeddings": ce, "query_embeddings": qe, "embed_time": time.time()-t0})
            res["engine"] = "api-raw"; res["model"] = alias; res["dim"] = dim
            json.dump(res, open(OUT / f"{alias}.json", "w"), ensure_ascii=False, indent=1)
            print(f"[OK] {alias} MRR={res['metrics']['mrr']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[FAIL] {alias}: {repr(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
