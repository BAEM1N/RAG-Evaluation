#!/usr/bin/env python3
"""API 임베딩 PROMPTED 재측정 — query/document input_type 구분(각 API 권장 방식).
raw 버전과 동일 데이터·metric. 출력: results/phase4_embedding_prompted/<alias>.json
openai는 input_type 개념 없음 → raw와 동일(스킵).
"""
import os, sys, json, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_utils import load_ground_truth
from bench_phase4_parallel import evaluate_model

CHUNKS = ROOT / "data" / "prepared_chunks.json"
OUT = ROOT / "results" / "phase4_embedding_prompted"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "voyage-3-large":        ("voyage", "voyage-3-large", 1024),
    "voyage-multilingual-2": ("voyage", "voyage-multilingual-2", 1024),
    "cohere-embed-v4":       ("cohere", "embed-v4.0", 1536),
    "upstage-solar-large":   ("upstage", None, 4096),
    "gemini-embed-001":      ("gemini", "gemini-embedding-001", 3072),
    "gemini-embed-2":        ("gemini", "gemini-embedding-2", 3072),
}


def embed_voyage(model, texts, dim, is_query):
    import voyageai
    c = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    it = "query" if is_query else "document"
    out = []
    for i in range(0, len(texts), 128):
        out += c.embed(texts[i:i+128], model=model, input_type=it).embeddings
    return out


def embed_cohere(model, texts, dim, is_query):
    import cohere
    c = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY") or os.environ["CO_API_KEY"])
    it = "search_query" if is_query else "search_document"
    out = []; i = 0
    while i < len(texts):
        for attempt in range(8):
            try:
                r = c.embed(texts=texts[i:i+48], model=model, input_type=it,
                            embedding_types=["float"], output_dimension=dim)
                out += [list(v) for v in r.embeddings.float]; break
            except Exception as e:
                if "TooMany" in type(e).__name__ or "429" in str(e): time.sleep(min(60,5*(attempt+1))); continue
                raise
        else: raise RuntimeError("cohere rate-limit")
        i += 48; time.sleep(1.2)
    return out


def embed_upstage(model, texts, dim, is_query):
    from openai import OpenAI
    c = OpenAI(api_key=os.environ["UPSTAGE_API_KEY"], base_url="https://api.upstage.ai/v1")
    mdl = "embedding-query" if is_query else "embedding-passage"
    out = []
    for i in range(0, len(texts), 100):
        out += [d.embedding for d in c.embeddings.create(model=mdl, input=texts[i:i+100]).data]
    return out


def embed_gemini(model, texts, dim, is_query):
    from google import genai
    from google.genai import types
    c = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])
    task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
    cfg = types.EmbedContentConfig(output_dimensionality=dim, task_type=task)
    def call(contents):
        for attempt in range(6):
            try: return c.models.embed_content(model=model, contents=contents, config=cfg).embeddings
            except Exception as e:
                if any(x in str(e) for x in ("503","429","500","UNAVAILABLE","RESOURCE_EXHAUSTED")): time.sleep(min(30,3*(attempt+1))); continue
                raise
        raise RuntimeError("gemini retry exhausted")
    out=[]; i=0
    while i < len(texts):
        batch=texts[i:i+100]; embs=call(batch)
        if len(embs)!=len(batch): embs=[call([t])[0] for t in batch]
        out += [list(e.values) for e in embs]; i+=100
    return out


EMB = {"voyage": embed_voyage, "cohere": embed_cohere, "upstage": embed_upstage, "gemini": embed_gemini}


def main():
    sel = sys.argv[1].split(",") if len(sys.argv) > 1 else list(MODELS)
    gt = load_ground_truth(); chunks = json.load(open(CHUNKS))
    texts = [c["text"] for c in chunks]; qs = [g["question"] for g in gt]
    print(f"청크 {len(chunks)} · 질문 {len(gt)}", flush=True)
    for alias in sel:
        if (OUT / f"{alias}.json").exists(): print(f"[skip] {alias}", flush=True); continue
        prov, mid, dim = MODELS[alias]
        print(f"\n=== {alias} ({prov}, prompted) ===", flush=True)
        try:
            t0=time.time(); fn=EMB[prov]
            ce = fn(mid, texts, dim, False)
            qe = fn(mid, qs, dim, True)
            res = evaluate_model(alias, {"dim": dim, "vram": 0.0}, gt, chunks,
                                 {"chunk_embeddings": ce, "query_embeddings": qe, "embed_time": time.time()-t0})
            res["engine"]="api-prompted"; res["model"]=alias; res["dim"]=dim
            json.dump(res, open(OUT/f"{alias}.json","w"), ensure_ascii=False, indent=1)
            print(f"[OK] {alias} MRR={res['metrics']['mrr']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[FAIL] {alias}: {repr(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
