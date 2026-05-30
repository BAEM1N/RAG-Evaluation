#!/usr/bin/env python3
"""HuggingFace dataset 빌드 — Phase A + B.

산출물:
  data/qa.parquet              — 300 Q&A
  data/retrieval.parquet       — gemma-embed-300m top-5 per question
  data/cand_answers.parquet    — 46 cand × 300 q = 13,800 rows
  data/judge_scores.parquet    — long format (Q3+Q4) ~456K rows
  leaderboards/q3_local-cand_api-judge.parquet
  leaderboards/q4_api-cand_api-judge.parquet
  leaderboards/rrf_combined.csv
  metadata/cand_models.json
  metadata/judge_models.json
  metadata/pipeline.json
"""
import json, re, os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(".")
HF = Path("./hf-export")
LOCAL_DIR = ROOT / "results/phase5_exp_b_llm"
PROVIDER_DIR = ROOT / "results/phase5_exp_b_provider"
FLAGSHIP = ROOT / "results/phase5_judge_flagship"
RAW_ANTH = FLAGSHIP / "_raw_anthropic"

CUSTOM = re.compile(r'[^a-zA-Z0-9_-]')
QID_RE = re.compile(r'q\d{3}')
def safe_id(model, qid, metric): return CUSTOM.sub('_', f"{model}__{qid}__{metric}")[:64]
def norm_qid(raw, idx):
    if raw:
        m = QID_RE.search(str(raw))
        if m: return m.group()
    return f"q{idx:03d}"

print("=" * 60)
print("Phase A: qa, retrieval, cand_answers")
print("=" * 60)

# === 1. qa.parquet ===
print("\n1. qa.parquet")
gt = json.load(open(ROOT / "data/ground_truth_filtered.json"))
qa_rows = []
for i, item in enumerate(gt):
    qa_rows.append({
        'qid': f'q{i:03d}',
        'domain': item['domain'],
        'question': item['question'],
        'target_answer': item['target_answer'],
        'target_file_name': item['target_file_name'],
        'target_page_no': str(item.get('target_page_no', '')),
        'context_type': item['context_type'],
    })
df_qa = pd.DataFrame(qa_rows)
df_qa.to_parquet(HF / 'data/qa.parquet', index=False)
print(f"   saved {len(df_qa)} rows")

# === 2. retrieval.parquet ===
print("\n2. retrieval.parquet")
retr = json.load(open(ROOT / "data/retrieval_cache/gemma-embed-300m.json"))
retr_rows = []
for i, item in enumerate(retr):
    qid = f'q{i:03d}'
    files = item.get('retrieved_files', [])
    pages = item.get('retrieved_pages', [])
    context = item.get('context', '')
    retr_rows.append({
        'qid': qid,
        'embed_model': 'gemma-embed-300m',
        'top_k': 5,
        'retrieved_files': files,
        'retrieved_pages': pages,
        'context_concatenated': context,
    })
df_retr = pd.DataFrame(retr_rows)
df_retr.to_parquet(HF / 'data/retrieval.parquet', index=False)
print(f"   saved {len(df_retr)} rows")

# === 3. cand_answers.parquet ===
print("\n3. cand_answers.parquet")

def cand_meta_local(model):
    """로컬 cand 메타 추정."""
    family = 'unknown'; size = ''; quant = ''; runtime = 'local-llamacpp'
    if model.startswith('deepseek-r1'): family = 'deepseek-r1'; size = '70b'; quant = 'Q4_K_M'
    elif model.startswith('exaone3.5'): family = 'exaone3.5'; size = '32b'
    elif model.startswith('gpt-oss'):
        family = 'gpt-oss'
        size = '120b' if '120b' in model else '20b'
    elif model.startswith('lfm2'): family = 'lfm2'; size = '24b'
    elif model.startswith('mistral-small'): family = 'mistral-small'; size = '24b'
    elif model.startswith('phi4'): family = 'phi4'; size = '14b'
    elif model.startswith('qwen3.5_122b'):
        family = 'qwen3.5'; size = '122b'; quant = 'Q4_K_M'
    elif model.startswith('qwen3.5_27b'):
        family = 'qwen3.5'; size = '27b'; quant = 'Q8_0'
    elif model.startswith('qwen3.5_9b'):
        family = 'qwen3.5'; size = '9b'
        quant = 'Q8_0' if 'q8' in model else 'Q4_K_M'
    return family, size, quant, runtime

def cand_meta_api(model):
    """API cand 메타."""
    family = 'unknown'
    if 'claude' in model: family = 'anthropic'
    elif 'gpt' in model: family = 'openai'
    elif 'gemini' in model: family = 'google'
    elif 'deepseek' in model: family = 'deepseek'
    elif 'kimi' in model: family = 'moonshotai'
    elif 'qwen' in model: family = 'alibaba'
    elif 'minimax' in model: family = 'minimax'
    elif 'grok' in model: family = 'x-ai'
    elif 'mistral' in model: family = 'mistralai'
    elif 'mimo' in model: family = 'xiaomi'
    elif 'glm' in model: family = 'z-ai'
    elif 'nemotron' in model: family = 'nvidia'
    elif 'solar' in model: family = 'upstage'
    return family, '', '', 'api'

LOCAL_PREFIX = ("deepseek-r1","exaone3.5","gpt-oss","lfm2","mistral-small","phi4",
                "qwen3.5_122b","qwen3.5_27b","qwen3.5_9b")

cand_rows = []
# 12 local
for f in sorted(LOCAL_DIR.glob("expB__*.json")):
    model = f.stem.replace("expB__gemma-embed-300m__","")
    if not any(model.startswith(p) for p in LOCAL_PREFIX): continue
    d = json.load(open(f))
    family, size, quant, runtime = cand_meta_local(model)
    for idx, r in enumerate(d.get("results", [])):
        qid = f'q{idx:03d}'
        cand_rows.append({
            'qid': qid,
            'cand_id': model,
            'cand_family': family,
            'cand_size': size,
            'cand_quantization': quant,
            'cand_runtime': runtime,
            'generated_answer': r.get('generated_answer') or '',
            'input_tokens': r.get('input_tokens'),
            'output_tokens': r.get('output_tokens'),
            'latency_sec': r.get('latency_sec'),
        })

# 34 API
for f in sorted(PROVIDER_DIR.glob("expB__*.json")):
    if any(s in f.name for s in ['dryrun','damaged','before_rebuild']): continue
    model = f.stem.replace("expB__gemma-embed-300m__","")
    d = json.load(open(f))
    family, size, quant, runtime = cand_meta_api(model)
    for idx, r in enumerate(d.get("results", [])):
        qid = norm_qid(r.get("qid") or r.get("custom_id"), idx)
        ans = r.get('generated_answer') or r.get('answer') or ''
        cand_rows.append({
            'qid': qid,
            'cand_id': model,
            'cand_family': family,
            'cand_size': size,
            'cand_quantization': quant,
            'cand_runtime': runtime,
            'generated_answer': ans,
            'input_tokens': r.get('input_tokens'),
            'output_tokens': r.get('output_tokens'),
            'latency_sec': r.get('latency_sec'),
        })

# 19 .jsonl (API cands)
for f in sorted(PROVIDER_DIR.glob("expB__*.jsonl")):
    if any(s in f.name for s in ['dryrun','damaged','before_rebuild']): continue
    model = f.stem.replace("expB__gemma-embed-300m__","").replace("_","/",1)
    family, size, quant, runtime = cand_meta_api(model)
    seen = {}
    for idx, line in enumerate(open(f)):
        try: r = json.loads(line)
        except: continue
        qid = norm_qid(r.get('qid'), idx)
        ans = r.get('answer') or r.get('generated_answer') or ''
        if qid in seen and not ans: continue
        if qid in seen and seen[qid] and len(ans) <= len(seen[qid]): continue
        seen[qid] = ans
    for qid, ans in seen.items():
        cand_rows.append({
            'qid': qid,
            'cand_id': model,
            'cand_family': family,
            'cand_size': size,
            'cand_quantization': quant,
            'cand_runtime': runtime,
            'generated_answer': ans,
            'input_tokens': None,
            'output_tokens': None,
            'latency_sec': None,
        })

df_cand = pd.DataFrame(cand_rows)
# 중복 제거 (qid, cand_id) 기준 — 가장 긴 답변 유지
df_cand['ans_len'] = df_cand['generated_answer'].str.len().fillna(0)
df_cand = df_cand.sort_values('ans_len', ascending=False).drop_duplicates(['qid', 'cand_id']).drop(columns=['ans_len'])
df_cand = df_cand.sort_values(['cand_id', 'qid']).reset_index(drop=True)
df_cand.to_parquet(HF / 'data/cand_answers.parquet', index=False)
print(f"   saved {len(df_cand)} rows ({df_cand['cand_id'].nunique()} cands × ~300 q)")

print("\n" + "=" * 60)
print("Phase B: judge_scores, leaderboards")
print("=" * 60)

# === 4. judge_scores.parquet ===
print("\n4. judge_scores.parquet (Q3 + Q4)")

scores = defaultdict(dict)  # judge_id → {(cand_safe, qid, metric): score}

def parse_anth(d):
    cm = d.get('custom_id','') or ''
    if 'result' in d and 'message' in d.get('result', {}):
        ok = d['result'].get('type') == 'succeeded'; msg = d['result'].get('message', {})
    else:
        ok = d.get('type') == 'succeeded'; msg = d.get('message', {})
    if not ok: return cm, None, None
    text = ''
    for c in msg.get('content', []):
        if c.get('type') == 'text': text = c.get('text', ''); break
    return cm, msg.get('model','?'), text

def extract_score(text):
    if not text: return None
    m = re.findall(r'\b([1-5])\b', text)
    return int(m[-1]) if m else None

def parse_key(k):
    p = k.rsplit('__', 2)
    return (p[0], p[1], p[2]) if len(p)==3 else (None, None, None)

# Anthropic batch raw
for f in RAW_ANTH.glob('*.jsonl'):
    for line in open(f):
        try: d = json.loads(line)
        except: continue
        cm, model, text = parse_anth(d)
        s = extract_score(text)
        if s is None or not cm: continue
        cand, qid, metric = parse_key(cm)
        if not cand: continue
        if 'sonnet' in (model or '').lower():
            scores['claude-sonnet-4-6'][(cand, qid, metric)] = s
        elif 'opus' in (model or '').lower():
            scores['claude-opus-4-7'][(cand, qid, metric)] = s

def collect_jsonl_to_judge(p, judge_id):
    if not p.exists(): return
    for line in open(p):
        if not line.strip(): continue
        try: d = json.loads(line)
        except: continue
        if 'error' in d: continue
        k = d.get('custom_id') or d.get('key')
        if not k: continue
        sc = d.get('score')
        cand, qid, metric = parse_key(k)
        if not cand: continue
        if isinstance(sc, int) and 1 <= sc <= 5:
            scores[judge_id][(cand, qid, metric)] = sc; continue
        text = d.get('text') or ''
        if not text:
            ch = d.get('choices') or []
            if ch: text = (ch[0].get('message') or {}).get('content') or ''
        s = extract_score(text)
        if s: scores[judge_id][(cand, qid, metric)] = s

# Anthropic retries
for f in FLAGSHIP.glob('q3_anthropic_sonnet*.jsonl'): collect_jsonl_to_judge(f, 'claude-sonnet-4-6')
for f in FLAGSHIP.glob('q4_anthropic_sonnet*.jsonl'): collect_jsonl_to_judge(f, 'claude-sonnet-4-6')
collect_jsonl_to_judge(FLAGSHIP / 'q3_anthropic_opus46_fallback.jsonl', 'claude-opus-4-7')
collect_jsonl_to_judge(FLAGSHIP / 'q4_anthropic_opus46_fallback.jsonl', 'claude-opus-4-7')

# Supplemental (judge_label tagged)
for fn in ['q4_supplemental_empty_cand_retry.jsonl', 'q4_supplemental_flash_lite_fix.jsonl', 'q4_supplemental_gpt54pro_q181_q223.jsonl']:
    p = FLAGSHIP / fn
    if not p.exists(): continue
    for line in open(p):
        try: d = json.loads(line)
        except: continue
        if d.get('score') is None: continue
        sc = d['score']; k = d.get('key'); jl = d.get('judge_label', '')
        cand, qid, metric = parse_key(k)
        if not cand: continue
        # judge label 매핑
        jmap = {
            'anthropic/sonnet-4-6': 'claude-sonnet-4-6',
            'anthropic/opus-4-7': 'claude-opus-4-7',
            'google/gemini-pro': 'gemini-3.1-pro-preview',
            'google/gemini-flash': 'gemini-3-flash-preview',
            'google/gemini-flash-lite': 'gemini-3.1-flash-lite-preview',
            'openai/gpt-5.4': 'gpt-5.4',
            'openai/gpt-5.4-mini': 'gpt-5.4-mini',
            'openai/gpt-5.4-nano': 'gpt-5.4-nano',
        }
        judge_id = jmap.get(jl)
        if judge_id: scores[judge_id][(cand, qid, metric)] = sc

# Gemini judges
for f in FLAGSHIP.glob('q3_gemini_or_pro*.jsonl'): collect_jsonl_to_judge(f, 'gemini-3.1-pro-preview')
for f in FLAGSHIP.glob('q4_gemini_or_pro*.jsonl'): collect_jsonl_to_judge(f, 'gemini-3.1-pro-preview')
for f in FLAGSHIP.glob('q3_gemini_or_flash.jsonl'): collect_jsonl_to_judge(f, 'gemini-3-flash-preview')
for f in FLAGSHIP.glob('q3_gemini_or_flash_retry*.jsonl'): collect_jsonl_to_judge(f, 'gemini-3-flash-preview')
for f in FLAGSHIP.glob('q4_gemini_or_flash.jsonl'): collect_jsonl_to_judge(f, 'gemini-3-flash-preview')
for f in FLAGSHIP.glob('q4_gemini_or_flash_retry*.jsonl'): collect_jsonl_to_judge(f, 'gemini-3-flash-preview')
for f in FLAGSHIP.glob('q3_gemini_or_flash_lite*.jsonl'): collect_jsonl_to_judge(f, 'gemini-3.1-flash-lite-preview')
for f in FLAGSHIP.glob('q3_gemini_flash_lite_realtime*.jsonl'): collect_jsonl_to_judge(f, 'gemini-3.1-flash-lite-preview')
for f in FLAGSHIP.glob('q4_gemini_or_flash_lite*.jsonl'): collect_jsonl_to_judge(f, 'gemini-3.1-flash-lite-preview')

# OpenAI raw
def parse_oai(line):
    d = json.loads(line); cm = d.get('custom_id') or ''
    body = (d.get('response') or {}).get('body') or {}
    out = body.get('output') or []
    text = ''
    for blk in out:
        if blk.get('type') == 'message':
            for c in blk.get('content') or []:
                if c.get('type') == 'output_text': text = c.get('text',''); break
            if text: break
    if not text:
        ch = body.get('choices') or []
        if ch: text = (ch[0].get('message') or {}).get('content') or ''
    return cm, text

for fn, jid in [('_raw_openai_5.4_q3.jsonl','gpt-5.4'),('_raw_openai_5.4_q4.jsonl','gpt-5.4'),
                ('_raw_openai_5.4_mini_q3.jsonl','gpt-5.4-mini'),('_raw_openai_5.4_mini_q4.jsonl','gpt-5.4-mini'),
                ('_raw_openai_5.4_nano_q3.jsonl','gpt-5.4-nano'),('_raw_openai_5.4_nano_q4.jsonl','gpt-5.4-nano'),
                ('_raw_openai_q4_nano.jsonl','gpt-5.4-nano'),('_raw_openai_q3_main.jsonl','gpt-5.5')]:
    p = FLAGSHIP / fn
    if not p.exists(): continue
    for line in open(p):
        if not line.strip(): continue
        try:
            cm, text = parse_oai(line)
            s = extract_score(text)
            if s and cm:
                cand, qid, metric = parse_key(cm)
                if cand: scores[jid][(cand, qid, metric)] = s
        except: pass
for f in FLAGSHIP.glob('q3_openai_55*.jsonl'): collect_jsonl_to_judge(f, 'gpt-5.5')

# cand_safe → cand_id 매핑 (df_cand 사용)
cand_orig_set = set(df_cand['cand_id'].unique())
safe_to_orig = {CUSTOM.sub('_', c)[:64]: c for c in cand_orig_set}

# Long format judge_scores rows
print(f"   collecting from {len(scores)} judges")
judge_rows = []
for judge_id, kv in scores.items():
    for (cand_safe, qid, metric), sc in kv.items():
        cand_id = safe_to_orig.get(cand_safe, cand_safe)
        # Q3 (local cand) 또는 Q4 (api cand) 로 분류
        quadrant = 'Q3' if cand_id in [c for c in cand_orig_set
                                        if any(c.startswith(p) for p in LOCAL_PREFIX)] else 'Q4'
        judge_rows.append({
            'qid': qid,
            'cand_id': cand_id,
            'judge_id': judge_id,
            'metric': metric,
            'score': int(sc),
            'quadrant': quadrant,
        })
df_judge = pd.DataFrame(judge_rows)
df_judge = df_judge.drop_duplicates(['qid','cand_id','judge_id','metric']).reset_index(drop=True)
df_judge.to_parquet(HF / 'data/judge_scores.parquet', index=False)
print(f"   saved {len(df_judge)} rows ({df_judge['judge_id'].nunique()} judges × {df_judge['cand_id'].nunique()} cands)")

# === 5. leaderboards ===
print("\n5. leaderboards")

def acc_per_cell(df):
    """(cand_id, judge_id) cell accuracy = O / scored, where O = ≥2 of 4 metrics ≥ 4."""
    rows = []
    for (cand, judge), g in df.groupby(['cand_id','judge_id']):
        # qid → metric → score
        per_q = defaultdict(dict)
        for _, r in g.iterrows():
            per_q[r['qid']][r['metric']] = r['score']
        scored = 0; o = 0
        for qid, mdict in per_q.items():
            if len(mdict) < 4: continue
            scored += 1
            if sum(1 for v in mdict.values() if v >= 4) >= 2: o += 1
        if scored:
            rows.append({'cand_id': cand, 'judge_id': judge, 'accuracy': o/scored,
                        'o_count': o, 'scored': scored})
    return pd.DataFrame(rows)

# Q3 (local cand judges)
local_cand_set = set(df_cand[df_cand['cand_runtime'] == 'local-llamacpp']['cand_id'])
api_cand_set = set(df_cand[df_cand['cand_runtime'] == 'api']['cand_id'])

q3_df = df_judge[df_judge['cand_id'].isin(local_cand_set)]
q4_df = df_judge[df_judge['cand_id'].isin(api_cand_set)]

q3_lb = acc_per_cell(q3_df)
q4_lb = acc_per_cell(q4_df)

q3_lb.to_parquet(HF / 'leaderboards/q3_local-cand_api-judge.parquet', index=False)
q4_lb.to_parquet(HF / 'leaderboards/q4_api-cand_api-judge.parquet', index=False)
print(f"   Q3 leaderboard: {len(q3_lb)} cells")
print(f"   Q4 leaderboard: {len(q4_lb)} cells")

# RRF combined
def rrf_compute(lb_df, k=60):
    fused = defaultdict(float)
    for judge in lb_df['judge_id'].unique():
        sub = lb_df[lb_df['judge_id'] == judge].sort_values('accuracy', ascending=False)
        for r, (_, row) in enumerate(sub.iterrows(), 1):
            fused[row['cand_id']] += 1.0 / (k + r)
    return sorted(fused.items(), key=lambda x: -x[1])

q3_rrf = rrf_compute(q3_lb)
q4_rrf = rrf_compute(q4_lb)

rrf_rows = []
for r, (cand, score) in enumerate(q3_rrf, 1):
    rrf_rows.append({'quadrant': 'Q3', 'rank': r, 'cand_id': cand, 'rrf_score': score})
for r, (cand, score) in enumerate(q4_rrf, 1):
    rrf_rows.append({'quadrant': 'Q4', 'rank': r, 'cand_id': cand, 'rrf_score': score})
pd.DataFrame(rrf_rows).to_csv(HF / 'leaderboards/rrf_combined.csv', index=False)
print(f"   RRF combined: Q3 {len(q3_rrf)} + Q4 {len(q4_rrf)} cands")

# === 6. metadata ===
print("\n6. metadata")
cand_meta = []
for cand_id in sorted(df_cand['cand_id'].unique()):
    r = df_cand[df_cand['cand_id'] == cand_id].iloc[0]
    cand_meta.append({
        'cand_id': cand_id,
        'family': r['cand_family'],
        'size': r['cand_size'],
        'quantization': r['cand_quantization'],
        'runtime': r['cand_runtime'],
    })
json.dump(cand_meta, open(HF / 'metadata/cand_models.json', 'w'), ensure_ascii=False, indent=2)

judge_meta = [
    {'judge_id': 'claude-sonnet-4-6', 'family': 'anthropic', 'access': 'api',
     'note': 'main batch + 343 retry (max_tokens=1024) + 16 supplemental'},
    {'judge_id': 'claude-opus-4-7', 'family': 'anthropic', 'access': 'api',
     'note': 'main batch + 148 fallback to claude-opus-4-6 (Q4 only) for safety refusals on q142/q240/q258'},
    {'judge_id': 'gemini-3.1-pro-preview', 'family': 'google', 'access': 'openrouter',
     'note': 'reasoning effort=low'},
    {'judge_id': 'gemini-3-flash-preview', 'family': 'google', 'access': 'openrouter'},
    {'judge_id': 'gemini-3.1-flash-lite-preview', 'family': 'google', 'access': 'openrouter'},
    {'judge_id': 'gpt-5.4', 'family': 'openai', 'access': 'batch-api'},
    {'judge_id': 'gpt-5.4-mini', 'family': 'openai', 'access': 'batch-api'},
    {'judge_id': 'gpt-5.4-nano', 'family': 'openai', 'access': 'batch-api'},
    {'judge_id': 'gpt-5.5', 'family': 'openai', 'access': 'batch-api',
     'note': 'Q3 only (Responses API)'},
]
json.dump(judge_meta, open(HF / 'metadata/judge_models.json', 'w'), ensure_ascii=False, indent=2)

pipeline_meta = {
    'parser': 'pymupdf4llm',
    'chunking': {'chunk_size': 500, 'chunk_overlap': 100},
    'vectorstore': 'FAISS',
    'embedding': {'model': 'gemma-embed-300m', 'dim': 768, 'pooling': 'default'},
    'retrieval': {'top_k': 5, 'similarity': 'cosine'},
    'metric_protocol': {
        'metrics': ['similarity', 'correctness', 'completeness', 'faithfulness'],
        'scale': '1-5',
        'threshold': 4,
        'majority': '>=2 of 4 metrics >= threshold => O',
        'source': 'allganize methodology',
    },
    'rrf': {'formula': 'sum(1 / (k + rank_j))', 'k': 60},
    'source_dataset': {
        'name': 'allganize/RAG-Evaluation-Dataset-KO',
        'url': 'https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO',
        'license': 'MIT',
        'questions': 300,
        'pdfs': 58,
        'domains': ['finance', 'public', 'medical', 'law', 'commerce'],
    },
}
json.dump(pipeline_meta, open(HF / 'metadata/pipeline.json', 'w'), ensure_ascii=False, indent=2)
print(f"   metadata saved")

print("\n" + "=" * 60)
print("BUILD COMPLETE")
print("=" * 60)
print(f"\n출력 위치: {HF}")
print(f"   data/qa.parquet                      ({len(df_qa)} rows)")
print(f"   data/retrieval.parquet               ({len(df_retr)} rows)")
print(f"   data/cand_answers.parquet            ({len(df_cand)} rows)")
print(f"   data/judge_scores.parquet            ({len(df_judge)} rows)")
print(f"   leaderboards/q3_*.parquet            ({len(q3_lb)} cells)")
print(f"   leaderboards/q4_*.parquet            ({len(q4_lb)} cells)")
print(f"   leaderboards/rrf_combined.csv        ({len(rrf_rows)} rows)")
print(f"   metadata/cand_models.json            ({len(cand_meta)} models)")
print(f"   metadata/judge_models.json           ({len(judge_meta)} models)")
print(f"   metadata/pipeline.json               (pipeline spec)")
