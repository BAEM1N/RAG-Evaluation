# Phase 5 Exp B — LLM-as-Judge Leaderboard

**Retrieval**: gemma-embed-300m (FAISS, top-5)  
**Judges**: 6 (all 300 Q&A scored, allganize methodology: 4 metrics × threshold=4 × majority vote)  
**LLMs**: 1/12 complete

## Judges

- `gemma4:31b_nothink`
- `nemotron-120b_nothink`
- `qwen3.5-27b-claude-distill_nothink`
- `qwen3.5:122b-a10b-q4_K_M_nothink`
- `qwen3.6-35b-a3b_nothink`
- `supergemma4-26b_nothink`

## Cross-judge accuracy (O rate)

| Rank | LLM | gemma4:31b | nemotron-120b | qwen3.5-27b-claude-dis | qwen3.5:122b-a10b-q4_K | qwen3.6-35b-a3b | supergemma4-26b | **Avg** |
|---|---|---|---|---|---|---|---|---|
| 1 | `deepseek-r1_70b_nothink` | 0.6033 | 0.7167 | 0.3600 | 0.6200 | 0.7233 | 0.7233 | **0.6244** |
