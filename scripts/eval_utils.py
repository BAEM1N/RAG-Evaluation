"""
RAG 벤치마크 공통 유틸리티
- Ground Truth 로딩
- 검색 메트릭 계산
- 결과 저장/로딩
"""
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("rag-bench")

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── Ground Truth ────────────────────────────────────────────────────────

def load_ground_truth() -> List[Dict]:
    """ground_truth.json 로딩."""
    with open(DATA_DIR / "ground_truth.json", encoding="utf-8") as f:
        return json.load(f)


# ── 메트릭 계산 ────────────────────────────────────────────────────────

def compute_retrieval_metrics(
    results: List[Dict],
    ks: List[int] = [1, 3, 5, 10],
) -> Dict:
    """검색 결과에서 메트릭 계산.

    results: [{
        "question": str,
        "target_file": str,
        "target_page": str,
        "domain": str,
        "context_type": str,
        "retrieved": [{"file": str, "page": int, "rank": int, "score": float}, ...]
    }]
    """
    n = len(results)
    if n == 0:
        return {}

    # 전체 메트릭
    page_hits = {k: 0 for k in ks}
    file_hits = {k: 0 for k in ks}
    mrr_sum = 0.0

    for r in results:
        target_file = r["target_file"]
        target_page = str(r["target_page"])

        for ret in r["retrieved"]:
            rank = ret["rank"]
            # 페이지 매칭
            if ret["file"] == target_file and str(ret["page"]) == target_page:
                mrr_sum += 1.0 / rank
                for k in ks:
                    if rank <= k:
                        page_hits[k] += 1
                break

        for ret in r["retrieved"]:
            rank = ret["rank"]
            if ret["file"] == target_file:
                for k in ks:
                    if rank <= k:
                        file_hits[k] += 1
                break

    metrics = {
        "total": n,
        "mrr": round(mrr_sum / n, 4),
    }
    for k in ks:
        metrics[f"page_hit@{k}"] = round(page_hits[k] / n, 4)
        metrics[f"file_hit@{k}"] = round(file_hits[k] / n, 4)

    return metrics


def compute_metrics_by_group(
    results: List[Dict],
    group_key: str,  # "domain" or "context_type"
) -> Dict[str, Dict]:
    """그룹별 메트릭 계산."""
    groups = {}
    for r in results:
        g = r.get(group_key, "unknown")
        groups.setdefault(g, []).append(r)

    return {g: compute_retrieval_metrics(items) for g, items in groups.items()}


# ── 결과 저장 ──────────────────────────────────────────────────────────

def save_result(phase: str, name: str, data: dict):
    """결과 JSON 저장."""
    out_dir = RESULTS_DIR / phase
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"결과 저장: {path}")
    return path


def load_result(phase: str, name: str) -> dict:
    """결과 JSON 로딩."""
    path = RESULTS_DIR / phase / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 타이머 ──────────────────────────────────────────────────────────────

class Timer:
    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            logger.info(f"[{self.label}] {self.elapsed:.2f}s")


# ── 임베딩 호출 ────────────────────────────────────────────────────────

def get_embeddings_batch(
    texts: List[str],
    base_url: str = "http://192.168.50.245:8080/v1",
    model: str = "qwen3-embed-8b",
    batch_size: int = 8,
    max_chars: int = 500,
) -> List[List[float]]:
    """AI-395 게이트웨이를 통해 배치 임베딩 생성.

    llama.cpp 서버의 physical batch size(512 토큰) 제한 대응:
    - 텍스트를 max_chars로 트렁케이트
    - 배치 실패 시 개별 재시도
    - 네트워크 단절 시 재연결 대기
    """
    import httpx

    truncated = [t[:max_chars] for t in texts]
    all_embeddings = [None] * len(texts)
    client = httpx.Client(timeout=120)

    try:
        for i in range(0, len(truncated), batch_size):
            batch = truncated[i:i + batch_size]
            batch_indices = list(range(i, i + len(batch)))

            success = False
            for retry in range(5):
                try:
                    resp = client.post(
                        f"{base_url}/embeddings",
                        json={"model": model, "input": batch},
                    )
                    data = resp.json()

                    if "error" in data:
                        # 서버 에러 (토큰 초과 등) → 개별 재시도
                        for idx, text in zip(batch_indices, batch):
                            emb = _embed_single_with_fallback(client, base_url, model, text)
                            all_embeddings[idx] = emb
                    else:
                        embs = data.get("data", [])
                        for e in sorted(embs, key=lambda x: x["index"]):
                            all_embeddings[batch_indices[e["index"]]] = e["embedding"]
                    success = True
                    break
                except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
                    wait = 10 * (retry + 1)
                    logger.warning(f"네트워크 오류 (offset={i}, retry={retry+1}/5): {e}, {wait}초 대기")
                    time.sleep(wait)
                    # 클라이언트 재생성
                    try:
                        client.close()
                    except Exception:
                        pass
                    client = httpx.Client(timeout=120)
                except Exception as e:
                    logger.warning(f"배치 임베딩 실패 (offset={i}): {e}, 개별 재시도")
                    for idx, text in zip(batch_indices, batch):
                        emb = _embed_single_with_fallback(client, base_url, model, text)
                        all_embeddings[idx] = emb
                    success = True
                    break

            if not success:
                logger.error(f"배치 {i} 영구 실패, 제로 벡터로 대체")
    finally:
        client.close()

    # None인 항목은 제로 벡터로 대체
    dim = next((len(e) for e in all_embeddings if e is not None), 0)
    return [e if e is not None else [0.0] * dim for e in all_embeddings]


def _embed_single_with_fallback(client, base_url, model, text, retries=3):
    """단일 텍스트 임베딩. 실패 시 점점 짧게 트렁케이트."""
    for attempt in range(retries):
        trunc = text[:max(50, len(text) // (attempt + 1))]
        try:
            resp = client.post(
                f"{base_url}/embeddings",
                json={"model": model, "input": [trunc]},
            )
            data = resp.json()
            if "error" not in data and data.get("data"):
                return data["data"][0]["embedding"]
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            time.sleep(5)
        except Exception:
            pass
    return None


def get_embedding_single(
    text: str,
    base_url: str = "http://192.168.50.245:8080/v1",
    model: str = "qwen3-embed-8b",
) -> List[float]:
    """단일 텍스트 임베딩."""
    result = get_embeddings_batch([text], base_url, model, batch_size=1)
    return result[0] if result else []


def get_query_embeddings_batch(
    queries: List[str],
    base_url: str = "http://192.168.50.245:8080/v1",
    model: str = "qwen3-embed-8b",
) -> List[List[float]]:
    """쿼리 목록을 한번에 임베딩."""
    return get_embeddings_batch(queries, base_url, model, batch_size=32)


# ── PDF 파싱 ────────────────────────────────────────────────────────────

def parse_pdf_pypdf(pdf_path: str) -> List[Dict]:
    """PyPDF로 페이지별 텍스트 추출."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    return pages


def parse_pdf_pymupdf4llm(pdf_path: str) -> List[Dict]:
    """pymupdf4llm으로 마크다운 추출."""
    import pymupdf4llm
    import pymupdf
    doc = pymupdf.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        md = pymupdf4llm.to_markdown(doc, pages=[i])
        text = md.strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def parse_pdf_pymupdf(pdf_path: str) -> List[Dict]:
    """기본 PyMuPDF 텍스트 추출."""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


# ── 청킹 ────────────────────────────────────────────────────────────────

def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 1000,
    overlap: int = 200,
    min_len: int = 50,
) -> List[Dict]:
    """페이지별 텍스트를 청크로 분할."""
    import re
    chunks = []
    for page_info in pages:
        page_num = page_info["page"]
        text = page_info["text"]

        if len(text) <= chunk_size:
            if len(text) >= min_len:
                chunks.append({
                    "page": page_num,
                    "chunk_index": len(chunks),
                    "text": text,
                })
            continue

        # 문단 기준 분할
        paragraphs = re.split(r"\n\s*\n|(?=\n#{1,3}\s+)", text)
        current = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) + 1 <= chunk_size:
                current = f"{current}\n\n{para}" if current else para
            else:
                if current and len(current) >= min_len:
                    chunks.append({
                        "page": page_num,
                        "chunk_index": len(chunks),
                        "text": current.strip(),
                    })
                current = para

        if current and len(current) >= min_len:
            chunks.append({
                "page": page_num,
                "chunk_index": len(chunks),
                "text": current.strip(),
            })

    return chunks


# ── 출력 ────────────────────────────────────────────────────────────────

def print_metrics_table(results: Dict[str, Dict], title: str = ""):
    """여러 실험의 메트릭을 테이블로 출력."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    header = "%-25s %8s %8s %8s %8s %8s" % ("Name", "Hit@1", "Hit@5", "MRR", "F-Hit@1", "F-Hit@5")
    print(header)
    print("-" * len(header))

    for name, m in sorted(results.items(), key=lambda x: x[1].get("mrr", 0), reverse=True):
        print("%-25s %8.1f%% %7.1f%% %8.4f %7.1f%% %7.1f%%" % (
            name[:25],
            m.get("page_hit@1", 0) * 100,
            m.get("page_hit@5", 0) * 100,
            m.get("mrr", 0),
            m.get("file_hit@1", 0) * 100,
            m.get("file_hit@5", 0) * 100,
        ))
