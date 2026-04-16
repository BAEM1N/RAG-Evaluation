#!/usr/bin/env python3
"""
RAG 벤치마크 통합 실행 스크립트

Usage:
  python bench_all.py --phase 1              # Parser 비교
  python bench_all.py --phase 2              # Chunking 비교
  python bench_all.py --phase 3              # VectorStore 비교
  python bench_all.py --phase 4              # Embedding 비교
  python bench_all.py --phase 4 --model snowflake-arctic-ko  # 단일 모델
  python bench_all.py --summary              # 전체 결과 요약
"""
import argparse
import json
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)

PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"

# ── 설정 ────────────────────────────────────────────────────────────────

GATEWAY_URL = "http://192.168.50.245:8080/v1"
T7910 = "192.168.50.250"

PARSERS = {
    "pypdf": parse_pdf_pypdf,
    "pymupdf4llm": parse_pdf_pymupdf4llm,
    "pymupdf": parse_pdf_pymupdf,
}

CHUNK_STRATEGIES = [
    {"name": "small",    "chunk_size": 500,  "overlap": 100},
    {"name": "baseline", "chunk_size": 1000, "overlap": 200},
    {"name": "medium",   "chunk_size": 1500, "overlap": 200},
    {"name": "large",    "chunk_size": 2000, "overlap": 300},
]

EMBEDDING_MODELS = {
    # 대형
    "qwen3-embed-8b":           {"vram": 9.0, "dim": 4096},
    "llama-embed-nemotron-8b":  {"vram": 9.0, "dim": 4096},
    "nemotron-embed-8b":        {"vram": 9.0, "dim": 4096},
    "e5-mistral-7b":            {"vram": 8.0, "dim": 4096},
    # 중형
    "qwen3-embed-4b":           {"vram": 5.0, "dim": 4096},
    "jina-v4-retrieval":        {"vram": 4.0, "dim": 4096},
    "jina-v4-code":             {"vram": 4.0, "dim": 4096},
    "jina-code-1.5b":           {"vram": 2.0, "dim": 1024},
    # 한국어 특화
    "snowflake-arctic-ko":      {"vram": 1.0, "dim": 1024},
    "pixie-rune-v1":            {"vram": 1.0, "dim": 1024},
    "kure-v1":                  {"vram": 1.0, "dim": 1024},
    "koe5":                     {"vram": 1.0, "dim": 1024},
    # 다국어 소형
    "qwen3-embed-0.6b":         {"vram": 1.0, "dim": 1024},
    "snowflake-arctic-l-v2":    {"vram": 1.0, "dim": 1024},
    "bge-m3":                   {"vram": 1.0, "dim": 1024},
    "me5-large-instruct":       {"vram": 1.0, "dim": 1024},
    "jina-v5-small-retrieval":  {"vram": 1.0, "dim": 1024},
    "harrier-0.6b":             {"vram": 0.8, "dim": 1024},
    "labse":                    {"vram": 1.0, "dim": 768},
    "nomic-embed-v2-moe":       {"vram": 1.0, "dim": 768},
    # 초소형
    "mxbai-embed-large":        {"vram": 0.5, "dim": 1024},
    "voyage-4-nano":            {"vram": 0.5, "dim": 1024},
    "gemma-embed-300m":         {"vram": 0.5, "dim": 768},
    "granite-278m":             {"vram": 0.5, "dim": 768},
    "harrier-270m":             {"vram": 0.3, "dim": 1024},
    "jina-v5-nano-matching":    {"vram": 0.5, "dim": 512},
    "granite-107m":             {"vram": 0.2, "dim": 768},
    # 초대형
    "harrier-27b":              {"vram": 20.0, "dim": 4096},
}

VECTORSTORES = ["pgvector", "faiss", "chroma", "milvus", "qdrant", "weaviate", "lancedb"]


# ── Phase 1: Parser ─────────────────────────────────────────────────────

def run_phase1():
    """Parser 비교."""
    gt = load_ground_truth()
    all_metrics = {}

    for parser_name, parser_fn in PARSERS.items():
        logger.info(f"=== Phase 1: Parser={parser_name} ===")

        # 모든 PDF 파싱
        all_chunks = []
        file_map = {}  # chunk_index -> (file_name, page)
        for gt_item in gt:
            fname = gt_item["target_file_name"]
            if fname in file_map:
                continue

            pdf_path = find_pdf(fname)
            if not pdf_path:
                logger.warning(f"PDF 없음: {fname}")
                continue

            pages = parser_fn(str(pdf_path))
            chunks = chunk_pages(pages, chunk_size=1000, overlap=200)
            for c in chunks:
                c["file"] = fname
            all_chunks.extend(chunks)
            file_map[fname] = True

        logger.info(f"  청크: {len(all_chunks)}개 / 파일: {len(file_map)}개")

        # 벡터화
        texts = [c["text"] for c in all_chunks]
        with Timer("embed"):
            embeddings = get_embeddings_batch(texts, GATEWAY_URL, "qwen3-embed-8b")

        # 검색 평가
        results = evaluate_retrieval_inmemory(gt, all_chunks, embeddings)
        metrics = compute_retrieval_metrics(results)
        by_domain = compute_metrics_by_group(results, "domain")
        by_ctype = compute_metrics_by_group(results, "context_type")

        all_metrics[parser_name] = metrics

        save_result("phase1_parser", parser_name, {
            "parser": parser_name,
            "total_chunks": len(all_chunks),
            "metrics": metrics,
            "by_domain": by_domain,
            "by_context_type": by_ctype,
        })

    print_metrics_table(all_metrics, "Phase 1: Parser 비교")
    return all_metrics


# ── Phase 2: Chunking ───────────────────────────────────────────────────

def run_phase2(best_parser: str = "pymupdf4llm"):
    """Chunking 전략 비교."""
    gt = load_ground_truth()
    parser_fn = PARSERS[best_parser]
    all_metrics = {}

    # PDF 파싱 (1회)
    parsed_cache = {}
    for gt_item in gt:
        fname = gt_item["target_file_name"]
        if fname not in parsed_cache:
            pdf_path = find_pdf(fname)
            if pdf_path:
                parsed_cache[fname] = parser_fn(str(pdf_path))

    for strategy in CHUNK_STRATEGIES:
        name = strategy["name"]
        logger.info(f"=== Phase 2: Chunk={name} ({strategy}) ===")

        all_chunks = []
        for fname, pages in parsed_cache.items():
            chunks = chunk_pages(pages, strategy["chunk_size"], strategy["overlap"])
            for c in chunks:
                c["file"] = fname
            all_chunks.extend(chunks)

        logger.info(f"  청크: {len(all_chunks)}개")

        texts = [c["text"] for c in all_chunks]
        with Timer("embed"):
            embeddings = get_embeddings_batch(texts, GATEWAY_URL, "qwen3-embed-8b")

        results = evaluate_retrieval_inmemory(gt, all_chunks, embeddings)
        metrics = compute_retrieval_metrics(results)
        all_metrics[name] = metrics

        save_result("phase2_chunking", name, {
            "strategy": strategy,
            "total_chunks": len(all_chunks),
            "metrics": metrics,
            "by_domain": compute_metrics_by_group(results, "domain"),
            "by_context_type": compute_metrics_by_group(results, "context_type"),
        })

    print_metrics_table(all_metrics, "Phase 2: Chunking 비교")
    return all_metrics


# ── Phase 3: VectorStore ────────────────────────────────────────────────

def run_phase3(
    best_parser: str = "pymupdf4llm",
    best_chunk: dict = None,
    embed_model: str = "qwen3-embed-8b",
):
    """VectorStore 7종 비교 (T7910)."""
    if best_chunk is None:
        best_chunk = {"chunk_size": 1000, "overlap": 200}

    gt = load_ground_truth()
    parser_fn = PARSERS[best_parser]

    # 청크 준비
    all_chunks = []
    parsed_cache = {}
    for gt_item in gt:
        fname = gt_item["target_file_name"]
        if fname not in parsed_cache:
            pdf_path = find_pdf(fname)
            if pdf_path:
                parsed_cache[fname] = parser_fn(str(pdf_path))

    for fname, pages in parsed_cache.items():
        chunks = chunk_pages(pages, best_chunk["chunk_size"], best_chunk["overlap"])
        for c in chunks:
            c["file"] = fname
        all_chunks.extend(chunks)

    logger.info(f"청크 준비 완료: {len(all_chunks)}개")

    # 임베딩 1회 생성 (모든 스토어에 동일 벡터 사용)
    texts = [c["text"] for c in all_chunks]
    with Timer("embed_chunks"):
        chunk_embeddings = get_embeddings_batch(texts, GATEWAY_URL, embed_model)

    questions = [item["question"] for item in gt]
    with Timer("embed_queries"):
        query_embeddings = get_query_embeddings_batch(questions, GATEWAY_URL, embed_model)

    dim = len(chunk_embeddings[0])
    logger.info(f"임베딩 완료: dim={dim}")

    all_metrics = {}

    for vs_name in VECTORSTORES:
        logger.info(f"=== Phase 3: VectorStore={vs_name} ===")

        try:
            result = _bench_vectorstore(
                vs_name, all_chunks, chunk_embeddings,
                gt, query_embeddings, dim,
            )
            all_metrics[vs_name] = result["metrics"]

            save_result("phase3_vectorstore", vs_name, result)
            logger.info(f"  MRR={result['metrics']['mrr']:.4f} "
                        f"insert={result.get('insert_time_sec', 0):.1f}s "
                        f"avg_latency={result.get('query_latency_avg_ms', 0):.1f}ms")

        except Exception as e:
            logger.error(f"  실패: {e}", exc_info=True)
            all_metrics[vs_name] = {"error": str(e)}

    print_metrics_table(all_metrics, "Phase 3: VectorStore 비교")
    return all_metrics


def _bench_vectorstore(
    vs_name: str,
    chunks: List[Dict],
    chunk_embeddings: List[List[float]],
    gt: List[Dict],
    query_embeddings: List[List[float]],
    dim: int,
) -> Dict:
    """개별 벡터스토어 벤치마크."""
    import numpy as np
    import time as _time

    # ── INSERT ──
    t0 = _time.perf_counter()

    if vs_name == "faiss":
        index, id_map = _insert_faiss(chunks, chunk_embeddings, dim)
    elif vs_name == "lancedb":
        table = _insert_lancedb(chunks, chunk_embeddings, dim)
    elif vs_name == "pgvector":
        conn = _insert_pgvector(chunks, chunk_embeddings, dim)
    elif vs_name == "chroma":
        collection = _insert_chroma(chunks, chunk_embeddings)
    elif vs_name == "milvus":
        collection_name = _insert_milvus(chunks, chunk_embeddings, dim)
    elif vs_name == "qdrant":
        _insert_qdrant(chunks, chunk_embeddings, dim)
    elif vs_name == "weaviate":
        wv_client = _insert_weaviate(chunks, chunk_embeddings, dim)

    insert_time = _time.perf_counter() - t0

    # ── SEARCH ──
    latencies = []
    results = []

    for gt_item, q_emb in zip(gt, query_embeddings):
        if not q_emb:
            continue

        t1 = _time.perf_counter()

        if vs_name == "faiss":
            retrieved = _search_faiss(index, id_map, chunks, q_emb, 10)
        elif vs_name == "lancedb":
            retrieved = _search_lancedb(table, chunks, q_emb, 10)
        elif vs_name == "pgvector":
            retrieved = _search_pgvector(conn, q_emb, 10)
        elif vs_name == "chroma":
            retrieved = _search_chroma(collection, q_emb, 10)
        elif vs_name == "milvus":
            retrieved = _search_milvus(collection_name, q_emb, 10)
        elif vs_name == "qdrant":
            retrieved = _search_qdrant(q_emb, 10)
        elif vs_name == "weaviate":
            retrieved = _search_weaviate(wv_client, q_emb, 10)

        latency_ms = (_time.perf_counter() - t1) * 1000
        latencies.append(latency_ms)

        results.append({
            "question": gt_item["question"],
            "target_file": gt_item["target_file_name"],
            "target_page": gt_item["target_page_no"],
            "domain": gt_item["domain"],
            "context_type": gt_item["context_type"],
            "retrieved": retrieved,
        })

    # ── CLEANUP ──
    try:
        if vs_name == "pgvector":
            conn.close()
        elif vs_name == "milvus":
            from pymilvus import connections
            connections.disconnect("default")
        elif vs_name == "weaviate":
            wv_client.close()
    except Exception:
        pass

    metrics = compute_retrieval_metrics(results)

    return {
        "vectorstore": vs_name,
        "total_chunks": len(chunks),
        "dim": dim,
        "insert_time_sec": round(insert_time, 2),
        "query_latency_avg_ms": round(np.mean(latencies), 2) if latencies else 0,
        "query_latency_p95_ms": round(np.percentile(latencies, 95), 2) if latencies else 0,
        "queries_per_sec": round(len(latencies) / (sum(latencies) / 1000), 1) if latencies else 0,
        "metrics": metrics,
        "by_domain": compute_metrics_by_group(results, "domain"),
        "by_context_type": compute_metrics_by_group(results, "context_type"),
    }


# ── VectorStore 어댑터: FAISS ──────────────────────────────────────────

def _insert_faiss(chunks, embeddings, dim):
    import faiss
    import numpy as np

    emb_array = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(emb_array)

    index = faiss.IndexFlatIP(dim)
    index.add(emb_array)
    id_map = list(range(len(chunks)))
    return index, id_map


def _search_faiss(index, id_map, chunks, query_emb, top_k):
    import faiss
    import numpy as np

    q = np.array([query_emb], dtype=np.float32)
    faiss.normalize_L2(q)
    scores, indices = index.search(q, top_k)

    retrieved = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        if idx < 0:
            continue
        retrieved.append({
            "file": chunks[idx]["file"],
            "page": chunks[idx]["page"],
            "rank": rank,
            "score": float(score),
        })
    return retrieved


# ── VectorStore 어댑터: LanceDB ────────────────────────────────────────

def _insert_lancedb(chunks, embeddings, dim):
    import lancedb

    db = lancedb.connect("/tmp/rag_bench_lancedb")
    data = []
    for i, (c, emb) in enumerate(zip(chunks, embeddings)):
        data.append({
            "id": i,
            "file": c["file"],
            "page": c["page"],
            "text": c["text"][:500],
            "vector": emb,
        })

    try:
        db.drop_table("bench")
    except Exception:
        pass

    table = db.create_table("bench", data)
    return table


def _search_lancedb(table, chunks, query_emb, top_k):
    results = table.search(query_emb).metric("cosine").limit(top_k).to_list()
    retrieved = []
    for rank, r in enumerate(results, 1):
        retrieved.append({
            "file": r["file"],
            "page": r["page"],
            "rank": rank,
            "score": float(1 - r.get("_distance", 0)),
        })
    return retrieved


# ── VectorStore 어댑터: pgvector ───────────────────────────────────────

def _insert_pgvector(chunks, embeddings, dim):
    import psycopg2
    from psycopg2.extras import execute_values

    conn = psycopg2.connect(
        host=T7910, port=5433,
        user="bench", password="bench", dbname="ragbench",
    )
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("DROP TABLE IF EXISTS bench_chunks")
    cur.execute(f"""
        CREATE TABLE bench_chunks (
            id SERIAL PRIMARY KEY,
            file TEXT,
            page INT,
            text TEXT,
            embedding vector({dim})
        )
    """)
    conn.commit()

    data = []
    for c, emb in zip(chunks, embeddings):
        vec_str = "[" + ",".join(str(v) for v in emb) + "]"
        data.append((c["file"], c["page"], c["text"][:500], vec_str))

    execute_values(
        cur,
        "INSERT INTO bench_chunks (file, page, text, embedding) VALUES %s",
        data, template="(%s, %s, %s, %s)",
        page_size=500,
    )
    conn.commit()

    # HNSW 인덱스
    if dim <= 2000:
        cur.execute(f"""
            CREATE INDEX ON bench_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200)
        """)
    # dim > 2000: pgvector HNSW/IVFFlat 모두 2000차원 제한 → 인덱스 없이 순차검색
    conn.commit()
    cur.close()
    return conn


def _search_pgvector(conn, query_emb, top_k):
    cur = conn.cursor()
    vec_str = "[" + ",".join(str(v) for v in query_emb) + "]"
    cur.execute(f"""
        SELECT file, page, 1 - (embedding <=> %s::vector) AS score
        FROM bench_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (vec_str, vec_str, top_k))

    retrieved = []
    for rank, (file, page, score) in enumerate(cur.fetchall(), 1):
        retrieved.append({"file": file, "page": page, "rank": rank, "score": float(score)})
    cur.close()
    return retrieved


# ── VectorStore 어댑터: Chroma ─────────────────────────────────────────

def _insert_chroma(chunks, embeddings):
    import chromadb

    client = chromadb.HttpClient(host=T7910, port=8100)
    try:
        client.delete_collection("bench")
    except Exception:
        pass

    collection = client.create_collection("bench", metadata={"hnsw:space": "cosine"})

    # Chroma는 배치 제한이 있으므로 분할
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embs = embeddings[i:i + batch_size]
        collection.add(
            ids=[str(i + j) for j in range(len(batch_chunks))],
            embeddings=batch_embs,
            metadatas=[{"file": c["file"], "page": c["page"]} for c in batch_chunks],
        )
    return collection


def _search_chroma(collection, query_emb, top_k):
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    retrieved = []
    if results["metadatas"] and results["metadatas"][0]:
        for rank, (meta, dist) in enumerate(
            zip(results["metadatas"][0], results["distances"][0]), 1
        ):
            retrieved.append({
                "file": meta["file"],
                "page": meta["page"],
                "rank": rank,
                "score": float(1 - dist),  # cosine distance → similarity
            })
    return retrieved


# ── VectorStore 어댑터: Milvus ─────────────────────────────────────────

def _insert_milvus(chunks, embeddings, dim):
    from pymilvus import (
        connections, Collection, CollectionSchema,
        FieldSchema, DataType, utility,
    )

    connections.connect("default", host=T7910, port="19530")

    collection_name = "bench_chunks"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("file", DataType.VARCHAR, max_length=512),
        FieldSchema("page", DataType.INT64),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dim),
    ])
    col = Collection(collection_name, schema)

    # 배치 삽입
    batch_size = 1000
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embs = embeddings[i:i + batch_size]
        col.insert([
            [c["file"] for c in batch],
            [c["page"] for c in batch],
            batch_embs,
        ])

    col.flush()
    col.create_index("embedding", {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    })
    col.load()
    return collection_name


def _search_milvus(collection_name, query_emb, top_k):
    from pymilvus import Collection

    col = Collection(collection_name)
    results = col.search(
        data=[query_emb],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["file", "page"],
    )

    retrieved = []
    for rank, hit in enumerate(results[0], 1):
        retrieved.append({
            "file": hit.entity.get("file"),
            "page": hit.entity.get("page"),
            "rank": rank,
            "score": float(hit.score),
        })
    return retrieved


# ── VectorStore 어댑터: Qdrant ─────────────────────────────────────────

def _insert_qdrant(chunks, embeddings, dim):
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct

    client = QdrantClient(host=T7910, port=6340, timeout=120)
    collection_name = "bench_chunks"

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    client.create_collection(
        collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    batch_size = 100  # 4096dim에서는 작은 배치 필요
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embs = embeddings[i:i + batch_size]
        points = [
            PointStruct(
                id=i + j,
                vector=emb,
                payload={"file": str(c["file"]), "page": int(c["page"])},
            )
            for j, (c, emb) in enumerate(zip(batch, batch_embs))
        ]
        client.upsert(collection_name, points)


def _search_qdrant(query_emb, top_k):
    from qdrant_client import QdrantClient

    client = QdrantClient(host=T7910, port=6340, timeout=120)
    results = client.query_points(
        collection_name="bench_chunks",
        query=query_emb,
        limit=top_k,
    )

    retrieved = []
    for rank, point in enumerate(results.points, 1):
        retrieved.append({
            "file": point.payload["file"],
            "page": point.payload["page"],
            "rank": rank,
            "score": float(point.score),
        })
    return retrieved


# ── VectorStore 어댑터: Weaviate ───────────────────────────────────────

def _insert_weaviate(chunks, embeddings, dim):
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType

    client = weaviate.connect_to_custom(
        http_host=T7910, http_port=8101, http_secure=False,
        grpc_host=T7910, grpc_port=50052, grpc_secure=False,
    )

    try:
        client.collections.delete("BenchChunks")
    except Exception:
        pass

    collection = client.collections.create(
        name="BenchChunks",
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="file_name", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
        ],
    )

    batch_size = 500
    with collection.batch.fixed_size(batch_size=batch_size) as batch:
        for c, emb in zip(chunks, embeddings):
            batch.add_object(
                properties={"file_name": c["file"], "page": c["page"]},
                vector=emb,
            )

    return client


def _search_weaviate(client, query_emb, top_k):
    collection = client.collections.get("BenchChunks")
    results = collection.query.near_vector(
        near_vector=query_emb,
        limit=top_k,
        return_metadata=["distance"],
    )

    retrieved = []
    for rank, obj in enumerate(results.objects, 1):
        retrieved.append({
            "file": obj.properties["file_name"],
            "page": obj.properties["page"],
            "rank": rank,
            "score": float(1 - (obj.metadata.distance or 0)),
        })
    return retrieved


# ── Phase 4: Embedding ──────────────────────────────────────────────────

def run_phase4(
    best_parser: str = "pymupdf4llm",
    best_chunk: dict = None,
    target_model: str = None,
):
    """Embedding 모델 비교."""
    if best_chunk is None:
        best_chunk = {"chunk_size": 1000, "overlap": 200}

    gt = load_ground_truth()
    parser_fn = PARSERS[best_parser]

    # 청크 준비 (1회)
    all_chunks = []
    parsed_cache = {}
    for gt_item in gt:
        fname = gt_item["target_file_name"]
        if fname not in parsed_cache:
            pdf_path = find_pdf(fname)
            if pdf_path:
                parsed_cache[fname] = parser_fn(str(pdf_path))

    for fname, pages in parsed_cache.items():
        chunks = chunk_pages(pages, best_chunk["chunk_size"], best_chunk["overlap"])
        for c in chunks:
            c["file"] = fname
        all_chunks.extend(chunks)

    texts = [c["text"] for c in all_chunks]
    logger.info(f"청크 준비 완료: {len(all_chunks)}개")

    # 모델별 테스트
    models = {target_model: EMBEDDING_MODELS[target_model]} if target_model else EMBEDDING_MODELS
    all_metrics = {}

    for model_name, model_info in models.items():
        logger.info(f"=== Phase 4: Embedding={model_name} (dim={model_info['dim']}, vram={model_info['vram']}GB) ===")

        try:
            # 게이트웨이에서 모델 로드
            swap_embedding_model(model_name)

            with Timer("embed") as t:
                embeddings = get_embeddings_batch(texts, GATEWAY_URL, model_name)

            embed_speed = len(texts) / t.elapsed if t.elapsed > 0 else 0

            results = evaluate_retrieval_inmemory(gt, all_chunks, embeddings, embed_model=model_name)
            metrics = compute_retrieval_metrics(results)
            metrics["embed_speed_chunks_per_sec"] = round(embed_speed, 1)
            metrics["dim"] = model_info["dim"]
            metrics["vram_gb"] = model_info["vram"]

            all_metrics[model_name] = metrics

            save_result("phase4_embedding", model_name, {
                "model": model_name,
                "dim": model_info["dim"],
                "vram_gb": model_info["vram"],
                "total_chunks": len(all_chunks),
                "embed_time_sec": round(t.elapsed, 2),
                "embed_speed": round(embed_speed, 1),
                "metrics": metrics,
                "by_domain": compute_metrics_by_group(results, "domain"),
                "by_context_type": compute_metrics_by_group(results, "context_type"),
            })

            logger.info(f"  MRR={metrics['mrr']:.4f} Hit@5={metrics.get('page_hit@5',0):.1%} ({embed_speed:.0f} chunks/sec)")

        except Exception as e:
            logger.error(f"  실패: {e}")
            all_metrics[model_name] = {"error": str(e)}

    print_metrics_table(all_metrics, "Phase 4: Embedding 비교")
    return all_metrics


# ── 헬퍼 함수 ──────────────────────────────────────────────────────────

def find_pdf(filename: str) -> Optional[Path]:
    """PDF 파일 경로 찾기."""
    for domain_dir in PDF_DIR.iterdir():
        if domain_dir.is_dir():
            pdf = domain_dir / filename
            if pdf.exists():
                return pdf
    # 플랫 구조도 확인
    pdf = PDF_DIR / filename
    if pdf.exists():
        return pdf
    return None


def evaluate_retrieval_inmemory(
    gt: List[Dict],
    chunks: List[Dict],
    embeddings: List[List[float]],
    top_k: int = 10,
    embed_model: str = "qwen3-embed-8b",
) -> List[Dict]:
    """인메모리 벡터 검색으로 평가."""
    import numpy as np

    emb_array = np.array(embeddings, dtype=np.float32)
    # L2 normalize for cosine
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_array = emb_array / norms

    # 쿼리 임베딩을 배치로 한번에 가져오기
    questions = [item["question"] for item in gt]
    query_embeddings = get_query_embeddings_batch(questions, GATEWAY_URL, embed_model)

    results = []
    for gt_item, q_emb in zip(gt, query_embeddings):
        if not q_emb:
            continue

        q_vec = np.array(q_emb, dtype=np.float32)
        q_vec = q_vec / (np.linalg.norm(q_vec) or 1)

        # cosine similarity
        scores = emb_array @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]

        retrieved = []
        for rank, idx in enumerate(top_idx, 1):
            retrieved.append({
                "file": chunks[idx]["file"],
                "page": chunks[idx]["page"],
                "rank": rank,
                "score": float(scores[idx]),
            })

        results.append({
            "question": gt_item["question"],
            "target_file": gt_item["target_file_name"],
            "target_page": gt_item["target_page_no"],
            "domain": gt_item["domain"],
            "context_type": gt_item["context_type"],
            "retrieved": retrieved,
        })

    return results


def swap_embedding_model(model_name: str):
    """게이트웨이에서 임베딩 모델 교체."""
    import httpx

    with httpx.Client(timeout=120) as client:
        # 현재 실행 중인 임베딩 모델 확인
        resp = client.get(f"http://192.168.50.245:8080/v1/models")
        data = resp.json()
        running_embeds = [m["alias"] for m in data["running"] if m["mode"] == "embedding"]

        # 이미 로드되어 있으면 스킵
        if model_name in running_embeds:
            return

        # 기존 임베딩 언로드
        for alias in running_embeds:
            client.post("http://192.168.50.245:8080/v1/models/unload", json={"model": alias})
            logger.info(f"  언로드: {alias}")

        # 새 모델 로드
        resp = client.post("http://192.168.50.245:8080/v1/models/load", json={"model": model_name})
        resp.raise_for_status()
        logger.info(f"  로드: {model_name} → port {resp.json().get('port')}")

        # 헬스체크 대기
        import time
        for _ in range(30):
            try:
                r = client.get(f"http://192.168.50.245:8080/v1/models")
                for m in r.json()["running"]:
                    if m["alias"] == model_name and m["alive"]:
                        return
            except Exception:
                pass
            time.sleep(2)

        raise RuntimeError(f"모델 로드 타임아웃: {model_name}")


# ── Phase 5: LLM 생성 비교 ─────────────────────────────────────────────

LLM_MODELS = {
    "qwen3.5-27b":      {"location": "local", "base_url": GATEWAY_URL},
    "qwen3.5-35b-a3b":  {"location": "local", "base_url": GATEWAY_URL},
    "gpt-4o":           {"location": "api",   "base_url": None},
}

RAG_PROMPT_TEMPLATE = (
    "다음 검색된 문맥을 사용하여 질문에 답하세요. "
    "답을 모르면 모른다고 하세요. 최대 3문장으로 간결하게 답하세요.\n\n"
    "질문: {question}\n\n문맥:\n{context}\n\n답변:"
)


def run_phase5(
    best_parser: str = "pymupdf4llm",
    best_chunk: dict = None,
    embed_model: str = "qwen3-embed-8b",
    target_llm: str = None,
    top_k: int = 5,
):
    """LLM 생성 비교."""
    if best_chunk is None:
        best_chunk = {"chunk_size": 1000, "overlap": 200}

    gt = load_ground_truth()
    parser_fn = PARSERS[best_parser]

    # 청크 준비
    all_chunks = []
    parsed_cache = {}
    for gt_item in gt:
        fname = gt_item["target_file_name"]
        if fname not in parsed_cache:
            pdf_path = find_pdf(fname)
            if pdf_path:
                parsed_cache[fname] = parser_fn(str(pdf_path))

    for fname, pages in parsed_cache.items():
        chunks = chunk_pages(pages, best_chunk["chunk_size"], best_chunk["overlap"])
        for c in chunks:
            c["file"] = fname
        all_chunks.extend(chunks)

    # 임베딩 + 인덱스
    import numpy as np
    texts = [c["text"] for c in all_chunks]
    logger.info(f"청크 {len(all_chunks)}개 임베딩 중...")
    with Timer("embed"):
        chunk_embeddings = get_embeddings_batch(texts, GATEWAY_URL, embed_model)

    emb_array = np.array(chunk_embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_array = emb_array / norms

    # 쿼리 임베딩
    questions = [item["question"] for item in gt]
    with Timer("embed_queries"):
        query_embeddings = get_query_embeddings_batch(questions, GATEWAY_URL, embed_model)

    # LLM 모델별 테스트
    models = {target_llm: LLM_MODELS[target_llm]} if target_llm else LLM_MODELS
    all_results_summary = {}

    for llm_name, llm_info in models.items():
        logger.info(f"=== Phase 5: LLM={llm_name} ===")

        try:
            results = _run_llm_generation(
                llm_name, llm_info, gt, all_chunks,
                emb_array, query_embeddings, top_k,
            )

            # 자동 평가 (LLM-as-judge)
            judge_results = _auto_judge(results)

            o_count = sum(1 for r in judge_results if r["result"] == "O")
            total = len(judge_results)
            accuracy = o_count / total if total else 0

            # 도메인별 집계
            by_domain = {}
            for r in judge_results:
                d = r.get("domain", "unknown")
                by_domain.setdefault(d, {"O": 0, "X": 0})
                by_domain[d][r["result"]] += 1

            summary = {
                "total": total,
                "correct": o_count,
                "accuracy": round(accuracy, 4),
            }
            all_results_summary[llm_name] = summary

            save_result("phase5_llm", llm_name, {
                "llm": llm_name,
                "location": llm_info["location"],
                "top_k": top_k,
                "total": total,
                "correct": o_count,
                "accuracy": round(accuracy, 4),
                "by_domain": {d: {
                    "o": c["O"], "total": c["O"] + c["X"],
                    "accuracy": round(c["O"] / (c["O"] + c["X"]), 4),
                } for d, c in by_domain.items()},
                "results": judge_results,
            })

            logger.info(f"  정답률: {o_count}/{total} ({accuracy:.1%})")

        except Exception as e:
            logger.error(f"  실패: {e}", exc_info=True)
            all_results_summary[llm_name] = {"error": str(e)}

    # 요약 출력
    print(f"\n{'='*60}")
    print(f"  Phase 5: LLM 생성 비교")
    print(f"{'='*60}")
    print("%-25s %10s %10s" % ("LLM", "정답률", "정답/전체"))
    print("-" * 50)
    for name, s in sorted(all_results_summary.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
        if "error" in s:
            print("%-25s %10s" % (name, "ERROR"))
        else:
            print("%-25s %9.1f%% %5d/%-5d" % (name, s["accuracy"] * 100, s["correct"], s["total"]))
    print(f"  ref: allganize gpt-4-turbo = 61.0% (183/300)")

    return all_results_summary


def _run_llm_generation(
    llm_name: str,
    llm_info: dict,
    gt: List[Dict],
    chunks: List[Dict],
    emb_array,
    query_embeddings: List[List[float]],
    top_k: int,
) -> List[Dict]:
    """LLM으로 답변 생성."""
    import numpy as np

    results = []
    for i, (gt_item, q_emb) in enumerate(zip(gt, query_embeddings)):
        if not q_emb:
            continue

        # 검색
        q_vec = np.array(q_emb, dtype=np.float32)
        q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
        scores = emb_array @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]

        context = "\n\n---\n\n".join(chunks[idx]["text"] for idx in top_idx)
        prompt = RAG_PROMPT_TEMPLATE.format(
            question=gt_item["question"], context=context,
        )

        # LLM 호출
        try:
            if llm_info["location"] == "local":
                answer = _call_local_llm(llm_info["base_url"], llm_name, prompt)
            else:
                answer = _call_openai_llm(llm_name, prompt)
        except Exception as e:
            logger.warning(f"  [{i+1}/{len(gt)}] LLM 호출 실패: {e}")
            answer = ""

        if (i + 1) % 50 == 0:
            logger.info(f"  [{i+1}/{len(gt)}] 진행 중...")

        results.append({
            "question": gt_item["question"],
            "target_answer": gt_item["target_answer"],
            "target_file": gt_item["target_file_name"],
            "domain": gt_item["domain"],
            "context_type": gt_item["context_type"],
            "generated_answer": answer,
        })

    return results


def _call_local_llm(base_url: str, model: str, prompt: str) -> str:
    """로컬 게이트웨이 LLM 호출."""
    import httpx

    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 512,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def _call_openai_llm(model: str, prompt: str) -> str:
    """OpenAI API LLM 호출."""
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


def _auto_judge(results: List[Dict]) -> List[Dict]:
    """LLM-as-judge 자동 평가 (GPT-4o-mini)."""
    import os

    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    scored = []

    for r in results:
        if not r.get("generated_answer"):
            scored.append({**r, "result": "X", "votes": {}})
            continue

        votes = {}
        for metric in ["similarity", "correctness"]:
            judge_prompt = (
                f"Assess {metric} between generated and reference answer.\n\n"
                f"Question: {r['question']}\n"
                f"Reference: {r['target_answer']}\n"
                f"Generated: {r['generated_answer']}\n\n"
                f"Score 1-5 (5=best). Return ONLY the integer."
            )

            try:
                from openai import OpenAI
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0,
                    max_tokens=10,
                )
                text = resp.choices[0].message.content.strip()
                score = int(next(c for c in text if c.isdigit()))
                votes[metric] = score
            except Exception:
                votes[metric] = 0

        # 2개 메트릭 중 하나라도 4 이상이면 O
        o_count = sum(1 for s in votes.values() if s >= 4)
        result = "O" if o_count >= 1 else "X"

        scored.append({**r, "result": result, "votes": votes})

    return scored


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG 벤치마크")
    parser.add_argument("--phase", type=int, help="실행할 Phase (1~5)")
    parser.add_argument("--parser", type=str, default="pymupdf4llm", help="Phase 2+ 고정 파서")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--model", type=str, help="Phase 4: 특정 임베딩 모델만 테스트")
    parser.add_argument("--embed-model", type=str, default="qwen3-embed-8b", help="Phase 3/5: 임베딩 모델")
    parser.add_argument("--llm", type=str, help="Phase 5: 특정 LLM만 테스트")
    parser.add_argument("--top-k", type=int, default=5, help="Phase 5: 검색 top-k")
    parser.add_argument("--summary", action="store_true", help="전체 결과 요약")
    args = parser.parse_args()

    if args.summary:
        show_summary()
        return

    chunk_cfg = {"chunk_size": args.chunk_size, "overlap": args.chunk_overlap}

    if args.phase == 1:
        run_phase1()
    elif args.phase == 2:
        run_phase2(best_parser=args.parser)
    elif args.phase == 3:
        run_phase3(
            best_parser=args.parser,
            best_chunk=chunk_cfg,
            embed_model=args.embed_model,
        )
    elif args.phase == 4:
        run_phase4(
            best_parser=args.parser,
            best_chunk=chunk_cfg,
            target_model=args.model,
        )
    elif args.phase == 5:
        run_phase5(
            best_parser=args.parser,
            best_chunk=chunk_cfg,
            embed_model=args.embed_model,
            target_llm=args.llm,
            top_k=args.top_k,
        )
    else:
        print("Usage: python bench_all.py --phase {1,2,3,4,5}")
        print("       python bench_all.py --summary")


def show_summary():
    """전체 Phase 결과 요약."""
    for phase_dir in sorted(RESULTS_DIR.iterdir()):
        if not phase_dir.is_dir():
            continue
        results = {}
        for f in sorted(phase_dir.glob("*.json")):
            data = json.loads(f.read_text())
            name = f.stem
            if "metrics" in data:
                results[name] = data["metrics"]
        if results:
            print_metrics_table(results, phase_dir.name)


if __name__ == "__main__":
    main()
