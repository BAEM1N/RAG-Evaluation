#!/usr/bin/env python3
"""
allganize 공식 LangChain Tutorial baseline 재현 (병렬 처리)
출처: https://colab.research.google.com/drive/1Jlzs8ZqFOqqIBBT2T5XGBhr23XxEsvHb

Pipeline:
  PyPDFLoader → RecursiveCharacterTextSplitter(1000/200)
  → OpenAIEmbeddings (ada-002) → Chroma → k=6
  → gpt-4-turbo + rlm/rag-prompt

병렬화:
  1. 58개 파일별 벡터스토어 사전 빌드 (ThreadPoolExecutor)
  2. 300개 질의 병렬 실행 (ThreadPoolExecutor)
"""
import os
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
RESULTS_DIR = ROOT / "results" / "baseline_langchain"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4-turbo")
INDEX_WORKERS = int(os.environ.get("INDEX_WORKERS", "8"))  # 파일 벡터화 동시 수
QUERY_WORKERS = int(os.environ.get("QUERY_WORKERS", "16"))  # 질의 동시 수

# rlm/rag-prompt 원본
RLM_RAG_PROMPT = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


_collection_counter = 0
_counter_lock = __import__('threading').Lock()


def _next_collection_name() -> str:
    global _collection_counter
    with _counter_lock:
        _collection_counter += 1
        return f"rag_bench_{_collection_counter}_{int(time.time() * 1000)}"


def build_vectorstore(pdf_path: Path) -> Chroma:
    """PDF → VectorStore (in-memory, 고유 콜렉션명)."""
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    return Chroma.from_documents(
        documents=all_splits,
        embedding=OpenAIEmbeddings(),
        collection_name=_next_collection_name(),
    )


def build_chain(vectorstore: Chroma, llm_model: str):
    """RAG 체인 생성."""
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )
    llm = ChatOpenAI(model=llm_model, temperature=0)
    prompt = PromptTemplate.from_template(RLM_RAG_PROMPT)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


def find_pdf(filename: str) -> Path:
    for domain in ["finance", "public", "medical", "law", "commerce"]:
        p = PDF_DIR / domain / filename
        if p.exists():
            return p
    return None


def main():
    with open(DATA_DIR / "ground_truth.json", encoding="utf-8") as f:
        gt = json.load(f)
    print(f"Ground Truth: {len(gt)} questions")
    print(f"LLM: {LLM_MODEL}")
    print(f"Index workers: {INDEX_WORKERS}, Query workers: {QUERY_WORKERS}")

    start_ts = time.time()

    # --- 1단계: 파일별 벡터스토어 병렬 빌드 ---
    unique_files = sorted(set(r["target_file_name"] for r in gt))
    print(f"\n[1/2] 벡터스토어 빌드: {len(unique_files)} files")

    vectorstores = {}
    lock = Lock()

    def _build_one(fname):
        pdf_path = find_pdf(fname)
        if not pdf_path:
            return fname, None, "PDF_NOT_FOUND"
        try:
            vs = build_vectorstore(pdf_path)
            return fname, vs, None
        except Exception as e:
            return fname, None, str(e)

    with ThreadPoolExecutor(max_workers=INDEX_WORKERS) as ex:
        futures = {ex.submit(_build_one, f): f for f in unique_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Index"):
            fname, vs, err = fut.result()
            if vs is not None:
                vectorstores[fname] = vs

    index_time = time.time() - start_ts
    print(f"  빌드 완료: {len(vectorstores)}/{len(unique_files)} ({index_time:.0f}s)")

    # --- 2단계: 질의 병렬 실행 ---
    print(f"\n[2/2] 질의 실행: {len(gt)} questions (workers={QUERY_WORKERS})")
    query_start = time.time()

    def _run_query(item):
        fname = item["target_file_name"]
        question = item["question"]
        vs = vectorstores.get(fname)
        if vs is None:
            return {**item, "generated_answer": None, "error": "NO_VECTORSTORE"}
        try:
            chain = build_chain(vs, LLM_MODEL)
            t0 = time.time()
            answer = chain.invoke(question)
            elapsed = time.time() - t0
            return {
                **item,
                "generated_answer": answer,
                "latency_sec": round(elapsed, 2),
                "error": None,
            }
        except Exception as e:
            return {**item, "generated_answer": None, "error": str(e)}

    results = []
    with ThreadPoolExecutor(max_workers=QUERY_WORKERS) as ex:
        futures = [ex.submit(_run_query, item) for item in gt]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Query"):
            results.append(fut.result())

    query_time = time.time() - query_start
    total_time = time.time() - start_ts

    # 저장
    ok = sum(1 for r in results if r.get("generated_answer"))
    err = sum(1 for r in results if r.get("error"))
    path = RESULTS_DIR / f"baseline_{LLM_MODEL}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "llm_model": LLM_MODEL,
            "pipeline": "allganize_langchain_tutorial",
            "total": len(results),
            "answered": ok,
            "errors": err,
            "index_time_sec": round(index_time, 1),
            "query_time_sec": round(query_time, 1),
            "total_time_sec": round(total_time, 1),
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n=== 완료 ===")
    print(f"  총 {len(results)}개 | 응답 {ok}개 | 에러 {err}개")
    print(f"  인덱스 빌드: {index_time:.0f}s")
    print(f"  질의 실행: {query_time:.0f}s")
    print(f"  전체: {total_time:.0f}s ({total_time/60:.1f}분)")
    print(f"  저장: {path}")


if __name__ == "__main__":
    main()
