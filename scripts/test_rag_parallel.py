"""Verify that the warmup + thread-lock fix in rag/embeddings.py
eliminates the meta-tensor race when 5 masters retrieve in parallel."""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from rag.embeddings import warmup_local_model
from rag.retriever import retrieve

print("[1] warmup_local_model ... ", end="", flush=True)
t0 = time.time()
warmup_local_model()
print(f"done in {time.time()-t0:.1f}s")

masters = ["buffett", "dalio", "marks", "greenblatt", "lynch"]
query = "当前A股大盘指数基金的估值水平是否合理？"

print("\n[2] 5 parallel retrieve() calls (simulating LangGraph fan-out):")


def _one(master):
    t = time.time()
    try:
        docs = retrieve(master, query, top_k=3, use_self_query=False)
        return master, len(docs), time.time() - t, None
    except Exception as e:  # noqa: BLE001
        return master, 0, time.time() - t, str(e)[:120]


with ThreadPoolExecutor(max_workers=5) as pool:
    futs = [pool.submit(_one, m) for m in masters]
    for fut in as_completed(futs):
        master, n, dt, err = fut.result()
        if err:
            print(f"   ❌ {master:<12}  {dt:5.2f}s   ERROR: {err}")
        else:
            print(f"   ✅ {master:<12}  {dt:5.2f}s   got {n} docs")

print("\nDone.")
