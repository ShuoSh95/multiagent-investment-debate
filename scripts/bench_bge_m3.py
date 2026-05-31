"""
BGE-M3 local benchmark — measures wall-clock time for loading the model
and encoding representative queries on this machine.

Usage:
    HF_ENDPOINT=https://hf-mirror.com python scripts/bench_bge_m3.py
"""

import os
import sys
import time

if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
import torch

MODEL = "BAAI/bge-m3"


def main():
    print(f"Python:        {sys.version.split()[0]}")
    print(f"torch:         {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available:{torch.cuda.is_available()}")
    print(f"HF_ENDPOINT:   {os.environ.get('HF_ENDPOINT')}")
    print()

    t0 = time.time()
    print(f"[1/3] Loading {MODEL} ...")
    model = SentenceTransformer(MODEL)
    load_sec = time.time() - t0
    print(f"      load time: {load_sec:.1f}s  device={model.device}")
    print()

    queries_cn = [
        "在高通胀环境下应如何配置资产？",
        "这家公司的护城河和估值是否合理？",
        "市场出现泡沫迹象时应该如何应对？",
    ]
    queries_en = [
        "How should I diversify in a stagflation regime?",
        "Is this company's moat durable and valuation fair?",
    ]
    all_q = queries_cn + queries_en

    print("[2/3] Warmup encode (1 query) ...")
    t0 = time.time()
    _ = model.encode([all_q[0]], normalize_embeddings=True, show_progress_bar=False)
    warmup_ms = (time.time() - t0) * 1000
    print(f"      warmup: {warmup_ms:.0f}ms")
    print()

    print("[3/3] Encoding 5 real queries (sequential)...")
    timings = []
    for i, q in enumerate(all_q):
        t0 = time.time()
        v = model.encode([q], normalize_embeddings=True, show_progress_bar=False)
        ms = (time.time() - t0) * 1000
        timings.append(ms)
        print(f"      q{i+1} ({len(q)} chars): {ms:6.0f}ms  dim={v.shape[1]}")

    print()
    print("=" * 50)
    print(f"Avg per-query encode: {sum(timings)/len(timings):.0f}ms")
    print(f"Model load one-time : {load_sec:.1f}s")
    print("=" * 50)
    print()
    print("Translation to debate experience:")
    per_q = sum(timings) / len(timings)
    per_round = per_q * 5  # 5 masters x 1 query each
    print(f"  Per master-turn retrieval: ~{per_q:.0f}ms (negligible vs LLM call)")
    print(f"  Per debate round (5 masters): ~{per_round:.0f}ms total")


if __name__ == "__main__":
    main()
