"""Quick smoke test: LLM factory + RAG retrieval."""
import os
import sys
import time
from pathlib import Path

# Add project root so `llm_provider` and `rag` are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("LLM_PROVIDER", "deepseek")
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-placeholder")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("=" * 50)
print("[0] Starting smoke test", flush=True)

print("[1] Testing llm factory...", flush=True)
from llm_provider import get_chat_llm, current_provider_summary
print(f"     chat config: {current_provider_summary()}", flush=True)
llm = get_chat_llm(temperature=0.7)
print(f"     ChatOpenAI model={llm.model_name}", flush=True)

print("[2] Testing RAG retrieve (will load BGE-M3 from cache)...", flush=True)
t0 = time.time()
from rag.retriever import retrieve
print(f"     imported in {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
docs = retrieve("buffett", "价值投资的核心是什么", top_k=3)
print(f"     got {len(docs)} docs in {time.time()-t0:.1f}s", flush=True)

for i, d in enumerate(docs[:2]):
    preview = d.page_content[:100].replace("\n", " ")
    print(f"     doc {i+1}: {preview}...", flush=True)

print("[3] DONE — full pipeline wired end-to-end.", flush=True)
