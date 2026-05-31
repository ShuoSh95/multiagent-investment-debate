"""Live API ping to verify Gemini key and model availability.

Sends one tiny request (~30 tokens total) to make sure:
  - the API key is valid
  - the region is reachable from this machine
  - the selected model name is usable
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from llm_provider import get_chat_llm, current_provider_summary

print("=" * 60)
print(f"Provider: {current_provider_summary()}", flush=True)

# Try default model first
candidates = [os.getenv("LLM_MODEL") or "gemini-2.5-flash",
              "gemini-2.5-pro",
              "gemini-2.0-flash",
              "gemini-1.5-flash"]

seen = set()
for model in candidates:
    if model in seen:
        continue
    seen.add(model)
    print(f"\n[test] → model={model}", flush=True)
    try:
        t0 = time.time()
        llm = get_chat_llm(temperature=0.2, model=model)
        resp = llm.invoke("Reply with exactly one word: PONG")
        dt = time.time() - t0
        text = resp.content if hasattr(resp, "content") else str(resp)
        print(f"        ✅ OK in {dt:.1f}s | reply: {text.strip()[:80]}")
        # Token usage if exposed
        meta = getattr(resp, "usage_metadata", None)
        if meta:
            print(f"        tokens: input={meta.get('input_tokens')} output={meta.get('output_tokens')}")
        break
    except Exception as e:
        msg = str(e)
        print(f"        ❌ failed: {msg[:200]}")

print("\nDone.")
