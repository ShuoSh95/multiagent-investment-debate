"""End-to-end smoke tests for the refactored debate pipeline.

Runs FAST (no full debate) — exercises only:
  1. web_search.perform_web_search — is Gemini grounding reachable?
  2. stratified_shuffle — distribution sanity
  3. should_continue — early-stop logic unit tests
  4. main.app compile — graph builds without error
"""
import os
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("TEST 1 — web_search.perform_web_search (Gemini grounding)")
print("=" * 60)

from web_search import perform_web_search, get_call_stats

t0 = time.time()
text, source = perform_web_search(
    query="特斯拉 (TSLA) 最新股价与 PE 估值",
    focus="2026 年 Q1 财报",
)
dt = time.time() - t0
print(f"  source: {source}  elapsed: {dt:.1f}s")
if text:
    print(f"  got {len(text)} chars. preview:")
    print("  " + text[:400].replace("\n", "\n  "))
else:
    print("  ⚠️ empty result")
print(f"  call stats: {get_call_stats()}")

print()
print("=" * 60)
print("TEST 2 — stratified_shuffle distribution over 1000 trials")
print("=" * 60)

from main import stratified_shuffle, MASTER_NAMES

first_counter: Counter = Counter()
pairwise: Counter = Counter()
for _ in range(1000):
    order = stratified_shuffle(MASTER_NAMES)
    first_counter[order[0]] += 1
    for i in range(len(order) - 1):
        pairwise[(order[i], order[i + 1])] += 1

print("  first-position frequency (ideal ~200 each):")
for name, cnt in first_counter.most_common():
    bar = "█" * (cnt // 10)
    print(f"    {name:<16} {cnt:>4}  {bar}")

print()
print("=" * 60)
print("TEST 3 — should_continue early-stop logic")
print("=" * 60)

from main import should_continue, MASTER_NAMES as NAMES

def _is_end(result):
    return result == "Researcher"


def _is_continue(result):
    # Continue = a list of master names (fan-out)
    return isinstance(result, list) and len(result) == len(NAMES)


# Case A: full votes collected → end
s = {
    "round_count": 2,
    "votes": {n: "看多" for n in NAMES},
    "tendencies": {},
}
assert _is_end(should_continue(s)), "Case A failed"
print("  ✅ Case A (full votes) → end")

# Case B: only 2 rounds, no votes, no consensus → continue
s = {
    "round_count": 2,
    "votes": {},
    "tendencies": {
        1: {"沃伦·巴菲特": "看多", "瑞·达利欧": "看空", "霍华德·马克斯": "观望"},
        2: {"沃伦·巴菲特": "看多", "瑞·达利欧": "看空"},
    },
}
assert _is_continue(should_continue(s)), "Case B failed"
print("  ✅ Case B (no consensus) → continue")

# Case C: 2 consecutive rounds unanimous 看多 (3+ speakers each) → end
s = {
    "round_count": 3,
    "votes": {},
    "tendencies": {
        2: {"沃伦·巴菲特": "看多", "瑞·达利欧": "看多", "霍华德·马克斯": "看多"},
        3: {"沃伦·巴菲特": "看多", "瑞·达利欧": "看多", "彼得·林奇": "看多"},
    },
}
assert _is_end(should_continue(s)), "Case C failed"
print("  ✅ Case C (2-round unanimous 看多, ≥3 speakers) → end")

# Case D: unanimous but only 2 speakers → NOT early-stop
s = {
    "round_count": 3,
    "votes": {},
    "tendencies": {
        2: {"沃伦·巴菲特": "看多", "瑞·达利欧": "看多"},
        3: {"沃伦·巴菲特": "看多", "瑞·达利欧": "看多"},
    },
}
assert _is_continue(should_continue(s)), "Case D failed"
print("  ✅ Case D (unanimous but <3 speakers) → continue")

# Case E: at max round → end
s = {"round_count": 6, "votes": {}, "tendencies": {}}
assert _is_end(should_continue(s)), "Case E failed"
print("  ✅ Case E (at MAX_ROUNDS) → end")

# Case F: everyone 观望 → not consensus for early-stop
s = {
    "round_count": 3,
    "votes": {},
    "tendencies": {
        2: {n: "观望" for n in NAMES},
        3: {n: "观望" for n in NAMES},
    },
}
assert _is_continue(should_continue(s)), "Case F failed"
print("  ✅ Case F (all 观望 is not consensus) → continue")

print()
print("=" * 60)
print("TEST 4 — graph compile sanity")
print("=" * 60)

from main import app, MASTER_NAMES as N
nodes = list(app.get_graph().nodes.keys())
print(f"  graph nodes: {nodes}")
for n in N:
    assert n in nodes, f"master {n} missing from graph"
for key in ["DataCollection", "CrossQuestion", "UpdateRound", "Researcher"]:
    assert key in nodes, f"{key} missing from graph"
print("  ✅ all expected nodes present")

print()
print("=" * 60)
print("ALL TESTS PASSED ✅")
print("=" * 60)
