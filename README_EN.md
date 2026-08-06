# 🏛️ Multi-Agent Investment Debate

[中文](README.md) | **English**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B.svg)](https://streamlit.io/)

> Watch AI agents of **Warren Buffett, Ray Dalio, Howard Marks, Joel Greenblatt, and Peter Lynch**
> debate your investment question live — each grounded in RAG over their own writings plus
> real-time web data — then get a structured research report from a neutral researcher agent.

📖 **[Read a real debate transcript →](docs/example_debate.md)** (in Chinese; question: "US–Iran peace breakthrough — time to go all-in on China A-share index funds?" — 5 masters, 6 rounds, unanimous bearish vote)

---

> [!WARNING]
> **Disclaimer:** This project is for AI / multi-agent research and education only. It is **not**
> financial advice. All "master" personas are AI simulations based on public materials and do not
> represent the real individuals' views. LLM outputs may contain errors or hallucinations.
> Any investment decisions are solely your own responsibility.

---

## ✨ Highlights

| Capability | Description |
|---|---|
| 🖥️ **Web UI** | Streamlit app: submit a question, watch the debate stream live, ask follow-ups afterwards |
| 🎭 **5 master agents** | Each with a distinct persona, thinking framework, and debate bottom-line prompt |
| 📚 **RAG on original writings** | Every turn re-retrieves from the master's own books / memos / shareholder letters (local BGE-M3 + ChromaDB + BM25 + RRF fusion) |
| 🔍 **On-demand data fetching** | Each round, every agent self-assesses (per its own framework) whether it lacks key data, and only then triggers a web search — Dalio asks for macro data, Greenblatt asks for ROIC |
| 🛡️ **No-fabrication rule** | Agents are hard-constrained to only cite retrieved data; on retrieval failure they must state the data is missing rather than make numbers up |
| 🤝 **Real debate, not monologues** | A moderator generates cross-examination questions after every round, forcing agents to rebut each other |
| 🏁 **Smart early stopping** | Converges early when all agents have voted or sentiment aligns for 2 consecutive rounds |
| 💬 **Post-debate Q&A** | Chat with a researcher agent that has the full debate context, streaming answers |
| 🗂️ **Debate history** | Every debate auto-saved to local SQLite; replay and continue asking from the sidebar |
| 🔌 **Multi-provider** | Switch between Gemini / DeepSeek / OpenAI / Claude / Qwen / GLM / Doubao with one `.env` line |

**Model split (default Gemini):** debate reasoning + final report + follow-ups use `gemini-2.5-pro`; data self-assessment + web search use `gemini-2.5-flash`.

---

## 🏗️ Architecture

```
User question
     │
     ▼
DataCollection ──► web_search (Gemini Google-Search grounding / Tavily fallback)
     │
     │  (fan-out, parallel)
     ▼
┌─────────────────────────────────────────────┐
│  Buffett   Dalio   Marks   Greenblatt  Lynch│
│  each turn:                                 │
│   1. RAG over own writings (BGE-M3, local)  │
│   2. self-assess data needs → optional web  │
│      search (strict no-fabrication rule)    │
│   3. speak: view + [tendency] + [vote?]     │
└─────────────────────┬───────────────────────┘
                      ▼
              CrossQuestion (moderator)
                      ▼
              UpdateRound ──► continue? ──► next round (≤6)
                      │
                      ▼ (all voted / 2 rounds aligned / max rounds)
                 Researcher ──► final structured report
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/ShuoSh95/multiagent-investment-debate.git
cd multiagent-investment-debate

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill in your Gemini API key (free tier works)

# Build the knowledge base (first run only, ~15-25 min;
# downloads BGE-M3 ~2.3GB + scrapes public sources)
python -m rag.build_kb --master all --rebuild

# Launch the web UI
streamlit run web/streamlit_app.py    # open http://localhost:8501

# ...or the terminal UI
python main.py
```

**Requirements:** Python 3.9+ (3.10+ recommended), ~5GB disk. Apple Silicon (MPS) or CUDA speeds up local embedding, CPU works too.

**Minimal `.env`:**

```bash
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_gemini_key      # free at https://aistudio.google.com/apikey
EMBEDDING_PROVIDER=local
```

See [.env.example](.env.example) for the full configuration (model split, provider switching, on-demand search toggle).

---

## 📚 Knowledge Base

Sources are tiered by credibility: the masters' own publications / official speeches / shareholder letters (Tier 1), high-quality interviews (Tier 2), Wikipedia (Tier 3, fallback). Raw corpora are **not** included in this repo (copyright + size); the build script scrapes public sources, and you can drop your own purchased e-books into `data/raw/<master>/` to enrich a master.

Retrieval is hybrid: BGE-M3 dense vectors + BM25 keywords, fused with Reciprocal Rank Fusion, with an LLM self-query rewrite on the first round. From round 2 on, queries are augmented with the live debate focus so retrieved passages track the argument.

---

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) (in Chinese). Planned: @mention a specific master in follow-ups, true serial debate (later speakers see earlier same-round speeches), mid-debate user interjection, cross-debate memory.

## 📄 License

[MIT](LICENSE) — with an explicit investment disclaimer. Master personas are AI simulations; no affiliation with or endorsement by the real individuals.
