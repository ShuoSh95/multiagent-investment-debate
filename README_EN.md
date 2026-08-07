# 🏛️ Multi-Agent Voting Arena · AI Investing-Legends Debate

[中文](README.md) | **English**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B.svg)](https://streamlit.io/)

---

You spot a sector rallying hard. One voice in your head yells **"Buy now — chances like this come once a decade!"**
The other warns **"Don't chase the top, you'll be the exit liquidity."**

If those two voices argue in your head too, outsource the fight:

**Let AI agents of Warren Buffett, Ray Dalio, Howard Marks, Joel Greenblatt, and Peter Lynch argue it out for you.**

They quote their own books and shareholder letters, fetch live market data mid-argument,
cross-examine each other, sometimes flip sides, then cast a final vote — and a neutral
researcher agent writes you a structured report.

📖 **[Watch a real debate transcript →](docs/example_debate.md)** (in Chinese; "US–Iran peace breakthrough — time to go all-in on China A-share index funds?" — 5 masters, 6 rounds, unanimous bearish vote)

---

> [!IMPORTANT]
> **Disclaimer:** This project is a research and entertainment demo of multi-agent AI. It is **not**
> financial, investment, or trading advice. Debate outcomes vary with the chosen language models,
> temperature, data quality, and other non-deterministic factors. All "master" personas are AI
> simulations based on public materials — they do not represent the real individuals' views, and have
> no affiliation with or endorsement from them. LLM outputs may contain errors or hallucinations.
> Any investment decisions and their consequences are solely your own responsibility.

---

## 🎭 What is this?

An AI investment debate you can **watch live, end to end**. You submit the motion, the court is in session:

```
Your question ──► Pre-trial research (live prices / valuations / news)
                        │
                        ▼
        5 masters debate over multiple rounds (≤6)
        ├─ each consults their own writings before speaking (RAG)
        ├─ short on evidence? they request live data themselves
        ├─ a moderator sparks cross-examination every round
        └─ each declares a stance: bullish / bearish / hold (can flip)
                        │
                        ▼  (all voted, or aligned 2 rounds in a row)
        🗳️ Final vote ──► 📝 Researcher's closing report ──► 💬 Ask follow-ups
```

### The five debaters

| Debater | School | Style |
|---|---|---|
| 🎩 Warren Buffett | Value | Moats and margin of safety; "better to miss than to lose" |
| 🌐 Ray Dalio | Global macro | Debt cycles and the economic machine; systemic-risk alarms |
| ⚖️ Howard Marks | Cycles / contrarian | The more euphoric the market, the more nervous he gets |
| 📊 Joel Greenblatt | Quantitative value | Only ROIC and earnings yield matter; macro is noise |
| 🛒 Peter Lynch | Growth / common sense | Finds ten-baggers in shopping malls |

Their schools clash by design — that's why the arguments get heated.

## ✨ Why it's worth watching

**1. They don't just roleplay — they cite the sources.**
Before every turn, each agent retrieves from a knowledge base of their own writings
(Buffett's shareholder letters, Marks' memos, Dalio's *Principles*... ~12,700 passages). Key claims quote the originals.

**2. They admit when they don't know.**
Agents self-assess data gaps and trigger their own web searches (Dalio pulls macro rates, Greenblatt pulls ROIC).
If retrieval fails, they must say "I lack that data" — fabricating numbers is forbidden at the system level.

**3. Real clashes, real flips.**
Cross-examination forces rebuttals. You'll see an agent argued from "hold" into "bearish", and another besieged yet unmoved.
Not five parallel essays — an actual debate with turning points.

**4. You leave with something useful.**
A neutral researcher distills consensus, disagreements, the vote, and risk caveats into a report — then answers your follow-up questions with full debate context.

## 🚀 Quick Start

Requires Python 3.9+, ~5GB disk, and a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/ShuoSh95/multiagent-investment-debate.git
cd multiagent-investment-debate
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # fill in your GOOGLE_API_KEY

python -m rag.build_kb --master all --rebuild   # build the knowledge base (first run, ~15-25 min)

streamlit run web/streamlit_app.py               # open http://localhost:8501 — court is in session
```

Other LLM providers (DeepSeek / OpenAI / Claude / Qwen...), feeding your own e-books to a master,
retrieval internals, debugging → **[full setup & technical docs](docs/SETUP.md)** (in Chinese).

## 🗺️ Roadmap

- [ ] 🌐 Public online demo (watch without installing)
- [ ] @mention a specific master in follow-ups
- [ ] True serial debate (later speakers see earlier same-round speeches)
- [ ] Mid-debate user interjection

See [ROADMAP.md](ROADMAP.md) (in Chinese).

## 📄 License

[MIT](LICENSE). Master personas are AI simulations with no affiliation with or endorsement by the real individuals; outputs are not investment advice — see the disclaimer above.
