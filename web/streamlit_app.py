"""AI Investment Decision Maker — Streamlit Web UI (v1.0).

Run from project root:
    cd /path/to/InvestmentAgent
    ./.venv/bin/streamlit run web/streamlit_app.py

Features (v1.0):
    - Live debate streaming: market data → 5 masters per round → cross-Q → final report
    - Follow-up chat: single unified assistant answers questions about the report
    - History sidebar: SQLite-backed, reload any past debate

See ROADMAP.md for planned v2.0 upgrades (per-master @mentions, true
serial debate, mid-debate interjection).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Make project root importable when launched via `streamlit run`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env before importing modules that read env vars
from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import streamlit as st


def _sync_streamlit_secrets_to_env() -> None:
    """Streamlit Cloud injects secrets via st.secrets; our code reads os.environ.
    Copy top-level string secrets into the environment so Gemini / HF / DEMO_*
    all work the same way locally and on Cloud."""
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in secrets:
        try:
            val = secrets[key]
        except Exception:
            continue
        if isinstance(val, (str, int, float, bool)):
            os.environ.setdefault(str(key), str(val))


_sync_streamlit_secrets_to_env()

# =====================================================================
#  Page config & style  (must be the first Streamlit "draw" call)
# =====================================================================
st.set_page_config(
    page_title="AI 投资决策器 · 多大师辩论",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from web import history_db, rate_limit
from web.debate_runner import run_debate
from llm_provider import current_provider_summary, get_chat_llm, is_demo_mode

# ---- Hosted-deployment bootstrap (both are no-ops on a local install) --
# 1. Seed showcase debates into an empty history DB (ephemeral disk on
#    Cloud starts blank after every restart).
history_db.seed_from_gallery(_ROOT / "web" / "gallery")
# 2. Fetch the vector store / BM25 index from the private HF dataset if
#    they are missing.
_kb_ok = True
try:
    from rag.fetch_kb import ensure_kb

    _kb_ok = bool(ensure_kb())
except Exception as _e:  # noqa: BLE001
    _kb_ok = False
    print(f"[bootstrap] KB fetch skipped: {_e}")


def _missing_cloud_config() -> list[str]:
    """Return human-readable list of required secrets that are absent."""
    missing: list[str] = []
    gkey = os.getenv("GOOGLE_API_KEY", "")
    if not gkey or gkey.startswith("your_"):
        missing.append("GOOGLE_API_KEY（Gemini）")
    if is_demo_mode() and not os.getenv("HF_TOKEN"):
        missing.append("HF_TOKEN（用于拉取私有知识库）")
    return missing

_CUSTOM_CSS = """
<style>
    .tendency-chip {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-right: 6px;
    }
    .tendency-看多 { background: #dcfce7; color: #166534; }
    .tendency-看空 { background: #fee2e2; color: #991b1b; }
    .tendency-观望 { background: #fef3c7; color: #92400e; }
    .vote-final {
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 6px;
        background: #1e293b;
        color: white;
        font-size: 0.85rem;
    }
    .round-header {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 600;
        margin: 18px 0 14px 0;
    }
    .cross-q {
        background: #fff7ed;
        border-left: 3px solid #f97316;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    .search-note {
        font-size: 0.82rem;
        padding: 6px 12px;
        border-radius: 6px;
        margin: 4px 0 8px 0;
    }
    .search-ok { background: #eff6ff; color: #1e40af; border-left: 3px solid #3b82f6; }
    .search-fail { background: #fef2f2; color: #991b1b; border-left: 3px solid #ef4444; }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
#  Session state init
# =====================================================================
def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "stage": "idle",            # idle | running | done
        "query": "",
        "market_data": "",
        "rounds": [],               # list[dict{round, turns, cross_question, votes}]
        "final_report": "",
        "debate_id": None,          # int after save
        "followup": [],             # list[dict{role, content}]
        "loaded_from_history": False,
        # NOT reset by 新建辩论 — enforces the per-session demo cap
        "debates_started": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# =====================================================================
#  Sidebar: history + reset
# =====================================================================
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🏛️ 投资决策器")
        st.caption(f"Chat: `{current_provider_summary()}`")

        st.divider()
        if st.button("🆕 新建辩论", use_container_width=True, type="primary"):
            for k in ("stage", "query", "market_data", "rounds",
                     "final_report", "debate_id", "followup",
                     "loaded_from_history"):
                st.session_state.pop(k, None)
            _init_state()
            st.rerun()

        st.divider()
        if is_demo_mode():
            st.markdown("#### 🎬 精选回放 & 历史辩论")
            used = history_db.count_debates_today()
            st.caption(f"今日公共辩论额度：{used}/{rate_limit.daily_limit()}")
        else:
            st.markdown("#### 📚 历史辩论")
        debates = history_db.list_debates(limit=30)
        if not debates:
            st.caption("（暂无历史记录）")
        for d in debates:
            q_preview = d["query"][:30] + ("…" if len(d["query"]) > 30 else "")
            if st.button(
                f"**#{d['id']}** · {q_preview}\n\n_{d['created_at']}_",
                key=f"hist_{d['id']}",
                use_container_width=True,
            ):
                _load_from_history(d["id"])
                st.rerun()


def _load_from_history(debate_id: int) -> None:
    data = history_db.load_debate(debate_id)
    if not data:
        return
    t = data["transcript"]
    st.session_state.update(
        stage="done",
        query=data["query"],
        market_data=t.get("market_data", ""),
        rounds=t.get("rounds", []),
        final_report=data["final_report"],
        debate_id=data["id"],
        followup=data.get("followup", []),
        loaded_from_history=True,
    )


# =====================================================================
#  Rendering helpers
# =====================================================================
def _render_market_card(md: str) -> None:
    if not md:
        return
    with st.expander("📊 实时市场数据（Web Search）", expanded=False):
        st.markdown(md)


def _render_tendency_chip(t: str | None) -> str:
    if not t:
        return ""
    return f'<span class="tendency-chip tendency-{t}">本轮倾向: {t}</span>'


def _render_master_turn(turn: Dict[str, Any]) -> None:
    name = turn["master"]
    emoji = turn["emoji"]

    with st.chat_message(name=name, avatar=emoji):
        if turn.get("skipped"):
            _render_search_notes(turn.get("search"))
            st.markdown(f"**{name}**  💤 _本轮选择不发言_")
            return

        header_parts = [f"**{name}**"]
        if turn.get("tendency"):
            header_parts.append(_render_tendency_chip(turn["tendency"]))
        if turn.get("vote"):
            header_parts.append(
                f'<span class="vote-final">📌 最终投票: {turn["vote"]}</span>'
            )
        st.markdown(" &nbsp; ".join(header_parts), unsafe_allow_html=True)
        _render_search_notes(turn.get("search"))
        st.markdown(turn["content"])


def _render_search_notes(search_log: List[Dict[str, Any]] | None) -> None:
    """Show any on-demand web searches this master triggered (Issue #1)."""
    if not search_log:
        return
    for s in search_log:
        query = s.get("query", "")
        if s.get("found"):
            st.markdown(
                f'<div class="search-note search-ok">🔍 主动检索数据：'
                f'<b>{query}</b> &nbsp;✅ 已获取（来源:{s.get("source","")}）</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="search-note search-fail">🔍 主动检索数据：'
                f'<b>{query}</b> &nbsp;⚠️ 未获取到可靠数据，已要求如实说明</div>',
                unsafe_allow_html=True,
            )


def _render_cross_question(entries: List[str]) -> None:
    if not entries:
        return
    st.markdown("**💬 交叉质疑**")
    for line in entries:
        st.markdown(f'<div class="cross-q">{line}</div>', unsafe_allow_html=True)


def _render_round(r: Dict[str, Any]) -> None:
    st.markdown(
        f'<div class="round-header">第 {r["round"]} 轮辩论</div>',
        unsafe_allow_html=True,
    )
    for turn in r.get("turns", []):
        _render_master_turn(turn)
    _render_cross_question(r.get("cross_question", []))


def _render_all_past_rounds() -> None:
    """Re-render everything currently in session state (used when loading
    from history, or when redrawing at the end of a live debate)."""
    _render_market_card(st.session_state["market_data"])
    for r in st.session_state["rounds"]:
        _render_round(r)
    if st.session_state["final_report"]:
        st.divider()
        st.markdown("## 📝 最终投资决策报告")
        st.markdown(st.session_state["final_report"])


# =====================================================================
#  Live debate streaming
# =====================================================================
def _stream_debate(query: str) -> None:
    """Run the debate in a streaming loop. Assumes session state has been
    reset to 'running' BEFORE this is called (see _start_debate)."""

    # Live containers — as events arrive we write into them; Streamlit
    # flushes incremental updates to the browser automatically.
    market_slot = st.empty()
    round_container = st.container()
    status_slot = st.empty()

    status_slot.info("🔎 正在采集市场数据（Gemini Grounding）……请稍候")
    t0 = time.time()

    current_round_data: Dict[str, Any] | None = None
    round_slot = None  # placeholder for the currently-rendering round

    for ev in run_debate(query):
        etype = ev["type"]

        if etype == "market_data":
            st.session_state["market_data"] = ev["text"]
            with market_slot.container():
                _render_market_card(ev["text"])

        elif etype == "round_start":
            current_round_data = {
                "round": ev["round"],
                "turns": [],
                "cross_question": [],
                "votes": {},
            }
            st.session_state["rounds"].append(current_round_data)
            round_slot = round_container.container()
            with round_slot:
                st.markdown(
                    f'<div class="round-header">第 {ev["round"]} 轮辩论 · '
                    f'大师思考中……</div>',
                    unsafe_allow_html=True,
                )
            status_slot.info(
                f"🧠 第 {ev['round']} 轮：{len(ev.get('order', []))} 位大师正在思考……"
            )

        elif etype == "master_turn":
            if current_round_data is not None:
                current_round_data["turns"].append(ev)
            with round_slot:
                _render_master_turn(ev)
            status_slot.info(
                f"✨ {ev['emoji']} {ev['master']} 已发言（第 {ev['round']} 轮）"
            )

        elif etype == "cross_question":
            if current_round_data is not None:
                current_round_data["cross_question"] = ev["entries"]
            with round_slot:
                _render_cross_question(ev["entries"])

        elif etype == "round_end":
            if current_round_data is not None:
                current_round_data["votes"] = ev.get("votes", {})
            reason = ev.get("reason", "")
            if reason == "max_rounds":
                status_slot.warning(f"⏰ 第 {ev['round']} 轮结束（已达最大轮次）")
            elif reason == "consensus_or_early_stop":
                status_slot.success(
                    f"🏁 第 {ev['round']} 轮达成共识，辩论提前收敛"
                )
            else:
                status_slot.info(f"🔁 第 {ev['round']} 轮结束")

        elif etype == "final_report":
            st.session_state["final_report"] = ev["text"]
            elapsed = time.time() - t0
            status_slot.success(f"✅ 辩论完成（用时 {elapsed:.0f} 秒）")
            st.divider()
            st.markdown("## 📝 最终投资决策报告")
            st.markdown(ev["text"])

        elif etype == "error":
            status_slot.error(f"❌ 出错：{ev['message']}")
            st.session_state["stage"] = "idle"
            return

    # Debate finished successfully → persist it
    try:
        debate_id = history_db.save_debate(
            query=query,
            final_report=st.session_state["final_report"],
            transcript={
                "market_data": st.session_state["market_data"],
                "rounds": st.session_state["rounds"],
            },
            model=os.getenv("LLM_MODEL") or "",
        )
        st.session_state["debate_id"] = debate_id
    except Exception as e:  # noqa: BLE001
        st.warning(f"辩论已完成，但历史记录保存失败: {e}")

    st.session_state["stage"] = "done"


# =====================================================================
#  Follow-up chat (D1-1 — single unified assistant)
# =====================================================================
_FOLLOWUP_SYS_PROMPT = """你是一名专业投资顾问，刚刚主持了一场由五位投资大师\
（巴菲特、达利欧、马克斯、格林布拉特、林奇）参与的圆桌辩论。
下面是完整的辩论产出物：

========= 用户原始问题 =========
{query}

========= 市场数据 =========
{market_data}

========= 最终决策报告 =========
{final_report}

现在用户可能会基于这场辩论继续追问。你的职责：
1. 忠实转述大师们在辩论中的观点，必要时明确指出是哪位大师的看法
2. 回答要简洁、专业，避免空泛套话
3. 若用户的追问触及辩论未覆盖的新维度，应坦诚说明并给出基于大师方法论的推断
4. 用中文回答
"""


def _build_followup_messages(user_msg: str):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    sys_prompt = _FOLLOWUP_SYS_PROMPT.format(
        query=st.session_state["query"],
        market_data=st.session_state["market_data"][:4000],
        final_report=st.session_state["final_report"],
    )
    messages = [SystemMessage(content=sys_prompt)]
    for m in st.session_state["followup"][-10:]:  # last 10 turns
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=user_msg))
    return messages


def _stream_followup(user_msg: str):
    """Yield answer chunks so the UI shows tokens as they arrive.

    Uses the main model (gemini-2.5-pro) for the deepest reasoning. Its
    'thinking' phase produces no content for ~10-20s, so the caller shows
    a dynamic 'thinking' indicator until the first chunk appears here."""
    messages = _build_followup_messages(user_msg)
    # Follow-ups use the SAME high-quality reasoning model as the debate
    # (LLM_MODEL, e.g. gemini-2.5-pro) — not the fast model.
    try:
        llm = get_chat_llm(temperature=0.4)
    except Exception as e:  # noqa: BLE001
        yield f"⚠️ 追问初始化失败：{e}"
        return

    got_any = False
    try:
        for chunk in llm.stream(messages):
            piece = getattr(chunk, "content", "") or ""
            if piece:
                got_any = True
                yield piece
    except Exception as e:  # noqa: BLE001
        # Streaming failed → try a one-shot invoke as fallback
        try:
            resp = llm.invoke(messages)
            text = getattr(resp, "content", "") or ""
            if text:
                yield text
                return
        except Exception as e2:  # noqa: BLE001
            yield f"⚠️ 追问回答失败：{e2}"
            return
        yield f"⚠️ 追问流式输出中断：{e}"
        return

    if not got_any:
        yield "⚠️ 模型返回了空内容，请换个说法再试一次（可能触发了内容过滤或额度限制）。"


def _render_streaming_answer(user_msg: str) -> str:
    """Show a dynamic '思考中' indicator (st.status spinner animates
    client-side, so it stays alive during 2.5 Pro's ~10-20s content-less
    thinking phase), then stream the answer token-by-token with a typing
    cursor. Returns the full accumulated text."""
    import time as _time

    answer_slot = st.empty()
    acc = ""
    first_token = False
    t0 = _time.time()

    with st.status("🧠 资深AI研究员正在输出回答中....", expanded=False) as status:
        for piece in _stream_followup(user_msg):
            if not first_token:
                first_token = True
                elapsed = _time.time() - t0
                status.update(
                    label=f"✅ 资深AI研究员正在输出回答中....（{elapsed:.0f}s）",
                    state="running",
                )
            acc += piece
            answer_slot.markdown(acc + " ▌")  # typing cursor

        if acc and first_token:
            status.update(label="✅ 回答完成", state="complete")
        else:
            status.update(label="⚠️ 未获得回复", state="error")

    # Final render without the cursor
    answer_slot.markdown(acc if acc else "⚠️ 未获得回复，请重试。")
    return acc


def _render_followup_section() -> None:
    st.divider()
    st.markdown("## 💬 继续追问")
    st.caption("基于上述辩论和研究报告，您可以继续提问。系统会带上完整上下文回答。")

    for msg in st.session_state["followup"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("输入您的追问……（例如：如果PE回到历史中位数再考虑，建议会变吗？）")
    if user_input:
        st.session_state["followup"].append({"role": "user", "content": user_input})
        if st.session_state.get("debate_id"):
            history_db.append_followup(
                st.session_state["debate_id"], "user", user_input
            )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            answer = _render_streaming_answer(user_input)

        st.session_state["followup"].append({"role": "assistant", "content": answer})
        if st.session_state.get("debate_id"):
            history_db.append_followup(
                st.session_state["debate_id"], "assistant", answer
            )


# =====================================================================
#  Main page layout
# =====================================================================
_render_sidebar()

st.title("🏛️ AI 投资决策器")
st.caption(
    "五位投资大师（巴菲特 · 达利欧 · 马克斯 · 格林布拉特 · 林奇）基于 RAG 知识库 "
    "+ 实时 Web 数据，就您的投资问题展开辩论，最终给出综合建议。"
)

_missing = _missing_cloud_config()
if _missing:
    st.error(
        "⚙️ **应用尚未配置完成**：请到 Streamlit Cloud → 该 App 的 "
        "**Settings → Secrets** 粘贴密钥后点 Save，再 **Reboot**。\n\n"
        "缺少：" + "、".join(_missing) + "\n\n"
        "完整 TOML 模板见仓库 [docs/DEPLOY_STREAMLIT.md]"
        "(https://github.com/ShuoSh95/multiagent-investment-debate/blob/main/docs/DEPLOY_STREAMLIT.md)。"
    )
elif is_demo_mode() and not _kb_ok:
    st.warning(
        "📚 知识库尚未就绪（HF_TOKEN 无效或网络拉取失败）。"
        "侧边栏精选回放仍可围观；实时辩论的原著引用可能为空。"
    )

if is_demo_mode() and not _missing:
    st.info(
        "🎪 **公开 Demo**：为控制成本，Demo 使用轻量模型、最多 4 轮辩论、"
        f"每日限 {rate_limit.daily_limit()} 场、每位访客限 {rate_limit.session_limit()} 场。"
        "额度用完时，欢迎在侧边栏围观精选辩论回放。"
        "想不限量地使用完整版（深度推理模型 / 6 轮辩论），请到 "
        "[GitHub](https://github.com/ShuoSh95/multiagent-investment-debate) 本地部署。"
    )

stage = st.session_state["stage"]

if stage == "idle":
    with st.form("query_form", clear_on_submit=False):
        q = st.text_area(
            "请输入您的投资问题",
            placeholder="例如：最近美伊战争迎来和平转机，是不是可以开始大笔买入A股的大盘指数基金了？",
            height=120,
        )
        col1, col2 = st.columns([1, 5])
        with col1:
            submit = st.form_submit_button("🚀 开始辩论", type="primary", use_container_width=True)
        with col2:
            if is_demo_mode():
                st.caption("⏱️ Demo 档位：轻量模型 · 最多 4 轮 · 一场约 3–6 分钟")
            else:
                st.caption(
                    "⏱️ 一场辩论通常 5–15 分钟（Gemini 2.5 Pro 推理较慢但更深入）"
                )

    if submit and q.strip():
        if _missing_cloud_config():
            st.error("请先在 Streamlit Cloud Settings → Secrets 配置好密钥，再开始辩论。")
        else:
            ok, reason = (True, "")
            if is_demo_mode():
                ok, reason = rate_limit.check_quota(
                    st.session_state.get("debates_started", 0)
                )
            if not ok:
                st.warning(reason)
            else:
                # Transition to running state and rerun. The running branch
                # below will actually execute the debate so the form is hidden.
                st.session_state.update(
                    stage="running",
                    query=q.strip(),
                    market_data="",
                    rounds=[],
                    final_report="",
                    followup=[],
                    debate_id=None,
                    loaded_from_history=False,
                )
                st.rerun()

elif stage == "running":
    st.markdown(f"### 📋 问题\n> {st.session_state['query']}")
    if is_demo_mode() and not rate_limit.try_acquire_slot():
        st.warning(
            "⏳ 当前有另一场辩论正在进行（公共资源同一时刻只跑一场，"
            "避免触发 API 限速）。请几分钟后重试，或先在侧边栏围观精选回放。"
        )
        st.session_state["stage"] = "idle"
        if st.button("← 返回"):
            st.rerun()
    else:
        st.session_state["debates_started"] = (
            st.session_state.get("debates_started", 0) + 1
        )
        try:
            _stream_debate(st.session_state["query"])
        finally:
            if is_demo_mode():
                rate_limit.release_slot()
        # _stream_debate sets stage to "done" when successful (or back to
        # "idle" on error); rerun to show the clean final layout + followup
        st.rerun()

elif stage == "done":
    st.markdown(f"### 📋 问题\n> {st.session_state['query']}")
    if st.session_state["loaded_from_history"]:
        st.caption(f"🕐 历史辩论 #{st.session_state['debate_id']}（只读回放）")
    _render_all_past_rounds()
    _render_followup_section()

st.divider()
st.caption(
    "⚠️ 本项目为多 Agent 技术的研究与娱乐演示。「大师」均为基于公开资料的 AI 模拟，"
    "不代表真实人物观点；全部输出由大语言模型生成，可能存在错误或幻觉，"
    "**不构成任何投资建议**。股市有风险，据此操作盈亏自负。"
)
