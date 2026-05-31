"""
AI 投资决策器 · 多大师辩论系统

核心流程：
  1. 用户提出投资问题
  2. DataCollection 节点通过 Web Search 获取实时市场数据
  3. 5 位大师 (巴菲特 / 达利欧 / 马克斯 / 格林布拉特 / 林奇) 并行辩论
     - 每次发言前都会基于当前讨论焦点重新做 RAG 检索
     - 每轮后由主持人生成交叉质疑 (CrossQuestion)
     - 最多 6 轮；全员投完票 OR 连续 2 轮意见一致即提前收敛
  4. Researcher 节点汇总产出最终投资建议报告
"""

from __future__ import annotations

import os
import random
import re
from typing import Annotated, Dict, List, Optional, TypedDict
from operator import add

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

load_dotenv()

from llm_provider import current_provider_summary, get_chat_llm


# ============================================================
#  State
# ============================================================

def merge_votes(existing: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    return {**existing, **new}


def merge_tendencies(
    existing: Dict[int, Dict[str, str]],
    new: Dict[int, Dict[str, str]],
) -> Dict[int, Dict[str, str]]:
    """Merge per-round per-master tendencies. Keys are round numbers."""
    merged = {k: dict(v) for k, v in existing.items()}
    for rnd, inner in new.items():
        if rnd not in merged:
            merged[rnd] = {}
        merged[rnd].update(inner)
    return merged


class AgentState(TypedDict):
    query: str
    market_data: str
    debate_history: Annotated[List[str], add]
    round_count: int
    votes: Annotated[Dict[str, str], merge_votes]
    # tendency[round][master_name] -> "看多" / "看空" / "观望"
    tendencies: Annotated[Dict[int, Dict[str, str]], merge_tendencies]
    early_stop: bool
    round_order: Dict[int, List[str]]
    final_report: str
    # Append-only log of on-demand data retrievals triggered by masters.
    # Each item: {"master","round","query","found"(bool),"source"}
    search_log: Annotated[List[dict], add]


# ============================================================
#  Output parsing — vote / tendency / skip
# ============================================================

VOTE_PATTERNS = [
    re.compile(r"[【\[]\s*投票\s*[:：]\s*(看多|看空|观望)\s*[】\]]"),
    re.compile(r"投票\s*[:：]\s*(看多|看空|观望)"),
    re.compile(r"\*\*投票\*\*\s*[:：]?\s*(看多|看空|观望)"),
]

TENDENCY_PATTERNS = [
    re.compile(r"[【\[]\s*本轮倾向\s*[:：]\s*(看多|看空|观望)\s*[】\]]"),
    re.compile(r"本轮倾向\s*[:：]\s*(看多|看空|观望)"),
]

SKIP_PATTERN = re.compile(r"[【\[]\s*跳过\s*[】\]]|本轮不发言|本轮跳过")


def extract_vote(text: str) -> Optional[str]:
    for pat in VOTE_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


def extract_tendency(text: str) -> Optional[str]:
    """Extract soft per-round tendency. Falls back to vote if present."""
    for pat in TENDENCY_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    # If the master went straight to a hard vote without an explicit
    # tendency line, treat the vote as this round's tendency too.
    return extract_vote(text)


def is_skip(text: str) -> bool:
    return bool(SKIP_PATTERN.search(text))


# ============================================================
#  Master persona configs
# ============================================================

MASTER_PERSONAS = {
    "沃伦·巴菲特": {
        "rag_key": "buffett",
        "school": "value",  # for stratified shuffle
        "persona": (
            "你的投资哲学建立在价值投资之上——买股票就是买企业的一部分。"
            "你只看重自由现金流、高且稳定的ROE，以及极深的商业护城河"
            "（品牌优势、转换成本、网络效应、成本优势）。"
            "你对短期宏观经济预测毫无兴趣，也极度厌恶没有利润、仅靠概念炒作的公司。"
        ),
        "thinking_framework": (
            "1. 业务理解：这家公司靠什么赚钱？产品我能看懂吗？\n"
            "2. 护城河：品牌优势、转换成本、网络效应还是成本优势？\n"
            "3. 财务健康：自由现金流和ROE表现如何？\n"
            "4. 估值：当前价格是否具有安全边际？"
        ),
        "debate_bottom_line": (
            "如果其他人用宏观向好来劝你买入高PE且无护城河的股票，你必须坚决反驳。"
            "宁可错过，也不做错。但标的若符合你的标准，即使宏观差，"
            "你也敢于在别人恐惧时贪婪。"
        ),
        "emoji": "🎩",
    },
    "瑞·达利欧": {
        "rag_key": "dalio",
        "school": "macro",
        "persona": (
            "你从宏观经济周期出发，关注通胀、利率、债务周期和全球流动性。"
            "核心理念是'全天候策略'和'经济机器'模型，自上而下审视世界，"
            "根据宏观环境来决定大类资产的仓位配置。"
        ),
        "thinking_framework": (
            "1. 经济周期定位：当前处于通胀/增长的哪个象限？\n"
            "2. 流动性环境：央行政策宽松还是紧缩？利率走向？\n"
            "3. 风险平价：该标的在当前周期中应高配还是低配？\n"
            "4. 债务周期：是否存在去杠杆或债务危机前兆？"
        ),
        "debate_bottom_line": (
            "即使一家公司财报好，若宏观环境恶化（利率上行、流动性收紧），"
            "你也必须警告系统性风险。不会因个股基本面好就忽略宏观逆风。"
        ),
        "emoji": "🌐",
    },
    "霍华德·马克斯": {
        "rag_key": "marks",
        "school": "cycle",
        "persona": (
            "你极其敏锐地关注市场周期和风险定价，是坚定的逆向投资者。"
            "核心理念是'市场钟摆'和'第二层思维'。"
            "当别人因好消息狂热时你警告风险，当市场恐慌时你寻找便宜货。"
        ),
        "thinking_framework": (
            "1. 市场温度：当前情绪处于贪婪端还是恐惧端？\n"
            "2. 风险收益比：当前价格是否充分反映了乐观预期？\n"
            "3. 第二层思维：大多数人怎么看？他们错在哪里？\n"
            "4. 安全边际：最坏情况下，下行空间多大？"
        ),
        "debate_bottom_line": (
            "即使基本面强劲、宏观顺风，若市场情绪过度乐观、"
            "估值已反映所有好消息，你必须提醒'钟摆终将回摆'。"
        ),
        "emoji": "⚖️",
    },
    "乔尔·格林布拉特": {
        "rag_key": "greenblatt",
        "school": "quant",
        "persona": (
            "你是纯粹的量化价值投资者，坚定信仰'神奇公式'。"
            "只看两个核心指标：资本回报率（ROIC）和盈利收益率（Earning Yield）。"
            "讨厌讲故事，只看数据是否能跑赢市场平均。"
        ),
        "thinking_framework": (
            "1. ROIC：公司资本回报率在行业中的水平？\n"
            "2. Earning Yield：当前价格对应的盈利收益率够高吗？\n"
            "3. 排名对比：与同行业其他公司的综合排名？\n"
            "4. 均值回归：异常高/低的指标是否可持续？"
        ),
        "debate_bottom_line": (
            "不关心宏观经济，不关心市场情绪，只看冷冰冰的财务数据。"
            "数据说便宜且优质，即使所有人都在恐慌，你也坚持买入。"
        ),
        "emoji": "📊",
    },
    "彼得·林奇": {
        "rag_key": "lynch",
        "school": "growth",
        "persona": (
            "你信仰'常识投资'和GARP（合理价格增长）。"
            "偏好在日常生活中看到产品热销、PEG较低的公司。"
            "善于从消费者情绪和产品潜力的角度发现Tenbagger。"
        ),
        "thinking_framework": (
            "1. 生活观察：产品/服务在日常生活中受欢迎吗？\n"
            "2. PEG估值：市盈率÷盈利增长率是否小于1？\n"
            "3. 增长故事：驱动增长的核心逻辑是什么？可持续吗？\n"
            "4. 分类：慢速增长、稳定增长、快速增长、周期、困境反转还是隐蔽资产型？"
        ),
        "debate_bottom_line": (
            "相信普通人的生活经验可以发现好公司。即使宏观分析师和量化模型不看好，"
            "如果你在商场里看到产品卖疯了、PEG合理，你也敢于坚持。"
        ),
        "emoji": "🛒",
    },
}

MASTER_NAMES = list(MASTER_PERSONAS.keys())
MAX_ROUNDS = 6


# ============================================================
#  Stratified shuffle — one master per school, order randomized
# ============================================================

def stratified_shuffle(masters: List[str], seed: Optional[int] = None) -> List[str]:
    """Since every school currently has exactly one master in this system,
    a stratified shuffle degenerates to a normal shuffle. This helper is
    kept as the seam for a future multi-master-per-school scenario where
    it will group-then-interleave to avoid clumping one school's voices."""
    rng = random.Random(seed)
    school_groups: Dict[str, List[str]] = {}
    for name in masters:
        school = MASTER_PERSONAS[name]["school"]
        school_groups.setdefault(school, []).append(name)
    # Shuffle within each school
    for group in school_groups.values():
        rng.shuffle(group)
    # Interleave schools (round-robin from a shuffled school order)
    ordered_schools = list(school_groups.keys())
    rng.shuffle(ordered_schools)
    result: List[str] = []
    while any(school_groups[s] for s in ordered_schools):
        for s in ordered_schools:
            if school_groups[s]:
                result.append(school_groups[s].pop(0))
    return result


# ============================================================
#  Market data — real-time web search
# ============================================================

def collect_market_data(state: AgentState) -> dict:
    from web_search import perform_web_search

    # Fresh debate → clear the per-debate on-demand search cache.
    reset_search_cache()

    # Pre-warm the BGE-M3 embedding model BEFORE the parallel master nodes
    # run, otherwise 5 masters race to do the first load and trigger a
    # PyTorch meta-tensor error on MPS.
    try:
        from rag.embeddings import warmup_local_model
        print("  🔥 预加载 BGE-M3 嵌入模型 (首次约 10-20s) ...")
        warmup_local_model()
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  BGE-M3 预加载失败: {e}")

    print("  🔎 正在进行实时 Web 检索 ...")
    text, source = perform_web_search(
        query=state["query"],
        focus="最新财务数据、估值水平、近期宏观与行业事件",
    )
    if text:
        header = f"【实时市场数据 · 来源: {source}】\n"
        market_data = header + text
    else:
        market_data = (
            "【市场数据】⚠️ 未能获取到实时数据（请在 .env 配置 "
            "GOOGLE_API_KEY 或 TAVILY_API_KEY），本次辩论仅基于大师知识库进行。"
        )

    # Pre-plan the stratified-random "display order" for all rounds.
    # Parallel mode still executes all masters simultaneously, but this
    # order is used for UI display to avoid "always Buffett first" bias.
    round_order: Dict[int, List[str]] = {
        r: stratified_shuffle(MASTER_NAMES, seed=os.urandom(2).hex() and None)
        for r in range(1, MAX_ROUNDS + 1)
    }

    return {
        "market_data": market_data,
        "round_count": 0,
        "round_order": round_order,
        "tendencies": {},
        "early_stop": False,
    }


# ============================================================
#  RAG helper — retrieve per-turn with dynamic query
# ============================================================

def _build_rag_query(
    master_name: str,
    base_query: str,
    debate_history: List[str],
    current_round: int,
) -> str:
    """Build a retrieval query that evolves with the debate focus.

    Round 1: just the user's original question (the retriever will
             self-rewrite using its own self-query layer).
    Round 2+: user question + a short focus snippet built from the
             most recent entries so retrieval tracks the live topic.
    """
    if current_round == 1 or not debate_history:
        return base_query

    # Take last few entries (skip own entries when possible)
    recent = debate_history[-6:]
    focus_lines = []
    for entry in recent:
        # Entries look like "【<name> (第N轮)】: <content>"
        if master_name and master_name in entry:
            continue
        # Keep first 120 chars of each to form a compact focus
        snippet = entry.replace("\n", " ")
        focus_lines.append(snippet[:160])
    focus = " | ".join(focus_lines[-3:])
    if not focus:
        return base_query
    return f"{base_query}\n讨论焦点: {focus}"


def _try_rag_retrieve(rag_key: str, query: str, use_self_query: bool) -> str:
    """Attempt RAG retrieval; return empty string if KB not built yet."""
    try:
        from rag.retriever import format_retrieved_context, retrieve
        from rag.vectorstore import collection_count

        if collection_count(rag_key) == 0:
            return ""
        docs = retrieve(rag_key, query, top_k=5, use_self_query=use_self_query)
        return format_retrieved_context(docs)
    except Exception as e:
        print(f"  [rag] retrieval failed for {rag_key}: {e}")
        return ""


# ============================================================
#  On-demand web search per master (Issue #1)
#  Each master self-assesses whether it needs extra data this round,
#  based on its OWN analysis framework. Only searches when genuinely
#  needed. Results are cached within a debate to avoid duplicate calls.
# ============================================================

import threading as _threading

_search_cache: Dict[str, tuple] = {}   # normalized_query -> (text, source)
_search_cache_lock = _threading.Lock()


def reset_search_cache() -> None:
    """Clear the within-debate search cache (called at debate start)."""
    with _search_cache_lock:
        _search_cache.clear()


def _agent_search_enabled() -> bool:
    return os.getenv("ENABLE_AGENT_SEARCH", "1").strip() not in ("0", "false", "False", "")


def _get_fast_llm(temperature: float = 0.0):
    """Cheap/fast model for the lightweight 'do I need data?' decision.
    Falls back to the main LLM if LLM_FAST_MODEL is not configured."""
    fast = os.getenv("LLM_FAST_MODEL")
    if fast:
        return get_chat_llm(temperature=temperature, model=fast)
    return get_chat_llm(temperature=temperature)


_NEED_PATTERN = re.compile(r"检索\s*[:：]\s*(.+)")


def _assess_data_need(
    master_name: str,
    config: dict,
    state: AgentState,
    current_round: int,
) -> Optional[str]:
    """Ask the master (via a cheap model) whether it needs to fetch extra
    data this round. Returns a concrete search query, or None."""
    if not _agent_search_enabled():
        return None
    try:
        llm = _get_fast_llm(temperature=0.0)
    except Exception:
        return None

    recent = "\n".join(state.get("debate_history", [])[-4:]) or "（暂无）"
    sys_prompt = (
        f"你是{master_name}。{config['persona']}\n"
        f"你的分析框架：\n{config['thinking_framework']}\n\n"
        "任务：判断基于你的分析框架，针对当前投资问题和辩论进展，你是否还"
        "缺少【现有资料里没有的】具体数据来支撑你这一轮的论证。\n"
        "规则：\n"
        "- 现有资料已足够 → 只回复：[无需检索]\n"
        "- 确实缺关键数据 → 回复：[检索: 一句话写清要查的具体数据]\n"
        "  （查询要具体：含公司/标的名、指标名、时间范围等，便于搜索）\n"
        "- 严格只输出这一行，不要任何解释。不要为了检索而检索。"
    )
    user_prompt = (
        f"【投资问题】{state['query']}\n\n"
        f"【已有市场数据(节选)】\n{(state.get('market_data') or '')[:1200]}\n\n"
        f"【近期辩论】\n{recent}\n\n"
        f"当前第{current_round}轮。按你的框架，你还需要额外检索数据吗？"
    )
    try:
        resp = llm.invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
        ).content or ""
    except Exception as e:  # noqa: BLE001
        print(f"  [agent_search] {master_name} 需求评估失败: {e}")
        return None

    if "无需检索" in resp:
        return None
    m = _NEED_PATTERN.search(resp)
    if m:
        q = m.group(1).strip().strip("】]").strip()
        return q or None
    return None


def _agent_web_search(query: str, focus: str) -> tuple:
    """Run a web search with within-debate caching. Returns (text, source)."""
    key = query.strip().lower()
    with _search_cache_lock:
        if key in _search_cache:
            return _search_cache[key]
    try:
        from web_search import perform_web_search
        text, source = perform_web_search(query=query, focus=focus)
    except Exception as e:  # noqa: BLE001
        print(f"  [agent_search] search failed: {e}")
        text, source = "", "none"
    with _search_cache_lock:
        _search_cache[key] = (text, source)
    return text, source


# ============================================================
#  Master node factory
# ============================================================

def create_master_node(master_name: str, config: dict):
    def master_node(state: AgentState) -> dict:
        current_round = state["round_count"] + 1
        already_voted = master_name in state.get("votes", {})

        llm = get_chat_llm(temperature=0.7)

        # RAG every round — self-query LLM-rewrite only on first round
        rag_key = config.get("rag_key", "")
        rag_context = ""
        if rag_key:
            rag_query = _build_rag_query(
                master_name,
                state["query"],
                state.get("debate_history", []),
                current_round,
            )
            rag_context = _try_rag_retrieve(
                rag_key,
                rag_query,
                use_self_query=(current_round == 1),
            )

        # On-demand web search: the master decides (per its framework)
        # whether it needs extra data this round.
        search_section = ""
        search_log_entry: Optional[dict] = None
        wanted_query = _assess_data_need(master_name, config, state, current_round)
        if wanted_query:
            focus = f"{master_name} 的分析视角：{config['persona'][:60]}"
            print(f"  🔍 {master_name} 主动检索: {wanted_query}")
            text, source = _agent_web_search(wanted_query, focus)
            found = bool(text and source != "none")
            search_log_entry = {
                "master": master_name,
                "round": current_round,
                "query": wanted_query,
                "found": found,
                "source": source,
            }
            if found:
                search_section = (
                    f"\n\n【你本轮主动检索到的补充数据 · 来源:{source}】\n"
                    f"（检索请求：{wanted_query}）\n{text}"
                )
            else:
                search_section = (
                    f"\n\n【你本轮请求检索「{wanted_query}」，但未获取到可靠数据。】\n"
                    "你必须在发言中如实说明该数据缺失、无法据此判断，"
                    "绝对不得编造任何数字或事实。"
                )

        sys_parts = [
            f"你是{master_name}。{config['persona']}",
            f"\n【思考框架】\n{config['thinking_framework']}",
            f"\n【辩论底线】\n{config['debate_bottom_line']}",
            "\n【行为规则】",
            "- 严格按照你的投资流派发言，不要偏离。",
            "- 结合【实时市场数据】做客观判断，参考或反驳其他人的发言。",
            "- 回答简明扼要，控制在 200 字以内。",
            "- 发言末尾必须附上本轮倾向：[本轮倾向: 看多] / [本轮倾向: 看空] / [本轮倾向: 观望]。",
            # Data integrity — applies to ALL masters, every round.
            "- 【数据诚信铁律】绝对禁止编造、猜测或杜撰任何数据、数字、财报、"
            "估值或事实。你只能引用【实时市场数据】【知识库参考】"
            "【你本轮主动检索到的补充数据】中真实出现的信息。"
            "若你需要的关键数据缺失或检索失败，必须如实说明"
            "（例如『缺乏该公司最新ROIC数据，无法据此判断』），不得编造。",
        ]

        if rag_context:
            sys_parts.append(
                "- 你必须基于【知识库参考】中的原著内容来支撑你的论点。"
                "在关键论断处用一句话引用原著精神或原文（不必带完整出处）。"
            )

        if current_round == 1:
            sys_parts.append("- 这是第一轮讨论，请务必发表你的初始观点。")
        else:
            sys_parts.append(
                "- 你可以选择本轮不发言。如果你的立场已充分表达、"
                "或本轮没有需要你回应的新观点，直接回复 [跳过] 即可。"
                "不必每轮都发言，只在有新价值时才说话。"
            )

        if already_voted:
            sys_parts.append("- 你已经投过最终票了，无需再次投票（但本轮倾向仍需附上）。")
        else:
            sys_parts.append(
                f"- 若信息已足够让你做最终决定，可在发言末尾附上硬投票："
                f"[投票: 看多] / [投票: 看空] / [投票: 观望]。"
                f"最迟第 {MAX_ROUNDS} 轮必须投票。当前第 {current_round} 轮。"
            )

        sys_prompt = "\n".join(sys_parts)

        history_str = "\n".join(state.get("debate_history", [])) or "（暂无）"
        rag_section = (
            f"\n\n【知识库参考 - 来自你的原著/演讲】\n{rag_context}"
            if rag_context
            else ""
        )
        user_prompt = (
            f"【投资问题】{state['query']}\n\n"
            f"{state['market_data']}{rag_section}{search_section}\n\n"
            f"【辩论历史】\n{history_str}\n\n"
            f"当前是第 {current_round} 轮辩论（最多 {MAX_ROUNDS} 轮），请发表你的看法："
        )

        messages = [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
        response = llm.invoke(messages).content

        search_log_out = [search_log_entry] if search_log_entry else []

        if is_skip(response):
            return {
                "debate_history": [],
                "votes": {},
                "tendencies": {},
                "search_log": search_log_out,
            }

        entry = f"【{master_name} (第{current_round}轮)】: {response}"

        vote = extract_vote(response)
        tendency = extract_tendency(response)

        new_votes: Dict[str, str] = {}
        if vote and not already_voted:
            new_votes[master_name] = vote
        elif current_round >= MAX_ROUNDS and not already_voted:
            new_votes[master_name] = "未明确"

        new_tendencies: Dict[int, Dict[str, str]] = {}
        if tendency:
            new_tendencies = {current_round: {master_name: tendency}}

        return {
            "debate_history": [entry],
            "votes": new_votes,
            "tendencies": new_tendencies,
            "search_log": search_log_out,
        }

    return master_node


# ============================================================
#  Cross-questioning node
# ============================================================

def cross_question(state: AgentState) -> dict:
    current_round = state["round_count"] + 1
    round_tag = f"第{current_round}轮"
    current_entries = [e for e in state["debate_history"] if f"({round_tag})" in e]
    if len(current_entries) < 2:
        return {"debate_history": []}

    llm = get_chat_llm(temperature=0.7)
    sys_prompt = (
        "你是一位投资辩论主持人。阅读本轮各位投资大师的发言，"
        "找出最突出的 1-2 对观点冲突，以对应大师的口吻生成简短的交叉质疑。\n"
        "每条质疑不超过 60 字，严格使用以下格式，每条一行：\n"
        "  [大师A → 大师B]: 质疑内容\n"
        "如果本轮没有明显冲突或发言太少，仅输出一个字：无。"
    )
    entries_str = "\n".join(current_entries)
    user_prompt = f"以下是{round_tag}的大师发言：\n\n{entries_str}\n\n请生成交叉质疑："

    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages).content.strip()

    if response == "无" or len(response) < 5:
        return {"debate_history": []}
    return {"debate_history": [f"【交叉质疑 ({round_tag})】\n{response}"]}


# ============================================================
#  Round management & convergence
# ============================================================

def update_round(state: AgentState) -> dict:
    return {"round_count": state["round_count"] + 1}


def _round_is_unanimous(tendencies_for_round: Dict[str, str]) -> Optional[str]:
    """Return the single tendency if all speakers agree (not 观望), else None.

    Requires at least 3 speakers this round for the signal to be meaningful.
    """
    if len(tendencies_for_round) < 3:
        return None
    values = set(tendencies_for_round.values())
    if len(values) != 1:
        return None
    only = next(iter(values))
    if only == "观望":
        # Treat universal 观望 as non-consensus for early-stop purposes
        return None
    return only


def should_continue(state: AgentState):
    """Return either 'Researcher' (end) or the list of master names to
    fan-out to for the next round. Returning a list directly makes this
    robust across LangGraph versions that no longer accept list values
    in a conditional_edges path_map."""
    total = len(MASTER_NAMES)
    votes_n = len(state.get("votes", {}))
    round_count = state["round_count"]

    # 1) Everyone hard-voted → stop
    if votes_n >= total:
        return "Researcher"

    # 2) Consecutive 2-round soft consensus → stop
    tendencies = state.get("tendencies", {})
    if round_count >= 2:
        last = tendencies.get(round_count, {})
        prev = tendencies.get(round_count - 1, {})
        last_agree = _round_is_unanimous(last)
        prev_agree = _round_is_unanimous(prev)
        if last_agree and prev_agree and last_agree == prev_agree:
            print(
                f"\n  🤝 连续 2 轮一致倾向 '{last_agree}' → 提前收敛辩论。"
            )
            return "Researcher"

    # 3) Hard cap
    if round_count < MAX_ROUNDS:
        return MASTER_NAMES  # fan-out to all masters for the next round
    return "Researcher"


# ============================================================
#  Researcher node
# ============================================================

def researcher_node(state: AgentState) -> dict:
    llm = get_chat_llm(
        temperature=0.2,
        model=os.getenv("LLM_RESEARCHER_MODEL") or None,
    )
    master_list_str = "、".join(MASTER_NAMES)
    sys_prompt = (
        "你是一位拥有顶级投行经验、兼具深厚政经法背景的资深宏观策略助理研究员。"
        f"你绝对中立客观。你的任务是根据 {len(MASTER_NAMES)} 位投资大师"
        f"（{master_list_str}）的辩论历史和最终投票，"
        "撰写一份结构化的投资建议报告。"
    )
    history_str = "\n".join(state["debate_history"])
    votes_str = "\n".join(f"  {k}: {v}" for k, v in state["votes"].items())

    user_prompt = (
        f"【用户问题】{state['query']}\n\n"
        f"【市场数据】\n{state['market_data']}\n\n"
        f"【大师辩论实录】\n{history_str}\n\n"
        f"【最终投票结果】\n{votes_str}\n\n"
        "请输出最终决策研报，包含以下模块：\n"
        "1. 客观现状概述\n"
        "2. 大师共识与分歧点提炼\n"
        "3. 投票结果解析\n"
        "4. 最终操作建议（含风险提示）"
    )
    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages).content
    return {"final_report": response}


# ============================================================
#  Graph
# ============================================================

workflow = StateGraph(AgentState)
workflow.add_node("DataCollection", collect_market_data)
for name in MASTER_NAMES:
    workflow.add_node(name, create_master_node(name, MASTER_PERSONAS[name]))
workflow.add_node("CrossQuestion", cross_question)
workflow.add_node("UpdateRound", update_round)
workflow.add_node("Researcher", researcher_node)

workflow.set_entry_point("DataCollection")
for name in MASTER_NAMES:
    workflow.add_edge("DataCollection", name)
    workflow.add_edge(name, "CrossQuestion")
workflow.add_edge("CrossQuestion", "UpdateRound")
workflow.add_conditional_edges("UpdateRound", should_continue)
workflow.add_edge("Researcher", END)

app = workflow.compile()


# ============================================================
#  Streaming terminal UI
# ============================================================

DIVIDER_HEAVY = "━" * 56
DIVIDER_LIGHT = "─" * 56


def format_vote_progress(votes: Dict[str, str]) -> str:
    if not votes:
        return "暂无投票"
    parts = [f"{n}[{v}]" for n, v in votes.items()]
    remaining = len(MASTER_NAMES) - len(votes)
    text = " ｜ ".join(parts)
    if remaining > 0:
        text += f"   （还有 {remaining} 位待投票）"
    else:
        text += "   ✅ 全员已投票"
    return text


def format_tendency_row(tendencies_for_round: Dict[str, str]) -> str:
    if not tendencies_for_round:
        return "（本轮无人发表倾向）"
    parts = []
    for name in MASTER_NAMES:
        if name in tendencies_for_round:
            parts.append(f"{name}:{tendencies_for_round[name]}")
    return " ｜ ".join(parts) if parts else "（本轮无人发表倾向）"


def run_streaming(query: str):
    inputs: AgentState = {
        "query": query,
        "market_data": "",
        "debate_history": [],
        "round_count": 0,
        "votes": {},
        "tendencies": {},
        "early_stop": False,
        "round_order": {},
        "final_report": "",
        "search_log": [],
    }

    accumulated_votes: Dict[str, str] = {}
    current_display_round = 0
    round_header_shown = False
    # Buffer parallel master events so we can render them in stratified-
    # random order instead of arrival order.
    round_buffer: List[tuple] = []
    round_order_plan: Dict[int, List[str]] = {}

    def _flush_round_buffer():
        if not round_buffer:
            return
        plan = round_order_plan.get(current_display_round, MASTER_NAMES)
        by_name = {name: payload for name, payload in round_buffer}
        for name in plan:
            if name in by_name:
                _render_master_turn(name, by_name[name])
        round_buffer.clear()

    def _render_master_turn(node_name: str, update: dict):
        emoji = MASTER_PERSONAS[node_name]["emoji"]
        new_entries = update.get("debate_history", [])
        new_votes = update.get("votes", {})
        search_log = update.get("search_log", []) or []
        accumulated_votes.update(new_votes)

        if not new_entries:
            print(f"\n  {emoji} {node_name}:  💤 本轮不发言")
            return
        raw = new_entries[0]
        content = raw.split("】: ", 1)[-1] if "】: " in raw else raw
        print(f"\n  {emoji} {node_name}:")
        for s in search_log:
            mark = "✅" if s.get("found") else "⚠️ 未获取到可靠数据"
            print(f"     🔍 主动检索: {s.get('query')}  {mark}")
        for line in content.strip().split("\n"):
            print(f"     {line}")
        if node_name in new_votes:
            print(f"     📌 最终投票: {new_votes[node_name]}")

    for event in app.stream(inputs, stream_mode="updates"):
        for node_name, update in event.items():

            if node_name == "DataCollection":
                current_display_round = 1
                round_order_plan = update.get("round_order", {}) or {}
                print(f"\n  📊 市场数据收集完成\n")
                for line in (update.get("market_data") or "").split("\n"):
                    print(f"  {line}")

            elif node_name in MASTER_NAMES:
                if not round_header_shown:
                    print(f"\n  ╔{'═' * 50}╗")
                    label = f" 第 {current_display_round} 轮辩论 "
                    print(f"  ║{label:═^50}║")
                    print(f"  ╚{'═' * 50}╝")
                    round_header_shown = True
                round_buffer.append((node_name, update))
                # Flush when we have collected all masters OR when the
                # round is effectively complete (LangGraph will emit
                # CrossQuestion only after all parallel masters finish,
                # so we rely on that below as a safety flush).

            elif node_name == "CrossQuestion":
                _flush_round_buffer()
                entries = update.get("debate_history", [])
                if entries:
                    print(f"\n  ┌── 💬 交叉质疑 {'─' * 36}┐")
                    for entry in entries:
                        content = entry.split("】\n", 1)[-1] if "】\n" in entry else entry
                        for line in content.strip().split("\n"):
                            print(f"  │  {line}")
                    print(f"  └{'─' * 52}┘")

            elif node_name == "UpdateRound":
                _flush_round_buffer()
                print(f"\n  {DIVIDER_LIGHT}")
                print(f"  📊 投票进度: {format_vote_progress(accumulated_votes)}")
                if current_display_round >= MAX_ROUNDS:
                    print(f"  ⏰ 已达最大轮次（{MAX_ROUNDS} 轮），辩论结束。")
                elif len(accumulated_votes) >= len(MASTER_NAMES):
                    print(f"  🏁 全员投票完毕，辩论提前收敛！")
                current_display_round += 1
                round_header_shown = False

            elif node_name == "Researcher":
                _flush_round_buffer()
                print(f"\n\n{DIVIDER_HEAVY}")
                print(f"{'📝 最终投资决策报告':^52}")
                print(f"{DIVIDER_HEAVY}\n")
                print(update.get("final_report", ""))
                print(f"\n{DIVIDER_HEAVY}")


if __name__ == "__main__":
    print(f"\n{DIVIDER_HEAVY}")
    print("    🏛️  AI 投资决策器 · 多大师辩论系统")
    print(DIVIDER_HEAVY)
    print(f"    Chat: {current_provider_summary()}")
    print(DIVIDER_HEAVY)

    query = input("\n  📋 请输入投资问题（回车使用默认）: ").strip()
    if not query:
        query = "考虑到目前的市场环境，现在适合重仓买入特斯拉(Tesla)股票吗？"
        print(f"     → 使用默认: {query}")

    try:
        run_streaming(query)
    except KeyboardInterrupt:
        print("\n\n  ⚠️  用户中断，辩论终止。")
    except Exception as e:
        print(f"\n  ❌ 执行失败: {e}")
        print(f"  当前 chat 配置: {current_provider_summary()}")
        print("  请确保 .env 正确配置 LLM_PROVIDER + 对应 API Key。")
