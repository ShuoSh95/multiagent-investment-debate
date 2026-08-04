# 🏛️ AI 投资决策器 · 多大师辩论系统

> 让**巴菲特、达利欧、马克斯、格林布拉特、林奇**五位投资大师在你的电脑里"同台辩论"，
> 结合实时 Web 数据 + 大师本人原著 RAG 检索，
> 最后由一位中立研究员 Agent 给出结构化投资建议。

---

> [!WARNING]
> ## ⚠️ 投资免责声明（务必阅读）
>
> **本项目仅用于人工智能、多 Agent 技术的学习与研究演示，不构成任何投资建议、要约或操作指引。**
>
> - 所有 Agent 的"大师"人设均为基于公开资料的 **AI 模拟**，**不代表**相关投资人本人的真实观点或立场。
> - 系统输出由大语言模型生成，**可能存在事实错误、数据过时或"幻觉"**，请勿作为投资决策依据。
> - 股市有风险，投资需谨慎。**任何人据本项目输出进行的投资操作，盈亏与后果由其本人完全自负**，作者不承担任何责任。
> - 本项目与文中提及的任何个人、公司、基金均无关联，亦未获其授权或背书。

---

## 目录

- [✨ 特性](#-特性)
- [🏗️ 系统架构](#-系统架构)
- [📂 目录结构](#-目录结构)
- [🚀 快速开始](#-快速开始)
- [⚙️ 配置说明 (.env)](#-配置说明-env)
- [🧠 LLM 与 Embedding Provider](#-llm-与-embedding-provider)
- [🔎 Web Search 机制](#-web-search-机制)
- [📚 RAG 知识库](#-rag-知识库)
- [💬 辩论流程详解](#-辩论流程详解)
- [🧪 测试与调试](#-测试与调试)
- [❓ 常见问题 (FAQ)](#-常见问题-faq)
- [📋 版权与致谢](#-版权与致谢)

---

## ✨ 特性

| 能力 | 说明 |
|---|---|
| 🖥️ **Web 网页界面** | Streamlit 网页：浏览器输入问题，全程围观大师辩论 + 流式研报，结束后可持续追问 |
| 🎭 **5 位大师 Agent** | 每位大师都有独立 Persona + 思考框架 + 辩论底线 prompt |
| 📚 **原著 RAG 检索** | 每次发言前都会基于当前讨论焦点重新检索大师自己的原著 / 演讲 / 股东信 |
| 🔍 **按需主动检索** | 每位大师每轮先按自身框架自评是否缺数据，**确有需要才**发起 Web 检索（达利欧查宏观、格林布拉特查 ROIC……） |
| 🛡️ **数据诚信铁律** | 禁止编造任何数据，检索失败时大师必须如实声明"数据缺失、无法判断" |
| 🔎 **实时 Web 数据** | 接入 Gemini 原生 Google Search Grounding（免费），Tavily 作为 fallback |
| 🤝 **真辩论，非独白** | 每轮后有"交叉质疑"微轮次，主持人找出观点冲突，促使大师互相回应 |
| 💬 **结论后追问** | 辩论结束后可基于完整研报继续对话，由资深研究员带上下文流式作答 |
| 🗂️ **历史存档** | 每场辩论自动存入本地 SQLite，侧边栏可随时回看 + 继续追问 |
| 🏁 **智能早停** | 全员投完票 **或** 连续 2 轮大师倾向一致 → 提前收敛，不浪费 token |
| 🎲 **分层随机出场** | 每轮大师出场顺序打乱，避免"巴菲特永远第一个说话"的视觉偏见 |
| 🔌 **多 LLM Provider** | `.env` 一行切换 DeepSeek / Gemini / OpenAI / Claude / 通义 / 智谱 / 豆包 |
| 🖥️ **本地 Embedding** | 默认 BGE-M3 跑在 Apple MPS，零 API 成本、完全离线、跨语言强 |

> 📌 模型分工（默认 Gemini）：辩论/研究员/追问用 **2.5 Pro**（深度推理），数据自评 + 网络检索用 **2.5 Flash**（快且省）。详见 [配置说明](#-配置说明-env)。

---

## 🏗️ 系统架构

```
┌────────────────┐
│ 用户输入投资问题 │
└────────┬───────┘
         │
         ▼
┌────────────────┐       ┌───────────────────────┐
│ DataCollection │──────▶│ web_search            │
│ (收集市场数据)  │       │ ├─ Gemini Grounding   │ (free)
└────────┬───────┘       │ └─ Tavily fallback    │
         │                └───────────────────────┘
         │ (fan-out to 5 masters in parallel)
         ▼
┌────────────────────────────────────────────────┐
│  Master Agents (每轮并行)                       │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐ ┌──────┐│
│  │ 巴菲特  │ │ 达利欧  │ │ 马克斯  │ │格林布拉特│ │ 林奇 ││
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬───┘ └──┬──┘│
│      │ 每次发言前都会:                          │
│      │  1. RAG 检索本人原著（BGE-M3 本地向量）    │
│      │  2. 读取市场数据 + 历史辩论                │
│      │  3. 输出观点 + [本轮倾向] + [可选投票]    │
└──────┴────────────────────────────────────────┘
         │
         ▼
┌────────────────┐
│ CrossQuestion  │ 主持人找出本轮最大观点冲突，
│ (交叉质疑)      │ 以冲突方口吻生成反问
└────────┬───────┘
         │
         ▼
┌────────────────┐      ┌──────────────────────┐
│  UpdateRound   │─────▶│ should_continue?     │
└────────────────┘      │ • 全员投完票 → end    │
                         │ • 连续 2 轮一致 → end │
                         │ • < 6 轮  → 下一轮   │
                         └──────┬───────────────┘
                                │
                       ┌────────┴─────────┐
                       ▼                  ▼
              (下一轮 5 大师)        ┌───────────┐
                                     │Researcher │ 中立研究员
                                     │ 总结辩论   │ 输出最终研报
                                     └─────┬─────┘
                                           ▼
                                    【最终投资决策报告】
```

---

## 📂 目录结构

```
InvestmentAgent/
├── main.py                  # 主入口：LangGraph 图编排 + 流式终端 UI + 按需检索
├── llm_provider.py          # 多 Provider 切换（DeepSeek / Gemini / Claude ...）
├── web_search.py            # 实时 Web 检索（Gemini Grounding / Tavily）
├── requirements.txt
├── .env.example             # 环境变量模板（复制为 .env 后填 Key）
├── .gitignore
├── README.md
├── ROADMAP.md               # 版本迭代规划（v2.0 方向：@指定大师、串行辩论、中途插话…）
│
├── web/                     # Streamlit 网页界面（v1.0）
│   ├── streamlit_app.py     # UI 主入口：辩论流式渲染 + 追问 + 历史侧边栏
│   ├── debate_runner.py     # 把 LangGraph stream 封装成结构化事件
│   └── history_db.py        # SQLite 历史辩论存储（本地）
│
├── rag/                     # RAG 检索系统
│   ├── config.py            # 各大师知识库源配置
│   ├── loader.py            # PDF/HTML/TXT/EPUB/MD 统一加载器
│   ├── chunker.py           # 语义切分 + 元数据增强
│   ├── embeddings.py        # BGE-M3 本地 / OpenAI 嵌入切换（含并发锁 + 预热）
│   ├── vectorstore.py       # ChromaDB 向量库封装
│   ├── retriever.py         # 混合检索（向量 + BM25 + RRF 融合）
│   └── build_kb.py          # CLI：构建 / 重建知识库
│
├── scrapers/                # 数据采集脚本
│   ├── github_sources.py    # 多 CDN fallback 的 GitHub 下载
│   └── official_sites.py    # 官方网站 / Wikipedia 等公开来源下载
│
├── scripts/                 # 测试 / 基准脚本
│   ├── smoke_e2e.py         # 端到端冒烟测试
│   ├── smoke_test.py        # LLM + RAG 接线测试
│   ├── test_gemini.py       # Gemini API 连通性测试
│   ├── test_rag_parallel.py # 并发 RAG 检索测试
│   └── bench_bge_m3.py      # BGE-M3 本地性能基准
│
├── architecture_design.md   # 架构设计文档
├── agent_capability_design.md
│
└── data/                    # ⚠️ 本地生成，已被 .gitignore 忽略（不进仓库）
    ├── raw/                 # 各大师原始资料（含受版权书籍，请自行采集）
    ├── chroma_db/           # ChromaDB 向量库（build_kb 生成）
    ├── bm25_index/          # BM25 索引（build_kb 生成）
    └── history.db           # 网页端辩论历史（SQLite）
```

> **关于 `data/`**：知识库（原始语料、向量库、BM25 索引）和辩论历史都**不在 Git 仓库里**——
> 原始语料含受版权材料，且体积近 300MB。Clone 本项目后请按 [快速开始](#-快速开始) 自行构建知识库。

---

## 🚀 快速开始

### 1. 环境准备

- macOS / Linux / Windows
- Python **3.9+**（推荐 3.10+，3.9 会有一些 google 库的弃用警告但不影响功能）
- ~5GB 磁盘空间（BGE-M3 模型 2.3GB + 知识库数据 ~80MB + Python 依赖 ~2GB）

### 2. 安装依赖

```bash
git clone <your-repo-url> InvestmentAgent
cd InvestmentAgent

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 如在国内网络，建议用清华镜像加速
pip install -r requirements.txt \
    --default-timeout=1800 --retries 10 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 3. 配置 .env

编辑项目根目录的 `.env`（见下方 [配置说明](#-配置说明-env)）。最小化配置示例：

```bash
LLM_PROVIDER=gemini
GOOGLE_API_KEY=你的_Gemini_key
EMBEDDING_PROVIDER=local
HF_ENDPOINT=https://hf-mirror.com
```

### 4. 构建知识库（首次必须，耗时约 15-25 分钟）

```bash
# 采集 + 切分 + 嵌入 + 索引（5 位大师全量）
python -m rag.build_kb --master all --rebuild

# 若已有 data/raw/*，只做向量化，不重新抓数据：
python -m rag.build_kb --master all --skip-acquire --rebuild

# 单独构建某位大师（常用于调试或补数据）：
python -m rag.build_kb --master buffett --skip-acquire
```

首次会触发 `BAAI/bge-m3` 从 HuggingFace Mirror 下载 ~2.3GB，之后会缓存在 `~/.cache/huggingface/`。

**预期产出：**

| 大师 | 约 Chunks |
|---|---|
| 沃伦·巴菲特 | 2,400 |
| 瑞·达利欧 | 6,500 |
| 霍华德·马克斯 | 3,700 |
| 乔尔·格林布拉特 | ~30 |
| 彼得·林奇 | ~30 |
| **合计** | **~12,700** |

> 格林布拉特和林奇的原始出版物版权限制较严，当前主要依赖 Wikipedia 条目，
> 后续可手动把你购买的电子书（PDF/EPUB）丢到 `data/raw/greenblatt/` 或
> `data/raw/lynch/` 下，然后 `--skip-acquire --rebuild` 即可重建。

### 5. 启动辩论

**方式 A · 网页界面（推荐）**

```bash
streamlit run web/streamlit_app.py
# 浏览器打开 http://localhost:8501
```

网页端能力：
- 输入框提交问题 → 市场数据卡片 → 5 位大师逐位流式发言（含主动检索标记）→ 交叉质疑 → 最终研报
- 大师发言带"本轮倾向"彩色标签 + "最终投票"徽章
- 辩论结束后底部出现**追问对话框**，可基于完整研报继续提问（资深研究员带上下文流式作答）
- 左侧栏列出**历史辩论**，点击任意一场回看 + 继续追问

**方式 B · 终端命令行**

```bash
python main.py
```

进入交互后输入你的投资问题，比如：

```
📋 请输入投资问题: 考虑到目前的市场环境，现在适合重仓买入特斯拉(Tesla)股票吗？
```

然后泡一杯咖啡，5-15 分钟后你会看到 5 位大师的激烈辩论 + 最终研报
（用 Gemini 2.5 Pro 推理较慢但更深入；想快可把 `LLM_MODEL` 换成 `gemini-2.5-flash`）。

---

## ⚙️ 配置说明 (.env)

完整 `.env` 示例：

> 完整模板见仓库里的 [`.env.example`](.env.example)，复制一份即可：`cp .env.example .env`

```bash
# ==================== Chat LLM ====================
LLM_PROVIDER=gemini             # deepseek | openai | anthropic | gemini | qwen | zhipu | doubao

# 只填对应 Provider 的 Key，其他注释掉
GOOGLE_API_KEY=你的_gemini_key       # Gemini
# DEEPSEEK_API_KEY=sk-xxxx
# OPENAI_API_KEY=sk-xxxx
# ANTHROPIC_API_KEY=sk-ant-xxxx
# DASHSCOPE_API_KEY=sk-xxxx            # 通义千问
# ZHIPU_API_KEY=xxxx.xxxx              # 智谱 GLM
# ARK_API_KEY=xxxx                     # 豆包 / 火山方舟

# ---- 模型分工（可选，留空则用各 Provider 默认）----
LLM_MODEL=gemini-2.5-pro             # 大师辩论 + 研究员总结 + 追问（深度推理）
LLM_FAST_MODEL=gemini-2.5-flash      # 大师"是否需要补充数据"自评（快/省）
WEB_SEARCH_MODEL=gemini-2.5-flash    # 网络检索用的模型
ENABLE_AGENT_SEARCH=1                # 大师按需主动检索：1=开 / 0=关

# ==================== Embedding ====================
EMBEDDING_PROVIDER=local          # local = BGE-M3 (推荐) / openai
HF_ENDPOINT=https://hf-mirror.com # 国内加速 HuggingFace

# ==================== Web Search fallback ====================
# Gemini 用户无需额外配置，会自动使用 Google Search Grounding（免费）
# 非 Gemini 用户可配置 Tavily 作为 web search:
# TAVILY_API_KEY=tvly-xxxx
```

---

## 🔎 Web Search 机制

**两层检索：**

1. **辩论开始前的公共检索** —— `DataCollection` 节点做一次 Web 搜索，获取用户问题的最新公开信息（价格、估值、财报、宏观事件等），塞进 `market_data`，供所有大师共享。

2. **大师按需主动检索（每轮）** —— 每位大师发言前，会用一个轻量模型（`LLM_FAST_MODEL`）按**自身分析框架**自评：是否还缺现有资料里没有的关键数据？
   - 缺 → 生成具体查询并发起检索（如达利欧查"GDP/CPI/央行利率/债务率"，格林布拉特查"ROIC/盈利收益率"）
   - 不缺 → 跳过，不浪费调用
   - 检索结果注入该大师本轮 prompt；**同一场辩论内相同查询会缓存去重**
   - 可用 `ENABLE_AGENT_SEARCH=0` 整体关闭

   > 🛡️ **数据诚信铁律**：所有大师都被强制约束——只能引用真实检索到/知识库里的数据，
   > 检索失败或数据缺失时必须如实声明（如"缺乏该公司最新 ROIC 数据，无法判断"），**绝不编造**。

**优先级链路：**

```
1. Gemini 原生 Google Search Grounding    ← 如果设置了 GOOGLE_API_KEY
   └─ 完全免费（算在 Gemini 免费额度内）
   └─ 3 次重试 + 多 model fallback（2.5-flash → 2.0-flash → 1.5-flash）

2. Tavily API                              ← 如果设置了 TAVILY_API_KEY
   └─ 免费 1000 次/月
   └─ https://tavily.com/

3. Graceful 降级                           ← 都没配
   └─ 提示用户未获取到实时数据，仅基于 RAG 辩论
```

**软性限流**：在 `web_search.py` 里定义了每日 200 次 Gemini / 每月 1000 次 Tavily 的 soft limit，到达后提醒并自动降级。

---

## 📚 RAG 知识库

### 数据来源（按可信度分层）

| Tier | 来源 | 示例 |
|---|---|---|
| **Tier 1 · 原著** | 大师本人公开出版物 / 官方演讲 / 股东信 | 巴菲特历年股东信、Marks Memos、Dalio *Principles* |
| **Tier 2 · 转述** | 高质量访谈、演讲文字稿 | Dalio 在 Harvard / Davos 的演讲 |
| **Tier 3 · 参考** | 维基百科条目 | 用于原著难以获取的大师（如 Greenblatt、Lynch） |

### 检索策略（hybrid retrieval）

对每次检索请求，执行：

1. **向量检索**（BGE-M3 embedding + Cosine 距离）取 Top-K
2. **BM25 关键词检索** 取 Top-K（捕捉专有名词）
3. **Reciprocal Rank Fusion (RRF)** 融合两路结果
4. **首轮**额外叠加一次 **Self-Query LLM rewrite**（把用户口语化问题改写成更适合检索的英文关键词），后续轮次为了省 token 不做

### 动态 Query 构造

每位大师第 2 轮起的检索 query 会自动拼接：

```
<用户原问题>
讨论焦点: <本轮前 3 条关键发言的摘要>
```

这样检索到的原著片段会跟**当前辩论焦点**对齐，而不是反复命中同一批"价值投资常识"。

---

## 💬 辩论流程详解

### 每轮流程

```
Round N:
  ├─ 5 位大师并行发言（每位都独立 RAG + 读 history）
  │    每位发言格式:
  │      正文（≤200字，引用原著精神）
  │      [本轮倾向: 看多/看空/观望]   ← 软信号
  │      [投票: 看多/看空/观望]        ← 硬信号（可选，最迟第6轮必投）
  │    大师可选择 [跳过]（仅限非第1轮）
  │
  ├─ CrossQuestion: 主持人读本轮全部发言，找出最大冲突，
  │   以冲突方口吻生成 1-2 条 60 字内的反问，写入 history
  │
  └─ UpdateRound: round_count += 1
      └─ should_continue 判定:
          • 全员投完最终票 → 跳 Researcher
          • 连续 2 轮本轮倾向全部一致（≥3 人发言，不含观望）→ 跳 Researcher
          • round_count < 6 → 继续
          • round_count = 6 → 跳 Researcher
```

### Researcher 最终报告结构

```
1. 客观现状概述
2. 大师共识与分歧点提炼
3. 投票结果解析
4. 最终操作建议（含风险提示）
```

可通过 `LLM_RESEARCHER_MODEL` 为研究员节点单独指定更强的模型（例如 `claude-opus-4-7-*` 或 `deepseek-reasoner`），只在这一次 LLM 调用中生效。

---

## 🧪 测试与调试

```bash
# 1. 端到端冒烟测试（不跑完整辩论，分钟级）
#    测: web_search Gemini 连通性、分层随机分布、早停逻辑、图编译
python scripts/smoke_e2e.py

# 2. LLM Provider + RAG 接线测试
python scripts/smoke_test.py

# 3. 仅 Gemini API 连通性
python scripts/test_gemini.py

# 4. BGE-M3 本地性能基准（首次会下载 2.3GB 模型）
python scripts/bench_bge_m3.py

# 5. 验证 ChromaDB 各集合 chunk 数
python -c "
from rag.vectorstore import collection_count
for k in ['buffett','dalio','marks','greenblatt','lynch']:
    print(f'{k:<12} {collection_count(k):>6} chunks')
"
```

常见的 debug 技巧：

- `main.py` 里 `MAX_ROUNDS` 调小到 2 以快速验证流程
- `rag/retriever.py` 里把 `top_k=5` 调大到 8-10 看检索召回
- 若想让某位大师**更嘴硬**，在 `MASTER_PERSONAS[...]['debate_bottom_line']` 加一句"绝不妥协"
- 若 `HF_ENDPOINT` 访问慢，把 `HF_ENDPOINT` 改成 `https://hf-mirror.com` 或删掉让它走官方

---

## ❓ 常见问题 (FAQ)

### Q1: 为什么选 BGE-M3 做本地 Embedding 而不是 OpenAI？

A: 辩论每轮每大师都会 RAG 一次（5 人 × 6 轮 = 30 次检索）。OpenAI Embedding 每次也要 API 调用，既有成本也有延迟（~300ms）。
BGE-M3 在 Apple MPS 上 ~75ms/query，零成本，完全离线。跨语言能力也强于 multilingual-large-3。

### Q2: 为什么默认只用 6 轮？

A: 辩论本质是"观点碰撞 → 澄清 → 趋同"，5 位大师实际超过 5 轮后，新增发言的信息量会急剧下降（边际收益递减）。
设 6 是给早停（全员投完 OR 连续 2 轮一致）一个缓冲，实际跑下来多数话题 3-4 轮就收敛了。

### Q3: 大师会"幻觉出"根本没说过的话吗？

A: 会，但我们用了三道防线：

1. Prompt 里强制"基于【知识库参考】中的原著内容支撑论点"
2. RAG 每次都检索本人原著片段塞进 context
3. Persona + thinking_framework + debate_bottom_line 三重约束人设

辩论风格高度贴合大师本人。但**不要把输出当成真实投资建议**——这始终是一个 LLM 仿真系统。

### Q4: 能否添加第 6、第 7 位大师？

A: 可以。在 `MASTER_PERSONAS` 加一份配置，准备 `data/raw/<new_master>/` 原始资料，在 `rag/config.py` 的 `MASTER_CONFIGS` 里加一项，然后 `python -m rag.build_kb --master <new_master>`。

### Q5: 同一轮大师真能"实时"互相看到吗？

A: **当前实现是并行 + 事后交叉质疑**，严格来说大师同轮只能看到上一轮的 history，同轮内看不到其他人。
"交叉质疑"节点弥补了这个不足——主持人会把同轮最大冲突整理后写进 history，下一轮大师就能针对冲突回应。

如果想要"真·串行辩论"（后发言者能实时看到前发言者同轮观点），改造点见代码里 `create_master_node` 周围的注释，我们已经预埋了 `stratified_shuffle` 出场顺序函数方便切换。

### Q6: 我换了 LLM Provider，之前构建的向量库还能用吗？

A: **能**。向量库只跟 Embedding Provider 绑定，跟 Chat Provider 无关。你改 `LLM_PROVIDER` 随意，`EMBEDDING_PROVIDER` 改了才需要 `--rebuild`。

### Q7: 我不想让 Gemini Grounding 每次都检索，太慢了怎么办？

A: 可以把 `DataCollection` 节点里的 `perform_web_search` 调用注释掉，或者改成缓存：同一 query 24h 内复用上次结果。
`web_search.py` 里已经预留了 `_call_counter` 做配额统计，加个磁盘缓存很简单。

### Q8: 免费 Gemini 额度用完了怎么办？

A: 免费额度是 1500 req/day（Gemini 2.5 Flash）。一场辩论约 25-30 个 LLM 调用 + 1 个 grounding 搜索，所以每天能跑 ~50 场辩论。到顶后要么等第二天，要么：

1. 切到付费 Gemini（绑境外卡，仍然很便宜）
2. 切到 DeepSeek（国内直连无限制）——改 `LLM_PROVIDER=deepseek` + `DEEPSEEK_API_KEY`

---

## 🛠️ Roadmap

- [ ] 真·串行辩论模式（后发言者可实时看到前发言者同轮观点）
- [ ] 向量库增量更新（仅处理有变化的文件，而非每次 --rebuild）
- [ ] 研究员节点支持输出 Markdown 文件 / 发送邮件
- [ ] 加入实时股价 / K 线图抓取（yfinance / akshare）
- [ ] Web UI (FastAPI + 前端)

---

*Built with ☕, 🎩, 🌐, ⚖️, 📊 and 🛒.*
