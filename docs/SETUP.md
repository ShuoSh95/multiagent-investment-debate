# 完整安装与技术文档

> 本文面向想在本地跑起来、调参、或二次开发的开发者。
> 如果你只想了解产品是什么，请看 [README](../README.md)。

## 目录

- [环境准备与安装](#环境准备与安装)
- [配置说明 (.env)](#配置说明-env)
- [构建知识库](#构建知识库)
- [启动方式](#启动方式)
- [目录结构](#目录结构)
- [Web Search 机制](#web-search-机制)
- [RAG 检索机制](#rag-检索机制)
- [辩论流程详解](#辩论流程详解)
- [添加新大师](#添加新大师)
- [测试与调试](#测试与调试)
- [FAQ](#faq)

---

## 环境准备与安装

- macOS / Linux / Windows
- Python **3.9+**（推荐 3.10+，3.9 会有一些 google 库的弃用警告但不影响功能）
- ~5GB 磁盘空间（BGE-M3 模型 2.3GB + 知识库数据 ~80MB + Python 依赖 ~2GB）

```bash
git clone https://github.com/ShuoSh95/multiagent-investment-debate.git
cd multiagent-investment-debate

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 本地完整版（含 BGE-M3 / torch）。公开 Demo 云端用根目录 requirements.txt（轻量）
# 如在国内网络，建议用清华镜像加速
pip install -r requirements-full.txt \
    --default-timeout=1800 --retries 10 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn
```

> 公开 Demo 部署到 Streamlit Community Cloud：见 [DEPLOY_STREAMLIT.md](DEPLOY_STREAMLIT.md)。

## 配置说明 (.env)

复制模板：`cp .env.example .env`，然后按需填写：

```bash
# ==================== Chat LLM ====================
LLM_PROVIDER=gemini             # deepseek | openai | anthropic | gemini | qwen | zhipu | doubao

# 只填对应 Provider 的 Key，其他注释掉
GOOGLE_API_KEY=你的_gemini_key       # Gemini（https://aistudio.google.com/apikey 免费申请）
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

可通过 `LLM_RESEARCHER_MODEL` 为研究员节点单独指定更强的模型，只在最终总结这一次 LLM 调用中生效。

## 构建知识库

首次运行必须构建，耗时约 15-25 分钟：

```bash
# 采集 + 切分 + 嵌入 + 索引（5 位大师全量）
python -m rag.build_kb --master all --rebuild

# 若已有 data/raw/*，只做向量化，不重新抓数据：
python -m rag.build_kb --master all --skip-acquire --rebuild

# 单独构建某位大师（常用于调试或补数据）：
python -m rag.build_kb --master buffett --skip-acquire
```

首次会触发 `BAAI/bge-m3` 从 HuggingFace Mirror 下载 ~2.3GB，之后缓存在 `~/.cache/huggingface/`。

**预期产出：**

| 大师 | 约 Chunks |
|---|---|
| 沃伦·巴菲特 | 2,400 |
| 瑞·达利欧 | 6,500 |
| 霍华德·马克斯 | 3,700 |
| 乔尔·格林布拉特 | ~30 |
| 彼得·林奇 | ~30 |
| **合计** | **~12,700** |

> 格林布拉特和林奇的原始出版物版权限制较严，当前主要依赖 Wikipedia 条目。
> 可手动把你购买的电子书（PDF/EPUB）放到 `data/raw/greenblatt/` 或
> `data/raw/lynch/` 下，然后 `--skip-acquire --rebuild` 重建。

**数据来源可信度分层：**

| Tier | 来源 | 示例 |
|---|---|---|
| **Tier 1 · 原著** | 大师本人公开出版物 / 官方演讲 / 股东信 | 巴菲特历年股东信、Marks Memos、Dalio *Principles* |
| **Tier 2 · 转述** | 高质量访谈、演讲文字稿 | Dalio 在 Harvard / Davos 的演讲 |
| **Tier 3 · 参考** | 维基百科条目 | 用于原著难以获取的大师（如 Greenblatt、Lynch） |

## 启动方式

**方式 A · 网页界面（推荐）**

```bash
streamlit run web/streamlit_app.py
# 浏览器打开 http://localhost:8501
```

网页端能力：

- 输入框提交问题 → 市场数据卡片 → 5 位大师逐位流式发言（含主动检索标记）→ 交叉质疑 → 最终研报
- 大师发言带"本轮倾向"彩色标签 + "最终投票"徽章
- 辩论结束后底部出现**追问对话框**，可基于完整研报继续提问（研究员带上下文流式作答）
- 左侧栏列出**历史辩论**（本地 SQLite），点击任意一场回看 + 继续追问

**方式 B · 终端命令行**

```bash
python main.py
```

进入交互后输入投资问题即可。用 Gemini 2.5 Pro 推理较慢但更深入（一场约 5-15 分钟）；
想快可把 `LLM_MODEL` 换成 `gemini-2.5-flash`。

## 目录结构

```
InvestmentAgent/
├── main.py                  # 主入口：LangGraph 图编排 + 流式终端 UI + 按需检索
├── llm_provider.py          # 多 Provider 切换（DeepSeek / Gemini / Claude ...）
├── web_search.py            # 实时 Web 检索（Gemini Grounding / Tavily）
├── requirements.txt
├── .env.example             # 环境变量模板（复制为 .env 后填 Key）
├── ROADMAP.md               # 版本迭代规划
│
├── web/                     # Streamlit 网页界面
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
│
├── docs/                    # 文档（本文件、示例辩论实录、设计文档）
│
└── data/                    # ⚠️ 本地生成，已被 .gitignore 忽略（不进仓库）
    ├── raw/                 # 各大师原始资料（含受版权书籍，请自行采集）
    ├── chroma_db/           # ChromaDB 向量库（build_kb 生成）
    ├── bm25_index/          # BM25 索引（build_kb 生成）
    └── history.db           # 网页端辩论历史（SQLite）
```

> **关于 `data/`**：知识库（原始语料、向量库、BM25 索引）和辩论历史都**不在 Git 仓库里**——
> 原始语料含受版权材料，且体积近 300MB。Clone 后请自行构建。

## Web Search 机制

**两层检索：**

1. **辩论开始前的公共检索** —— `DataCollection` 节点做一次 Web 搜索，获取用户问题的最新公开信息（价格、估值、财报、宏观事件等），塞进 `market_data`，供所有大师共享。

2. **大师按需主动检索（每轮）** —— 每位大师发言前，用轻量模型（`LLM_FAST_MODEL`）按**自身分析框架**自评：是否还缺现有资料里没有的关键数据？
   - 缺 → 生成具体查询并发起检索（如达利欧查"GDP/CPI/央行利率/债务率"，格林布拉特查"ROIC/盈利收益率"）
   - 不缺 → 跳过，不浪费调用
   - 检索结果注入该大师本轮 prompt；**同一场辩论内相同查询会缓存去重**
   - 可用 `ENABLE_AGENT_SEARCH=0` 整体关闭

   > 🛡️ **数据诚信铁律**：所有大师只能引用真实检索到 / 知识库里的数据，
   > 检索失败或数据缺失时必须如实声明（如"缺乏该公司最新 ROIC 数据，无法判断"），**绝不编造**。

**优先级链路：**

```
1. Gemini 原生 Google Search Grounding    ← 如果设置了 GOOGLE_API_KEY
   └─ 完全免费（算在 Gemini 免费额度内）
   └─ 3 次重试 + 多 model fallback（2.5-flash → 2.0-flash → 1.5-flash）

2. Tavily API                              ← 如果设置了 TAVILY_API_KEY
   └─ 免费 1000 次/月（https://tavily.com/）

3. Graceful 降级                           ← 都没配
   └─ 提示未获取到实时数据，仅基于 RAG 辩论
```

**软性限流**：`web_search.py` 里定义了每日 200 次 Gemini / 每月 1000 次 Tavily 的 soft limit，到达后提醒并自动降级。

## RAG 检索机制

对每次检索请求，执行混合检索（hybrid retrieval）：

1. **向量检索**（BGE-M3 embedding + Cosine 距离）取 Top-K
2. **BM25 关键词检索** 取 Top-K（捕捉专有名词）
3. **Reciprocal Rank Fusion (RRF)** 融合两路结果
4. **首轮**额外叠加一次 **Self-Query LLM rewrite**（把口语化问题改写成更适合检索的英文关键词），后续轮次为省 token 不做

**动态 Query 构造**：每位大师第 2 轮起的检索 query 会自动拼接：

```
<用户原问题>
讨论焦点: <本轮前 3 条关键发言的摘要>
```

这样检索到的原著片段会跟**当前辩论焦点**对齐，而不是反复命中同一批"价值投资常识"。

## 辩论流程详解

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

**Researcher 最终报告结构**：客观现状概述 → 大师共识与分歧点提炼 → 投票结果解析 → 最终操作建议（含风险提示）。

**架构图：**

```
用户问题 → DataCollection（Web 检索市场数据）
        → 5 大师并行（每轮：RAG 原著 + 按需 Web 检索 + 发言/倾向/投票）
        → CrossQuestion（交叉质疑）
        → should_continue?（早停判定 / 最多 6 轮）
        → Researcher（结案研报）
```

更多设计细节见 [architecture_design.md](architecture_design.md) 与 [agent_capability_design.md](agent_capability_design.md)。

## 添加新大师

1. 在 `main.py` 的 `MASTER_PERSONAS` 加一份配置（persona + thinking_framework + debate_bottom_line）
2. 准备 `data/raw/<new_master>/` 原始资料（PDF/EPUB/TXT/HTML/MD 均可）
3. 在 `rag/config.py` 的 `MASTER_CONFIGS` 里加一项
4. 运行 `python -m rag.build_kb --master <new_master>`

## 测试与调试

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

常用 debug 技巧：

- `main.py` 里 `MAX_ROUNDS` 调小到 2 以快速验证流程
- `rag/retriever.py` 里把 `top_k=5` 调大到 8-10 看检索召回
- 若想让某位大师**更嘴硬**，在 `MASTER_PERSONAS[...]['debate_bottom_line']` 加一句"绝不妥协"
- 若 HuggingFace 访问慢，设置 `HF_ENDPOINT=https://hf-mirror.com`

## FAQ

### Q1: 为什么选 BGE-M3 做本地 Embedding 而不是 OpenAI？

辩论每轮每大师都会 RAG 一次（5 人 × 6 轮 = 30 次检索）。OpenAI Embedding 每次都要 API 调用，既有成本也有延迟（~300ms）。
BGE-M3 在 Apple MPS 上 ~75ms/query，零成本，完全离线，跨语言能力也强。

### Q2: 为什么默认只有 6 轮？

辩论本质是"观点碰撞 → 澄清 → 趋同"，超过 5 轮后新增发言的信息量急剧下降。
设 6 是给早停（全员投完 OR 连续 2 轮一致）一个缓冲，实际多数话题 3-4 轮就收敛了。

### Q3: 大师会"幻觉出"根本没说过的话吗？

会，但有三道防线：① Prompt 强制"基于【知识库参考】中的原著内容支撑论点"；② RAG 每次检索本人原著片段塞进 context；③ Persona + thinking_framework + debate_bottom_line 三重约束人设。
辩论风格高度贴合大师本人，但**不要把输出当成真实投资建议**——这始终是一个 LLM 仿真系统。

### Q4: 同一轮大师真能"实时"互相看到吗？

当前实现是**并行发言 + 事后交叉质疑**：同轮内看不到其他人，但"交叉质疑"节点会把同轮最大冲突写进 history，下一轮大师即可针对冲突回应。
"真·串行辩论"在 [ROADMAP](../ROADMAP.md) 中，代码里已预埋 `stratified_shuffle` 出场顺序函数方便切换。

### Q5: 换了 LLM Provider，之前构建的向量库还能用吗？

**能**。向量库只跟 Embedding Provider 绑定，跟 Chat Provider 无关。改 `LLM_PROVIDER` 随意，改 `EMBEDDING_PROVIDER` 才需要 `--rebuild`。

### Q6: 免费 Gemini 额度用完了怎么办？

一场辩论约 25-30 个 LLM 调用 + 1 次 grounding 搜索。免费额度到顶后：

1. 等第二天配额刷新
2. 切到付费 Gemini（仍然很便宜）
3. 切到 DeepSeek（国内直连）——改 `LLM_PROVIDER=deepseek` + `DEEPSEEK_API_KEY`
