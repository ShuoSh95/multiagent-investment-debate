# 🏛️ 多Agent投票决策器 · AI版金融大鳄辩论赛

**中文** | [English](README_EN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B.svg)](https://streamlit.io/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ShuoSh95/multiagent-investment-debate/pulls)

---

看到某个板块涨势正猛，脑子里一个小人喊：**「快买！绝佳时机，错过再等十年！」**
另一个小人劝：**「别追高，小心被挂在山顶。」**

如果你也常被这两个小人吵得拿不定主意——不如把问题交给五位「金融大鳄」：

**让 AI 巴菲特、达利欧、马克斯、格林布拉特、林奇，替你吵一架。**

他们会引用自己的原著、当庭查证实时行情、互相交叉质疑、有人当场倒戈、有人死不松口，
最后投票表决，并由一位中立研究员为你写一份有理有据的研报。

**听大儒为你辩经，看完你大概就知道该怎么想了。**

📖 **[围观一场真实辩论 →](docs/example_debate.md)**
（辩题："美伊迎来和平转机，能大笔买入 A 股大盘指数基金吗？"——5 位大鳄 6 轮交锋，全票看空）

---

> [!IMPORTANT]
> ## ⚠️ 免责声明
>
> **本项目是多 Agent 技术的研究与娱乐演示，不构成任何投资建议、要约或操作指引。**
>
> 与 AI 交易类研究项目的通行原则一致：辩论结果受所选模型、温度参数、数据质量、时间窗口等**非确定性因素**影响，
> 每次运行都可能不同，仅供学习参考与娱乐。
>
> - 「大师」人设均为基于公开资料的 **AI 模拟**，不代表真实人物的观点或立场，与其本人及所属机构无任何关联，亦未获授权或背书。
> - 大语言模型的输出**可能存在事实错误、数据过时或幻觉**。
> - 股市有风险。真实投资决策请咨询持牌专业人士；任何人据本项目输出操作，**盈亏与后果完全自负**。

---

## 🎭 这是什么？

一场你可以**全程围观**的 AI 投资辩论赛。你出辩题，系统开庭：

```
你的问题 ──► 开庭前调查（实时行情/估值/新闻自动检索）
                │
                ▼
     五位大鳄多轮辩论（最多 6 轮）
     ├─ 每位发言前翻阅自己的原著（RAG 检索股东信/备忘录/著作）
     ├─ 觉得证据不足？当庭申请查数据（实时 Web 检索）
     ├─ 每轮结束：主持人挑起交叉质疑，逼他们互相回应
     └─ 亮出本轮倾向：看多 / 看空 / 观望（可随辩论进程倒戈）
                │
                ▼（全员投票完毕，或连续两轮意见一致）
     🗳️ 最终投票 ──► 📝 研究员结案陈词（结构化研报）──► 💬 你可以继续追问
```

### 五位辩手

| 辩手 | 流派 | 辩论风格 |
|---|---|---|
| 🎩 沃伦·巴菲特 | 价值投资 | 只谈护城河和安全边际，"宁可错过，不可做错" |
| 🌐 瑞·达利欧 | 全球宏观 | 张口就是债务周期和经济机器，警告系统性风险 |
| ⚖️ 霍华德·马克斯 | 周期/逆向 | 市场越乐观他越警惕，"最大的错误都在牛市犯下" |
| 📊 乔尔·格林布拉特 | 量化价值 | 只认 ROIC 和盈利收益率，宏观叙事一概"与我无关" |
| 🛒 彼得·林奇 | 成长/常识 | 从商场货架找十倍股，嫌指数基金没有惊喜 |

流派天然冲突，所以吵得起来。**但如果他们观点出奇得一致，那你可得好好看看喽！**

## ✨ 为什么值得一看

**1. 他们不是在演，是真的在引经据典。**
每位大鳄发言前都会检索自己的原著知识库（巴菲特历年股东信、马克斯备忘录、达利欧《原则》……约 12,700 个原文片段），
关键论断直接引用原文。你看到的"价格是你付出的，价值是你得到的"，真的出自股东信。

**2. 缺数据会承认，绝不编数字。**
辩手觉得手头证据不够时，会**自己**发起实时检索（达利欧查宏观利率，格林布拉特查 ROIC）；
查不到就必须当庭承认"缺乏该数据，无法判断"——系统层面禁止编造。

**3. 有真实的观点交锋和立场变化。**
交叉质疑环节逼着他们互相回应；辩论中你能看到有人从"观望"被说服转向"看空"，有人被围攻依然死守立场。
不是五篇平行小作文，是一场有胜负手的辩论。

**4. 吵完给你能带走的东西。**
中立研究员输出结案研报：共识与分歧、投票解析、操作建议与风险提示。看不懂的地方，直接在页面里追问。

## 🚀 快速开始

需要：Python 3.9+、约 5GB 磁盘、一个免费的 [Gemini API Key](https://aistudio.google.com/apikey)。

```bash
git clone https://github.com/ShuoSh95/multiagent-investment-debate.git
cd multiagent-investment-debate
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-full.txt   # 本地完整版（含 BGE-M3）；云端 Demo 用 requirements.txt

cp .env.example .env          # 填入你的 GOOGLE_API_KEY

python -m rag.build_kb --master all --rebuild   # 首次构建大师知识库（约 15-25 分钟）

streamlit run web/streamlit_app.py               # 浏览器打开 http://localhost:8501，开庭！
```

🌐 **不想本地装？** 可部署到免费的 [Streamlit Community Cloud](docs/DEPLOY_STREAMLIT.md)（Demo 档：BM25 检索 + Flash 模型 + 每日限量）。

装依赖慢、想换模型、想喂自己的电子书、想看检索与辩论机制 →
**[完整安装与技术文档 docs/SETUP.md](docs/SETUP.md)**

## ❓ 三个最常被问的问题

**Q：这能当投资建议吗？**
不能。这是技术演示与思维工具——它的价值是把一个问题的多空理由、风险点、不同流派的视角摆开给你看，帮你把"脑内打架"外化成可检视的论证。决策请自己做，或咨询持牌人士。

**Q：大师说的话是真的吗？**
发言中的引文来自本人公开著作的 RAG 检索，行情数据来自实时搜索，都可溯源。但组织语言的是 LLM，人设是模拟，**整场辩论请当作"高质量的仿真"而非本人观点**。

**Q：能加入第 6 位大师吗？比如芒格？**
能，加一份 persona 配置 + 喂原始资料即可，见 [SETUP 文档](docs/SETUP.md#添加新大师)。欢迎 PR。

## 🗺️ Roadmap

- [x] 🌐 公开在线 Demo（[Streamlit Cloud 部署指南](docs/DEPLOY_STREAMLIT.md)）
- [ ] @指定大师单独追问
- [ ] 真·串行辩论（后发言者实时看到前面同轮发言）
- [ ] 辩论中途插话，向大师提问
- [ ] 实时股价 / K 线数据源（yfinance / akshare）

详见 [ROADMAP.md](ROADMAP.md)。

## 📄 License

[MIT](LICENSE)。大师人设为 AI 模拟，与真实人物无关联、无授权、无背书；输出不构成投资建议，详见文首免责声明。

---

*Built with ☕, 🎩, 🌐, ⚖️, 📊 and 🛒.*
