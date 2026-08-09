# 部署到 Streamlit Community Cloud（公开 Demo）

长期免费、直接跑 Streamlit。因免费档约 **2.7GB RAM**，Demo **不加载 BGE-M3**，只用 BM25 检索原著片段（效果略逊于本地混合检索，但够围观）。

## 你需要准备

1. GitHub 账号（仓库已推到 [ShuoSh95/multiagent-investment-debate](https://github.com/ShuoSh95/multiagent-investment-debate)）
2. [Gemini API Key](https://aistudio.google.com/apikey)
3. HuggingFace Token（读私有知识库 `vae01/investment-debate-kb`）

## 一键部署步骤

1. 打开 [share.streamlit.io](https://share.streamlit.io/) → 用 GitHub 登录  
2. **New app** → 选择仓库 `ShuoSh95/multiagent-investment-debate`（或你的 fork）  
3. 填写：
   - **Main file path**: `web/streamlit_app.py`
   - **App URL**（可选）: `multiagent-investment-debate`
4. 点 **Advanced settings** → **Secrets**，粘贴：

```toml
DEMO_MODE = "1"
LLM_PROVIDER = "gemini"
DEMO_LLM_MODEL = "gemini-2.5-flash"
LLM_FAST_MODEL = "gemini-2.5-flash"
WEB_SEARCH_MODEL = "gemini-2.5-flash"
ENABLE_AGENT_SEARCH = "1"
EMBEDDING_PROVIDER = "bm25"
DEMO_DAILY_DEBATE_LIMIT = "30"
DEMO_SESSION_DEBATE_LIMIT = "2"
DEMO_AGENT_SEARCH_CAP = "1"
KB_DATASET = "vae01/investment-debate-kb"

GOOGLE_API_KEY = "你的_Gemini_Key"
HF_TOKEN = "你的_HF_Token"
```

5. **Deploy!** 首次会装依赖并拉取 ~120MB 知识库，约 5–10 分钟。

部署成功后地址形如：

`https://<your-name>-multiagent-investment-debate.streamlit.app`

## 说明

| 项 | Demo（Cloud） | 本地完整版 |
|---|---|---|
| 检索 | BM25 only | BGE-M3 + BM25 + RRF |
| 辩论模型 | gemini-2.5-flash | gemini-2.5-pro（可配） |
| 轮次 | 4 | 6 |
| 依赖 | `requirements.txt`（轻量） | `requirements-full.txt` |

## 常见问题

**App 启动后侧边栏没有历史？**  
画廊精选会在首次启动时写入；若磁盘被清空，重启后会重新 seed。

**报错拉不到知识库？**  
确认 `HF_TOKEN` 对 `vae01/investment-debate-kb` 有读权限，且 Secrets 里不要设置 `HF_ENDPOINT`（Cloud 在海外，走官方即可）。

**内存超限 / 被杀？**  
确认 `DEMO_MODE=1` 且 `EMBEDDING_PROVIDER=bm25`，不要装 `sentence-transformers`。
