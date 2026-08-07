# 多Agent投资决策器 - LangGraph 工作流架构设计

## 1. 核心设计理念

基于您的需求，我们将采用 **LangGraph** 来编排这个多Agent系统。LangGraph 非常适合这种具有循环（辩论）、条件分支（轮次判断）以及状态管理（共享辩论上下文）的复杂工作流。

**系统特点：**
- **状态驱动 (State-driven)**：维护一个全局 `State`，记录标的信息、当前辩论轮次、所有历史发言、投票结果。
- **并行思考 (Parallel Execution)**：每一轮中，多位大师Agent并行获取市场数据并发表观点，提高效率。
- **有界循环 (Bounded Loop)**：严格限制辩论不超过10轮，控制Token成本。
- **投票与归纳双重机制**：结合量化的“多数投票”与定性的“助理研究员总结”，确保最终输出既有明确倾向，又有严密的逻辑支撑。

---

## 2. 工作流状态图 (State Graph)

以下是该系统的执行流转过程：

```mermaid
graph TD
    %% 节点定义
    Start([用户输入：投资标的/问题])
    DataCollection[数据收集节点<br>调用API获取财报/K线/宏观数据]
    
    subgraph 辩论场循环 (Max 10 Rounds)
        direction TB
        DebateManager{轮次判断器<br>Round < 10 ?}
        
        %% 大师并行发言
        MasterA[巴菲特 Agent<br>调取价值投资RAG库]
        MasterB[达利欧 Agent<br>调取宏观周期RAG库]
        MasterC[马克斯 Agent<br>调取周期风险RAG库]
        MasterD[格林布拉特 Agent<br>调取量化价值RAG库]
        MasterE[彼得·林奇 Agent<br>调取GARP与常识RAG库]
        
        UpdateState[更新辩论历史上下文]
    end
    
    VotingNode[多数投票节点<br>统计各方最终立场: 看多/看空/观望]
    
    Researcher[中立助理研究员 Agent<br>综合观点、处理冲突、输出研报]
    
    End([输出最终决策报告])

    %% 连线关系
    Start --> DataCollection
    DataCollection --> DebateManager
    
    %% 循环内逻辑
    DebateManager -- "Yes (继续辩论)" --> MasterA
    DebateManager -- "Yes (继续辩论)" --> MasterB
    DebateManager -- "Yes (继续辩论)" --> MasterC
    DebateManager -- "Yes (继续辩论)" --> MasterD
    DebateManager -- "Yes (继续辩论)" --> MasterE
    
    MasterA --> UpdateState
    MasterB --> UpdateState
    MasterC --> UpdateState
    MasterD --> UpdateState
    MasterE --> UpdateState
    
    UpdateState --> DebateManager
    
    %% 结束循环
    DebateManager -- "No (达到10轮)" --> VotingNode
    
    VotingNode --> Researcher
    Researcher --> End
```

---

## 3. 核心节点与Agent角色设定

### 3.1 状态管理器 (State Definition)
在 LangGraph 中，我们需要定义一个 `TypedDict` 来在所有节点间传递信息：
- `query`: 用户的原始问题（如“现在可以买入特斯拉吗？”）
- `market_data`: 收集到的客观市场数据（实时更新）
- `debate_history`: 数组，记录每一轮每位大师的发言内容
- `round_count`: 整型，当前辩论轮次（初始为0）
- `votes`: 字典，记录最终轮各位大师的投票（如 `{"Buffett": "看空", "Dalio": "观望", "Marks": "看空"}`）
- `final_report`: 助理研究员生成的最终结论

### 3.2 专家Agent (The Masters)
- **输入**：`query` + `market_data` + `debate_history`
- **动作**：
  1. 根据历史发言（尤其是别人反驳自己的点），检索自己的 RAG 知识库。
  2. 结合实时数据，为自己的投资哲学辩护，或指出其他大师逻辑中的漏洞。
  3. 在最后一轮时，必须强制输出一个明确的立场标签：`[看多]`、`[看空]` 或 `[观望]`。
- **提示词约束**：绝对禁止脱离自身流派。例如，即使达利欧强调宏观利好，巴菲特也必须紧盯高昂的市盈率进行反驳。

### 3.3 中立助理研究员Agent (The Researcher)
这是系统的“大脑”和最终输出接口。
- **角色设定**：一位拥有顶级投行经验、兼具深厚政经法背景的资深宏观/策略助理研究员。ta 绝对理性，不偏袒任何流派。
- **输入**：完整的10轮 `debate_history` + `votes` (例如：2票看多，3票看空)。
- **任务**：
  1. **客观陈述事实**：基于 `market_data` 简述当前标的客观现状。
  2. **提炼共识与分歧**：总结各位大师在辩论中达成的共识点，以及根本的分歧点在哪里（例如：分歧在于短期宏观波动 vs 长期企业护城河）。
  3. **解释投票结果**：说明为什么最终投票结果是 X 胜出。
  4. **给出最终建议**：结合大师观点、投票结果和自身的政经法知识库，给出一份具有高可操作性的投资建议及风险提示。

---

## 4. 为什么这样设计？

1. **解决 Token 消耗与动态收敛**：
   - 设定了**硬性边界**（最多10轮）来防止死循环和控制成本。
   - 引入了**提前收敛机制**（Early Convergence）。允许大师们在觉得信息充分时提前给出投票。系统会实时检测，一旦5位大师全部投票完毕，不论当前是第几轮，都会立即终止辩论，大幅节省Token。
2. **结合了民主与精英主义**：“多数投票”体现了群智决策的直观倾向，而“助理研究员”的介入避免了简单的数字比拼掩盖了深层次的风险逻辑（即精英主义的最后把关）。
3. **极佳的用户体验**：用户最终看到的不仅仅是一个“买/卖”的指令，而是一份**包含了顶级大师思维碰撞过程的深度研报**。

---

## 5. 下一步建议 (Step 4)

架构已经明确。接下来，我们可以开始**搭建基础代码框架**了。

由于您可能还没有准备好真实的 RAG 数据库和复杂的外部 API，我建议我们先写一个 **LangGraph 的核心骨架代码（Mock 版本）**。在这个代码中：
1. 我们将定义完整的 State。
2. 我们用简单的 Prompt 模拟 5 位大师（比如让他们通过 LLM 直接扮演角色，暂不外接 RAG）。
3. 实现 10 轮循环的控制逻辑。
4. 实现助理研究员的归纳输出。

这样您可以直接运行这段代码，在控制台看到大师们互相辩论和最终研报生成的过程。如果您同意，我将为您编写这段 Python 核心代码框架。