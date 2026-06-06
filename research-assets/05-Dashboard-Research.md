# Research Dashboard

> 当前研究全景、关键发现与下一步行动的总控视图

---

## 研究方向总览

| 方向 | 核心项目 | 论文状态 | 活跃度 | 下一步动作 |
|------|---------|---------|--------|-----------|
| **Knowledge Tracing** | KCGen-KT | ✅ arxiv 2502.18632 | 🟢 活跃 | KC Graph + 遗忘建模 |
| **Education AI** | 评论审核 / 笔记筛选 | 📝 进行中 | 🟡 中等 | 构建 Benchmark |
| **Agent Engineering** | Codex / Cursor Workflow | 🔬 探索中 | 🟢 活跃 | Education Tutor Agent |
| **LLM System** | Prompt / Eval / LoRA | 🧰 工具链 | 🟢 持续 | RAG + Memory |

---

## 关键发现（按重要性排序）

### Tier 1: 可直接写论文的发现

| # | 发现 | 证据强度 | 关联论文 |
|---|------|---------|---------|
| 1 | 知识追踪本质是条件生成问题 | ⭐⭐⭐⭐⭐ | KCGen-KT (已发表) |
| 2 | LLM 可从代码 AST 自动提取超越人工的 KC | ⭐⭐⭐⭐⭐ | KCGen-KT (已发表) |
| 3 | 错误案例驱动的知识提取优于经验驱动 | ⭐⭐⭐⭐ | 待写 position paper |

### Tier 2: 方法论层面的发现

| # | 发现 | 适用范围 |
|---|------|---------|
| 4 | 结构化约束比 Prompt 措辞优化更高效 | 所有 LLM 应用 |
| 5 | Few-shot 示例多样性 > 数量 | Prompt Engineering |
| 6 | KC 粒度应由数据驱动而非专家决定 | 教育 AI |

### Tier 3: 技术洞察

| # | 发现 | 技术细节 |
|---|------|---------|
| 7 | LLM embedding 空间可表示连续概率 | True/False 加权嵌入 |
| 8 | Multi-task loss 归一化是必要的工程实践 | `loss/loss.detach()` |

---

## 论文分布

```
已发表
└── KCGen-KT (arxiv 2502.18632) — Knowledge Tracing × LLM × Education

可写
├── Error-Driven Knowledge Discovery in Educational AI — 跨项目 position paper
├── Generative Knowledge Tracing — KCGen-KT 的理论延伸
└── Data-Driven KC Granularity Optimization — 消融实验扩展

需要更多实验
├── KC Graph Construction from Code AST — 需要图结构实验
├── Cross-Language KC Transfer — 需要多语言数据
└── Education Tutor Agent with KC-aware Memory — 需要 Agent 框架
```

---

## 实验分布（共 11 个实验）

```
按研究问题分组:

[KC 自动生成] ← 4 个实验
├── E1: LLM Few-shot KC 生成 (正确 vs 正确+错误代码)
├── E4: KC 聚类粒度消融
├── E6: 代码采样策略对比
└── E7: Baseline KC 对比 (人工18类 vs 自动75类)

[知识状态建模] ← 3 个实验
├── E2: Mastery Level 嵌入策略 (True/False 加权)
├── E5: Transition Layer 消融
└── E8: Prompt 构造 (KC-aware 输入模板)

[训练策略] ← 3 个实验
├── E3: 多任务联合训练 (Generation + Prediction + KC)
├── E9: Predictor 架构消融 (单层 vs 多层 MLP)
└── E10: Loss Function 消融 (BCE vs CrossEntropy)

[泛化性验证] ← 1 个实验
└── E11: 跨数据集验证 (CodeWorkout/Java vs Falcon/Python)
```

---

## 研究方向关联图

```
                    Education AI
                   /     |      \
          评论审核   笔记筛选   Exam Predict
              \        |        /
               ↓       ↓       ↓
              LLM System (Prompt / Eval / LoRA)
               ↑       ↑       ↑
              /        |        \
    Agent Eng.   KCGen-KT ★   问一问 Agent
        |          /    \         |
    Eval Frame  DKT    CodeKT    |
        |        |       |       |
    Auto Research   Solution Path
                 \     |
              KC Graph / Cognitive Graph
```

---

## 待解决的核心问题 (Research Gaps)

| 优先级 | 问题 | 难度 | 前置条件 |
|--------|------|------|---------|
| P0 | KC 间依赖关系的自动发现 | 高 | KCGen-KT 完成 ✅ |
| P0 | 真实教育平台的 A/B 测试 | 中 | 需要平台合作 |
| P1 | KC 时序演化 + 遗忘曲线建模 | 中 | KCGen-KT + 时序数据 |
| P1 | 跨编程语言 KC 迁移 | 中 | 多语言数据集 |
| P2 | Education Tutor Agent | 高 | KC + Agent 框架 |
| P2 | 基于 KC mastery 的个性化练习生成 | 中 | KCGen-KT + 题库 |

---

## 技术栈总览

| 层级 | 工具/技术 | 用途 |
|------|----------|------|
| LLM 骨架 | LLaMA-3-8B-Instruct | 代码生成 / KT |
| 微调 | LoRA (r=128, alpha=256) | 高效微调 |
| 量化 | BitsAndBytes 8-bit | 降低显存 |
| 知识提取 | GPT-4o + JSON output | KC 自动生成 |
| 序列建模 | LSTM (1-layer) | KC mastery 追踪 |
| 嵌入 | SentenceTransformer | KC 语义聚类 |
| 代码嵌入 | GraphCodeBERT | 代码多样性采样 |
| 聚类 | Scipy linkage + HDBSCAN | KC 去重 |
| 评估 | CodeBLEU + Acc/AUC/F1 + Distinct-N | 多维评估 |
| 实验管理 | Hydra + WandB | 配置 + 跟踪 |
