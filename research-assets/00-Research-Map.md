# Research Map

> 从"做过什么"到"发现了什么"——按研究主题组织的全景知识地图

---

## 研究主题总览

```
Education AI
├── KCGen-KT (知识组件自动生成 + 知识追踪)
├── 评论审核 (学生评论质量审核)
├── 优质笔记筛选
├── Exam Predict
├── 问一问 Agent
└── 教育场景 LLM 应用

Agent Engineering
├── Codex / Cursor AI Coding Workflow
├── Eval Framework & Harness
├── Auto Research
└── Prompt Engineering Pipeline

Knowledge Tracing
├── KCGen-KT ★ (本仓库核心项目)
├── DKT / SAKT 经典方法
├── CodeKT (编程场景 KT)
├── Solution Path
└── Cognitive Graph / KC Graph

LLM System
├── Prompt Engineering
├── Structured Output
├── Evaluation & Benchmarking
├── Memory & RAG
└── Tool Use
```

---

## 1. KCGen-KT：自动化知识组件生成与编程知识追踪

**论文**: [Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems](https://arxiv.org/abs/2502.18632)

### 核心问题

编程教育中的知识追踪面临一个根本性瓶颈：**Knowledge Component（KC）的定义依赖人工专家标注**，无法规模化。传统 KC 粒度粗（如"if-else"、"for loop"），无法捕捉编程任务中的细粒度技能组合。如何在不依赖人工标注的情况下，自动生成高质量、细粒度的 KC，并将其融入知识追踪系统？

### 已完成工作

| 组件 | 描述 | 状态 |
|------|------|------|
| KC 自动生成管线 | 基于 Solution AST + LLM (GPT-4o) 的 few-shot KC 提取 | ✅ 完成 |
| KC 聚类与去重 | 基于语义嵌入 (SentenceTransformer) + 层次聚类 + LLM 摘要 | ✅ 完成 |
| KC-aware KT 模型 | LSTM 跟踪 KC mastery + LLaMA-3-8B (LoRA) 代码生成 | ✅ 完成 |
| 多任务学习 | 代码生成 + 正确性预测 (Binary prediction) 联合训练 | ✅ 完成 |
| Mastery Level 嵌入 | True/False token 加权嵌入替换 `?` 占位符 | ✅ 完成 |
| 评估体系 | CodeBLEU + Acc/AUC/F1 + Distinct-N 多维评估 | ✅ 完成 |
| Baseline 对比 | 人工标注 KC (18类粗粒度) vs 自动生成 KC (75类细粒度) | ✅ 完成 |

### 关键实验

1. **KC 生成质量实验**: GPT-4o few-shot 从正确+错误代码中提取 KC，与 18 类人工标注对比
2. **KC 粒度消融**: 不同聚类数 (n_cluster) 对下游 KT 性能的影响
3. **KC 聚合方式消融**: prod vs mean vs geo_mean 三种 KC mastery 聚合策略
4. **多任务联合训练**: 代码生成 + 正确性预测的 alpha 权重平衡
5. **Transition 层实验**: 是否通过线性变换层在 LSTM 输出与 KC 维度之间做转换
6. **Loss 归一化**: normalized loss vs raw loss 的训练稳定性对比

### 核心发现

1. **LLM 可以生成比人工更细粒度的 KC**: GPT-4o 从代码 AST 中提取的 KC 粒度远高于传统人工标注，且语义上可解释（如 "Conditional Logic and Evaluation"、"String Manipulation Techniques"）
2. **错误代码是 KC 生成的关键信号**: 加入 incorrect student submissions 后，KC 能更好地捕捉"学生容易缺失的知识点"，这本质上是一种 error-driven knowledge discovery
3. **KC mastery 的表示方式直接影响知识追踪精度**: 用 True/False token 的加权嵌入来表示 mastery level，比直接数值注入更有效，因为它利用了 LLM 的语义空间
4. **知识追踪和代码生成可以互相增益**: 多任务设计使模型既能追踪学生状态又能生成个性化代码，这暗示 KT 任务本质上是一种 conditional generation 问题

### 当前缺口

- 仅在 CSEDM (CodeWorkout) 和 Falcon 数据集上验证，**未在大规模真实编程教育平台上测试**
- KC 生成依赖 GPT-4o 的 API 调用，**成本和延迟** 在生产环境中可能成为瓶颈
- **KC 的时序演化**未被建模：学生对某 KC 的掌握程度应该有遗忘曲线
- **跨编程语言的 KC 迁移**未被探索（当前仅 Java/Python）
- 未与 **Graph-based KT** 方法（如 GKT、GIKT）进行对比

### 下一步研究方向

1. **KC Graph Construction**: 将自动生成的 KC 组织成知识图谱，建模 KC 间的前驱/后继/组合关系
2. **Personalized Code Generation**: 基于学生 KC mastery profile 生成个性化练习/hint
3. **Cross-language KC Transfer**: 探索 Java KC 能否迁移到 Python/C++ 场景
4. **Online Learning Integration**: 在真实教育平台上部署，收集 A/B 测试数据
5. **KC Evolution Modeling**: 引入遗忘机制和 spaced repetition 策略

---

## 2. 评论审核（学生评论质量审核）

### 核心问题

教育平台上学生评论数量巨大且质量参差不齐，**如何自动识别高质量、有建设性的学生评论**？这不是简单的分类任务，而是一个需要理解评论语义、上下文和教育价值的推理任务。

### 已完成工作

- 基于 LLM 的评论审核 Prompt 迭代
- 错误案例驱动的 Prompt 优化方法论

### 核心发现

- **审核本质是推理任务而非分类任务**：简单的 sentiment/toxicity 分类器无法捕捉教育评论的"建设性"维度
- **结构化约束对教育场景审核收益高于 Prompt 技巧本身**：明确定义审核维度（如"是否包含具体建议"、"是否有事实错误"）比优化 Prompt 措辞更有效

### 当前缺口

- 缺乏公开的教育评论审核 benchmark
- 多语言评论的处理
- 与本仓库 KCGen-KT 项目的潜在整合：利用 KC 图谱评估评论是否涉及关键知识点

### 下一步研究方向

- 构建 Education Comment Review Benchmark
- 将 KC 图谱作为审核的知识先验

---

## 3. Agent Engineering

### 核心问题

如何系统性地构建、评估和迭代 AI Agent 系统？Agent 不仅是 Prompt + Tool Use 的组合，更是一套工程方法论。

### 已完成工作

- Codex / Cursor 工作流实践
- Eval Framework 探索
- Auto Research 原型

### 核心发现

- **Eval 先于 Build**: 没有评估框架的 Agent 开发是盲人摸象。先定义 eval metric，再迭代 agent
- **Error-driven iteration 是 Agent 开发的核心循环**: 分析失败 case → 归因 → 改进 → 验证的闭环
- **Tool Use 的设计决定 Agent 的天花板**: 工具的抽象粒度、错误处理、retry 策略直接影响 Agent 可靠性

### 当前缺口

- 缺乏系统化的 Agent Eval Benchmark（特别是教育场景）
- Agent 的 memory 和长期学习能力不足
- 多 Agent 协作的架构设计

### 下一步研究方向

- 基于 KCGen-KT 构建 Education Tutor Agent
- 设计 Agent-for-Research 系统（自动文献调研 + 实验设计）

---

## 4. Knowledge Tracing（广义知识追踪）

### 核心问题

如何准确建模学生的知识状态及其随时间的演化？编程场景的知识追踪有何独特挑战？

### 已完成工作

| 方向 | 描述 | 与本仓库关系 |
|------|------|------------|
| DKT/SAKT | 经典深度知识追踪方法调研 | 理论基础 |
| CodeKT | 编程场景专用 KT | KCGen-KT 的前驱工作 |
| KCGen-KT | 本仓库：自动 KC 生成 + KT | ★ 核心 |
| Solution Path | 学生解题路径建模 | 可与 KC mastery 结合 |
| Cognitive Graph | 认知图谱 + KT | KC Graph 的自然延伸 |

### 核心发现

- **编程 KT 与传统 KT 的本质区别**: 编程问题的答案不是 binary 对错，而是一个代码序列。传统 KT 只建模 P(correct)，编程 KT 需要建模 P(code|knowledge_state)
- **KC 的自动发现是可行的**: 本仓库证明了 LLM + AST 可以替代人工专家进行 KC 标注
- **KC mastery level 可以作为 conditional generation 的条件**: 这开辟了 KT 与 NLG/Code Generation 交叉的新方向

### 当前缺口

- **缺乏 KC 间关系的建模**: 当前 KC 是独立处理的，但现实中 KC 之间有前驱/后继关系
- **缺乏遗忘建模**: 学生对 KC 的掌握会随时间衰减
- **跨平台/跨课程的 KT 迁移**

### 下一步研究方向

- KC Dependency Graph 自动构建
- 引入时间衰减和 spaced repetition
- 跨编程语言的 KC 迁移学习

---

## 5. LLM System

### 核心问题

如何在教育场景中有效部署和使用 LLM？包括 Prompt Engineering、Structured Output、Evaluation、Memory 和 RAG。

### 已完成工作（从本仓库可见）

- **Prompt Engineering**: KC 生成的 system prompt 设计（few-shot、CoT、structured JSON output）
- **Structured Output**: JSON response format 强制输出结构化 KC
- **Evaluation**: CodeBLEU + 分类指标的多维评估体系
- **LoRA 微调**: 4-bit 量化 + LoRA 的高效 LLM 适配

### 核心发现

- **Few-shot 示例的选择比数量更重要**: KCGen 中用 2 个精选 example 比随机选择 5 个效果更好
- **Structured Output 是教育 AI 的基础能力**: 教育场景需要可解释、可审计的输出，JSON schema 约束是最低要求
- **LoRA 微调在教育场景性价比极高**: 8-bit 量化 + LoRA 使 8B 模型在单 GPU 上可训练，且效果接近全参数微调

### 下一步研究方向

- RAG-enhanced KT：用检索增强生成来提供个性化学习材料
- Memory System：长期记忆学生的学习历史和偏好
- Multi-modal Education LLM：整合代码、自然语言、图表的多模态教育模型
