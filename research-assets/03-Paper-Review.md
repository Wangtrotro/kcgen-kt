# Paper Review Registry

> 不是 Summary，而是"这篇论文对我的研究有什么用"

---

## Paper 1: KCGen-KT (Self — arxiv 2502.18632)

**标题**: Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems

### Problem
编程教育中，Knowledge Component (KC) 的定义依赖人工标注，无法规模化。现有 KT 模型使用粗粒度 KC（如 "if-else"、"for loop"），无法精准追踪学生的编程知识状态。

### Why Existing Methods Fail
- **人工标注**: 成本高、主观性强、粒度不统一。CSEDM 数据集只有 18 个 KC，无法捕捉 "String indexing" 和 "String length" 的区别
- **传统 KT (DKT/SAKT)**: 将编程问题视为 binary 正错判断，丢失了代码结构中的丰富信息
- **Code-aware KT (CodeKT)**: 考虑了代码表示，但仍使用人工 KC，且不生成代码

### Core Idea
用 LLM (GPT-4o) 从解题代码的 AST 结构中自动提取细粒度 KC，再通过 LSTM + LoRA-LLaMA 的多任务框架同时追踪学生知识状态和生成个性化代码。

### Key Insight
KC 的自动生成不仅是一个"标注替代"问题，更是重新定义了 KC 的来源：从"专家认为的技能"到"代码结构中隐含的技能"。这使得 KC 具有可验证性——每个 KC 都可以追溯到具体的代码模式。

### Reusable For Me

| 方向 | 关联度 | 具体价值 |
|------|--------|---------|
| Education AI | ★★★★★ | 直接核心工作 |
| Agent Engineering | ★★★☆☆ | KC 生成 pipeline 的自动化程度可提升（Agent 化） |
| Knowledge Tracing | ★★★★★ | 定义了 Generative KT 的新范式 |
| LLM System | ★★★★☆ | Few-shot + Structured Output + LoRA 的完整技术栈 |

### 是否值得精读
★★★★★ — 自己的论文，必须精读并持续改进

---

## Paper 2: DKT — Deep Knowledge Tracing

**标题**: Deep Knowledge Tracing (Piech et al., NeurIPS 2015)

### Problem
传统知识追踪（BKT）使用隐马尔科夫模型，表达能力有限，且每个 KC 独立建模，无法捕捉 KC 之间的关联。

### Why Existing Methods Fail
- BKT 对每个 KC 独立训练，参数多、泛化差
- BKT 假设知识状态只有"掌握/未掌握"两种，过于简化

### Core Idea
用 RNN/LSTM 对学生的答题序列建模，隐状态自然地编码了跨 KC 的知识状态。

### Key Insight
知识状态是一个连续的、高维的隐变量，不应被离散化。LSTM 的隐状态天然适合表示这种连续演化的知识状态。

### Reusable For Me

| 方向 | 关联度 | 具体价值 |
|------|--------|---------|
| Education AI | ★★★☆☆ | 理论基础 |
| Knowledge Tracing | ★★★★★ | KCGen-KT 中 LSTM 部分直接继承了 DKT 思想 |
| LLM System | ★★☆☆☆ | RNN → Transformer 的范式迁移启示 |

### 是否值得精读
★★★★☆ — KT 领域的奠基论文，方法已被后续工作超越，但思想永不过时

---

## Paper 3: SAKT — Self-Attentive Knowledge Tracing

**标题**: A Self-Attentive model for Knowledge Tracing (Pandey & Karypis, EDM 2019)

### Problem
DKT 使用 RNN，存在长程依赖问题；且 RNN 的顺序计算限制了并行化。

### Why Existing Methods Fail
- DKT 的 RNN 在长序列上性能下降（梯度消失/爆炸）
- DKT 无法显式地选择"与当前问题最相关的历史交互"

### Core Idea
用 Transformer 的 self-attention 替代 RNN，让模型显式地 attend 到历史序列中与当前问题最相关的交互。

### Key Insight
知识追踪中，"哪些历史交互与当前预测最相关"这个问题，attention 机制比 RNN 的隐状态压缩更好地回答了。

### Reusable For Me

| 方向 | 关联度 | 具体价值 |
|------|--------|---------|
| Knowledge Tracing | ★★★★☆ | 与 KCGen-KT 的 LSTM 方案形成对比，可做 ablation |
| LLM System | ★★★☆☆ | Self-attention 在序列建模中的应用 |

### 是否值得精读
★★★☆☆ — 方法论有参考价值，但 KCGen-KT 选择了 LSTM 而非 Transformer 来编码 mastery（因为 KC mastery 的更新是天然顺序的）

---

## Paper 4: OKT — Open-ended Knowledge Tracing

**标题**: Open-ended Knowledge Tracing (Liu et al.)

### Problem
传统 KT 只预测 binary 正确性，无法生成开放式响应（如代码、文本）。

### Why Existing Methods Fail
- 传统 KT 的输出空间被限制在 {0, 1}
- 无法利用生成模型的丰富表示能力
- 无法为学生提供个性化的生成式反馈

### Core Idea
将 KT 与开放式文本生成结合，用 LLM 生成学生级别的响应。

### Key Insight
知识追踪的输出不应限于"对错预测"，而应扩展到"学生行为预测"——预测学生会写什么、会犯什么错。

### Reusable For Me

| 方向 | 关联度 | 具体价值 |
|------|--------|---------|
| Education AI | ★★★★★ | KCGen-KT 直接建立在 OKT 的框架之上 |
| Knowledge Tracing | ★★★★★ | Generative KT 的直接前驱工作 |
| LLM System | ★★★★☆ | LLM + KT 融合的技术方案 |

### 是否值得精读
★★★★★ — KCGen-KT 的直接基础，必须精读

---

## Paper 5: LoRA — Low-Rank Adaptation of Large Language Models

**标题**: LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., ICLR 2022)

### Problem
LLM 的全参数微调成本极高（需大量 GPU 内存和计算），且容易过拟合小数据集。

### Why Existing Methods Fail
- 全参数微调：GPU 显存不足
- Prompt Tuning：表达能力有限
- Adapter：增加推理延迟

### Core Idea
只微调注入在 attention 层的低秩矩阵 (ΔW = AB, A∈R^{d×r}, B∈R^{r×d}, r≪d)。

### Key Insight
预训练模型的权重更新具有低秩结构——大部分微调带来的变化集中在一个很低维的子空间中。

### Reusable For Me

| 方向 | 关联度 | 具体价值 |
|------|--------|---------|
| Education AI | ★★★★☆ | KCGen-KT 使用 LoRA 微调 LLaMA-3-8B |
| LLM System | ★★★★★ | 所有 LLM 微调项目的基础技术 |
| Agent Engineering | ★★★☆☆ | Agent 微调的低成本方案 |

### 是否值得精读
★★★★☆ — 技术细节已熟悉，定期回顾最新 LoRA 变体即可

**KCGen-KT 中的具体配置**: lora_alpha=256, lora_dropout=0.05, lora_r=128, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

---

## Paper 6: CodeBLEU

**标题**: CodeBLEU: a Method for Automatic Evaluation of Code Synthesis

### Problem
标准 BLEU 只考虑 n-gram 匹配，忽略了代码的结构（AST）和数据流信息。

### Why Existing Methods Fail
- BLEU：纯文本匹配，`if (a > b)` 和 `if (b < a)` 得分为 0
- Human evaluation：成本高、不可复现

### Core Idea
将 BLEU、加权 n-gram 匹配、AST 匹配和数据流匹配加权组合（默认 0.25, 0.25, 0.25, 0.25）。

### Key Insight
代码评估必须同时考虑表面形式、语法结构和语义行为三个层次。

### Reusable For Me

| 方向 | 关联度 | 具体价值 |
|------|--------|---------|
| Education AI | ★★★★☆ | KCGen-KT 的核心评估指标 |
| LLM System | ★★★★☆ | 代码生成任务的标准评估工具 |
| Agent Engineering | ★★★☆☆ | AI Coding Agent 的评估 |

### 是否值得精读
★★★☆☆ — 工具性论文，理解指标含义和局限即可

---

## 阅读策略总结

### 必读论文 (与核心方向直接相关)
1. OKT — KCGen-KT 的直接前驱
2. DKT — KT 领域奠基
3. 自己的论文 (KCGen-KT) — 持续改进

### 选读论文 (技术方案可复用)
4. LoRA — 微调技术
5. CodeBLEU — 评估工具
6. SAKT — 对比方法

### 待读方向 (当前缺口)
- Graph-based KT (GKT, GIKT) — 与 KC Graph 构建相关
- Forgetting Curve in KT — 与 KC 时序建模相关
- Multi-modal Educational AI — 与下一步研究方向相关
- LLM-based Automated Assessment — 与评论审核/笔记筛选相关
