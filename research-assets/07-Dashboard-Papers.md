# Paper Dashboard

> 论文不是读过就完了——核心问题是"对我的研究有什么用"

---

## 论文总表

| 论文 | 核心思想 | 值得复现？ | 研究关联度 | 精读优先级 |
|------|---------|-----------|-----------|-----------|
| **KCGen-KT** (Self) | LLM 自动生成 KC + Generative KT | ★ 已完成 | ⬛⬛⬛⬛⬛ | ★★★★★ |
| **OKT** | 开放式知识追踪 → 生成式 KT | ✅ 值得 | ⬛⬛⬛⬛⬛ | ★★★★★ |
| **DKT** | RNN 建模知识状态 | ❌ 已被超越 | ⬛⬛⬛⬛☐ | ★★★★☐ |
| **SAKT** | Self-attention 替代 RNN | ⚠️ 可做对比 | ⬛⬛⬛⬛☐ | ★★★☐☐ |
| **LoRA** | 低秩矩阵微调 LLM | ★ 已使用 | ⬛⬛⬛⬛☐ | ★★★★☐ |
| **CodeBLEU** | AST + 数据流的代码评估 | ★ 已使用 | ⬛⬛⬛☐☐ | ★★★☐☐ |

---

## 按研究方向分布

### Knowledge Tracing 相关

| 论文 | 与我研究的关系 | 技术可复用性 |
|------|-------------|-------------|
| DKT | KCGen-KT 中 LSTM 部分的理论基础 | LSTM 序列建模 |
| SAKT | 可做 attention-based KT baseline 对比 | Self-attention 机制 |
| OKT | KCGen-KT 的直接前驱，定义了 Generative KT | LLM + KT 架构 |

**缺口**: 缺少 Graph-based KT (GKT, GIKT) 的系统阅读 → 与 KC Graph 方向直接相关

### LLM System 相关

| 论文 | 与我研究的关系 | 技术可复用性 |
|------|-------------|-------------|
| LoRA | KCGen-KT 的微调方案 | 所有 LLM 微调项目 |
| CodeBLEU | KCGen-KT 的评估指标 | 代码生成评估 |

**缺口**: 缺少 RAG、LLM Memory、Tool Use 的系统阅读

### Education AI 相关

**缺口**: 缺少 Automated Assessment、Intelligent Tutoring System 的近期论文

---

## 待读论文优先级队列

| 优先级 | 论文方向 | 原因 | 预期价值 |
|--------|---------|------|---------|
| P0 | GKT / GIKT (Graph-based KT) | 与 KC Graph 构建直接相关 | 高 — 可能成为下一篇论文的 baseline |
| P0 | AKT (Context-Aware Attentive KT) | KT 领域 SOTA，需要对比 | 高 — 实验对比 |
| P1 | Forgetting Curve in KT | 与 KC 时序演化相关 | 中高 — 拓展 KCGen-KT |
| P1 | LLM for Automated Code Assessment | 与评论审核/评分相关 | 中高 — 新项目方向 |
| P2 | RAG for Education | 与个性化学习材料推荐相关 | 中 — 中长期方向 |
| P2 | Multi-modal Learning Analytics | 与多模态教育 AI 相关 | 中 — 长期方向 |

---

## 论文 → 项目映射

```
KCGen-KT 项目直接使用的论文:
├── OKT → 框架设计
├── DKT → LSTM 知识建模
├── LoRA → LLM 微调
├── CodeBLEU → 评估指标
└── SentenceTransformer → KC 嵌入

下一步项目可能使用的论文:
├── GKT → KC Graph 构建
├── AKT → KT 对比实验
├── Forgetting Curve → 时序 KC 建模
└── RAG → 个性化学习推荐
```

---

## 核心思想精华提取

### 最有影响力的 3 个思想

1. **知识状态是连续的、高维的隐变量** (DKT)
   - 打破了 BKT "掌握/未掌握" 二元假设
   - 在 KCGen-KT 中体现为 LSTM 的连续 hidden state

2. **KT 的输出不应限于 binary prediction** (OKT)
   - 知识追踪应该预测"学生会怎么做"，而不仅是"对不对"
   - 在 KCGen-KT 中体现为代码生成任务

3. **预训练模型的权重更新具有低秩结构** (LoRA)
   - 使得 8B 参数模型在单 GPU 上可微调
   - 在 KCGen-KT 中使得整个系统实际可用

### 最可能被引用的概念

| 概念 | 来源 | 在我研究中的体现 |
|------|------|----------------|
| Deep Knowledge State | DKT | LSTM hidden state |
| Generative Knowledge Tracing | OKT + KCGen-KT | Code Generation conditioned on KS |
| Low-Rank Adaptation | LoRA | LLaMA-3 微调 |
| Automated KC Discovery | KCGen-KT (self) | LLM + AST → KC |
| CodeBLEU Metric | CodeBLEU | 代码评估标准 |
