# Methodology Dashboard

> 已沉淀的方法论：从哪来，到哪去

---

## 方法论总表

| 方法论 | 来源实验 | 成熟度 | 复用次数 | 可迁移场景 |
|--------|---------|--------|---------|-----------|
| Error-Driven Knowledge Extraction | E1 (KC生成), 评论审核 | 🟢 成熟 | 3+ | 所有知识提取场景 |
| Structured Output Engineering | E1 (KC生成), 评论审核 | 🟢 成熟 | 5+ | 所有 LLM 应用 |
| Semantic Clustering Pipeline | E4 (KC聚类), E6 (采样) | 🟢 成熟 | 2 | 文本去重/标签构建 |
| LLM Semantic Space Injection | E2 (Mastery嵌入) | 🟡 实验验证 | 1 | 连续值注入 LLM |
| Multi-task Loss Normalization | E3 (多任务训练) | 🟡 实验验证 | 1 | 多任务学习 |
| Diversity-Aware Sampling | E6 (代码采样) | 🟡 实验验证 | 2 | Few-shot 示例选择 |

---

## 方法论详情

### 🟢 Error-Driven Knowledge Extraction

**成熟度**: 生产级 — 在多个项目中一致有效

```
输入: 失败案例集合 (错误代码 / 审核漏判 / Agent 失败)
输出: 结构化知识 (KC / 审核规则 / Agent 改进)

Pipeline:
失败案例 → 归因标注 → 模式聚类 → 知识编码 → 验证闭环
```

**来源实验**:
- E1: incorrect submissions → 更好的 KC
- 评论审核: 漏判案例 → 更好的审核 Prompt

**复用场景清单**:
| 场景 | 输入 | 期望输出 |
|------|------|---------|
| KC 生成 | 学生错误代码 | 知识缺口列表 |
| Prompt 优化 | LLM 错误输出 | Prompt 改进规则 |
| Agent 改进 | Agent 执行失败日志 | Tool 设计改进 |
| 代码审核 | 被遗漏的 bug | Bug pattern 库 |
| 考试预测 | 预测错误的题目 | 特征工程改进 |

---

### 🟢 Structured Output Engineering

**成熟度**: 生产级 — LLM 应用的基础能力

```
输入: 任务需求 + 期望输出格式
输出: 可解析、可审计的结构化 LLM 输出

Pipeline:
定义 Schema → 选择约束机制 → 添加 reasoning 字段 → 验证修复 → 迭代
```

**来源实验**:
- E1: `response_format={"type": "json_object"}` + JSON template
- KC 聚类总结: 二选一结构消除歧义

**核心技巧**:
```
1. 永远加 "reasoning" 字段 — CoT 提升输出质量
2. 用 null 标记不适用的字段 — 比不返回更清晰
3. 给 enum 类型的字段提供明确选项 — 缩小输出空间
4. 用 JSON Schema 验证输出 — 不合规则 retry
```

---

### 🟢 Semantic Clustering Pipeline

**成熟度**: 生产级 — 文本聚类的标准方案

```
输入: 大量语义相近但措辞不同的文本条目
输出: 去重后的代表性条目 + 映射字典

Pipeline:
语义编码 → 距离计算 → 层次聚类 → 簇内摘要 → 映射回溯
```

**来源实验**:
- E4: KC 聚类 (150+ → 75)
- E6: 代码多样性聚类采样

**技术选择指南**:
| 数据类型 | 推荐 Encoder | 推荐聚类方法 |
|---------|-------------|-------------|
| 自然语言标签 | all-MiniLM-L6-v2 | Linkage (average) |
| 代码片段 | GraphCodeBERT | AgglomerativeClustering |
| 短文本 | SBERT | HDBSCAN |
| 长文档 | 分段编码+拼接 | Spectral Clustering |

---

### 🟡 LLM Semantic Space Injection

**成熟度**: 实验验证 — 仅在 1 个场景中使用

```
输入: 连续概率值 p ∈ [0, 1]
输出: LLM 可理解的嵌入向量

公式: emb = p * emb(True) + (1-p) * emb(False)
```

**来源实验**:
- E2: KC mastery level → LLaMA 输入嵌入

**待验证**:
- [ ] 在 True/False 以外的 anchor pair 上验证 (如 Yes/No, Positive/Negative)
- [ ] 当 p ≈ 0.5 时嵌入是否有意义
- [ ] 在非 LLaMA 模型上的效果

---

### 🟡 Multi-task Loss Normalization

**成熟度**: 实验验证 — 工程实践

```
输入: 多个不同量级的 loss
输出: 归一化后的总 loss

公式: L_total = α * (L1/L1.detach() + L2/L2.detach()) 
              + (1-α) * L3/L3.detach()
```

**来源实验**:
- E3: gen_loss (~2.0) + kc_loss (~0.01) + pred_loss (~0.5)

**注意事项**:
```
⚠️ 当 loss → 0 时需要 epsilon 保护: loss / (loss.detach() + 1e-8)
⚠️ 过度归一化会抹平训练前后期的梯度差异
```

---

### 🟡 Diversity-Aware Sampling

**成熟度**: 实验验证 — 有理论支撑

```
输入: 候选示例池
输出: 多样性最大化的 k 个示例

Pipeline:
候选编码 → 聚类 (k 簇) → 每簇选代表 → 人工检查
```

**来源实验**:
- E6: 代码采样 (GraphCodeBERT + Clustering)
- E1: few-shot 示例选择 (条件判断 + 数组操作)

---

## 方法论演化路线

```
当前已沉淀 (2025):
├── Error-Driven Knowledge Extraction  [成熟]
├── Structured Output Engineering      [成熟]
├── Semantic Clustering Pipeline       [成熟]
├── LLM Semantic Space Injection       [实验]
├── Multi-task Loss Normalization      [实验]
└── Diversity-Aware Sampling           [实验]

下一步可沉淀 (2025-2026):
├── KC Graph Auto-Construction         [规划中]
├── Forgetting-Aware KT Training       [规划中]
├── Agent Eval Pipeline                [探索中]
└── Cross-Language Transfer Method     [规划中]
```

---

## 方法论复用矩阵

| | KC生成 | 评论审核 | Agent | Exam | 笔记筛选 | RAG |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Error-Driven | ✅ | ✅ | ✅ | ⭕ | ⭕ | ⭕ |
| Structured Output | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Semantic Clustering | ✅ | ⭕ | ⭕ | ❌ | ✅ | ✅ |
| Semantic Injection | ✅ | ❌ | ❌ | ⭕ | ❌ | ❌ |
| Loss Normalization | ✅ | ❌ | ❌ | ⭕ | ❌ | ❌ |
| Diversity Sampling | ✅ | ⭕ | ⭕ | ⭕ | ✅ | ✅ |

✅ 已使用 | ⭕ 可迁移 | ❌ 不适用
