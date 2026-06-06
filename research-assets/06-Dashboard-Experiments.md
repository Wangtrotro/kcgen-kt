# Experiment Dashboard

> 所有实验的结构化视图：假设 → 设计 → 发现 → 状态

---

## 实验总表

| ID | 实验名称 | 研究问题 | 核心假设 | 状态 | 论文? |
|----|---------|---------|---------|------|------|
| E1 | KC 自动生成 Pipeline | 能否自动化 KC 标注？ | LLM + AST → 细粒度 KC | ✅ 完成 | ✅ |
| E2 | Mastery Level 嵌入 | 如何注入连续 mastery？ | True/False 加权嵌入 | ✅ 完成 | ✅ |
| E3 | 多任务联合训练 | 代码生成+KT 能否互益？ | 共享表示带来双向增益 | ✅ 完成 | ✅ |
| E4 | KC 粒度消融 | 最优 KC 数量？ | 数据驱动的最优粒度 | ✅ 完成 | ✅ |
| E5 | Transition Layer | 是否需要维度转换？ | Bottleneck 防过拟合 | ✅ 完成 | ✅ |
| E6 | 代码采样策略 | 多样性采样有效吗？ | 聚类采样 > 随机采样 | ✅ 完成 | ✅ |

---

## 实验详情卡片

### E1: KC 自动生成 Pipeline

```
研究问题: 能否用 LLM 从代码中自动提取 Knowledge Components？
假设验证: ✅ 成功 — LLM 生成的 KC 比人工标注更细粒度且可解释

实验变量:
  - 代码来源: GPT 生成 vs 学生正确代码 vs 学生正确+错误代码
  - Few-shot 示例数: 2 (精选)
  - LLM: GPT-4o, temperature=0
  - 后处理: SentenceTransformer + 层次聚类 + LLM 摘要

关键指标:
  - 唯一 KC 数: 18 (人工) → 75 (自动, 聚类后)
  - 每题 KC 数: ~3.5 (人工) → ~5 (自动)
  - KC 可解释性: 主观评估 — 自动 KC 更具体

核心发现:
  "错误代码是知识发现的金矿。LLM 不仅能识别'需要什么技能'，
   还能识别'学生通常缺失什么技能'。"

方法论遗产: Error-Driven Knowledge Extraction
```

### E2: Mastery Level 嵌入策略

```
研究问题: 如何在 LLM 输入空间中表示连续的 KC mastery level？
假设验证: ✅ 成功 — True/False 加权嵌入优于直接数值注入

实验变量:
  - 注入方式: True/False 加权 vs 数值 token vs soft token
  - KC 聚合: prod vs mean vs geo_mean
  - LSTM → mastery: sigmoid 激活

关键指标:
  - 正确性预测 AUC: prod > geo_mean > mean
  - 代码生成 CodeBLEU: mean ≈ geo_mean > prod

核心发现:
  "利用 LLM 已有的语义空间比创造新表示更有效。
   True/False 是 LLM 已充分学习的概念。"

方法论遗产: LLM Semantic Space Injection
```

### E3: 多任务联合训练

```
研究问题: 代码生成和正确性预测能否在同一模型中互相增益？
假设验证: ✅ 成功 — 多任务训练带来双向增益

实验变量:
  - 任务组合: Generation only / Generation + Prediction / Gen + Pred + KC
  - Alpha 权重: 0.8 (生成主导)
  - Loss 归一化: loss/loss.detach()
  - Predictor: Linear(4096→1) or MLP(4096→512→64→1)

关键指标:
  - 多任务 AUC > 单任务 AUC
  - 多任务 CodeBLEU ≈ 单任务 CodeBLEU (不降低)
  - Loss 归一化使训练收敛更稳定

核心发现:
  "知识追踪和代码生成是同一硬币的两面。
   正确性预测是知识状态的'压缩'，代码生成是'展开'。"

方法论遗产: Multi-task Loss Normalization
```

### E4: KC 粒度消融

```
研究问题: 自动生成的 KC 应该聚类到多细的粒度？
假设验证: ✅ 部分成功 — 存在最优粒度，但数值依赖数据集

实验变量:
  - n_cluster: {20, 30, 50, 75, 100, 150}
  - 聚类方法: average linkage, cosine distance
  - 评估: 下游 KT 性能

关键发现:
  "KC 粒度不是越细越好。CodeWorkout 上 75 是最优。
   数据集规模决定了最优粒度。"

方法论遗产: 数据驱动的 KC 粒度选择
```

### E5: Transition Layer 消融

```
研究问题: LSTM 输出维度 ≠ KC 数量时，是否需要转换层？
假设验证: ✅ 成功 — Bottleneck 防止过拟合

实验变量:
  - transition_dim: 64 (bottleneck)
  - 结构: Linear(64→KC数) + ReLU

关键发现:
  "KC mastery 的内在维度 (~64) 远小于表面 KC 数量 (75)。
   很多 KC 的掌握程度是相关的。"
```

### E6: 代码采样策略

```
研究问题: 用于 KC 生成的正确代码如何采样？
假设验证: ✅ 成功 — 聚类多样性采样 > 随机采样

实验变量:
  - 采样方式: 随机 vs GraphCodeBERT + AgglomerativeClustering
  - n_correct: {1, 2, 5}

关键发现:
  "代码多样性比数量更重要。聚类采样能捕获不同解法风格。"

方法论遗产: Diversity-Aware Sampling
```

---

## 实验间关系图

```
E1 (KC 生成)
  ├── E4 (粒度消融) — E1 的输出经聚类后需要确定最优粒度
  ├── E6 (采样策略) — E1 的输入代码如何选择
  └── 输出 KC → E2, E3, E5

E2 (Mastery 嵌入)
  └── 与 E3 联合 — mastery 注入方式影响多任务训练效果

E3 (多任务训练)
  ├── 依赖 E2 — 需要 mastery 嵌入方案
  └── 与 E5 联合 — transition layer 影响多任务损失平衡

E5 (Transition)
  └── 依赖 E4 — KC 数量决定是否需要 transition

E6 (采样策略)
  └── 影响 E1 — 采样质量决定 KC 生成质量
```

---

## 假设验证矩阵

| 假设 | 支持实验 | 验证结果 | 可信度 |
|------|---------|---------|--------|
| LLM 可自动生成高质量 KC | E1, E6 | ✅ 成立 | 高 |
| 错误代码提升 KC 质量 | E1 | ✅ 成立 | 高 |
| True/False 嵌入优于数值注入 | E2 | ✅ 成立 | 中高 |
| 多任务训练带来双向增益 | E3 | ✅ 成立 | 高 |
| 存在最优 KC 粒度 | E4 | ✅ 成立 | 中高 |
| Bottleneck 防止过拟合 | E5 | ✅ 成立 | 中 |
| 多样性采样 > 随机采样 | E6 | ✅ 成立 | 中高 |
