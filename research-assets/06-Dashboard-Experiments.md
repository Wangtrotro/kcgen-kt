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
| E7 | Baseline KC 对比 | 自动 KC > 人工 KC？ | 粗粒度 KC 维度冗余 | ✅ 完成 | ✅ |
| E8 | Prompt 构造 | KC-aware 模板怎么设计？ | 位置+占位符影响 attention | ✅ 完成 | 部分 |
| E9 | Predictor 架构消融 | 单层 vs 多层 MLP？ | 多层捕捉非线性模式 | ✅ 完成 | 部分 |
| E10 | Loss Function 消融 | BCE vs CrossEntropy？ | BCE 概率输出更自然 | ✅ 完成 | ❌ |
| E11 | 跨数据集验证 | 方法跨语言泛化？ | Java→Python 可迁移 | ✅ 完成 | ✅ |

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

### E7: Baseline KC 对比

```
研究问题: 人工 18 类粗粒度 KC vs 自动 75 类细粒度 KC？
假设验证: ✅ 成功 — 自动 KC 在所有指标上优于人工 baseline

实验变量:
  - configs.baseline: True (人工18 KC) / False (自动75 KC)
  - Baseline 来源: prompt_concept.csv (18列 binary 标注)

关键发现:
  "人工 KC 的问题不是'不够细'，而是'维度冗余'。
   18 个 KC 中条件判断类的 4 个几乎完全共现。"

方法论遗产: 对比实验应关注 baseline 的"失效模式"
```

### E8: Prompt 构造

```
研究问题: KC-aware 输入模板怎么设计最有效？
假设验证: ✅ 成功 — KC 信息位于 question 和 code 之间效果最好

实验变量:
  - KC 位置: question 前 / question 后 code 前 / code 后
  - 占位符: ? token → True/False 加权嵌入
  - 分界标记: "written" token 定位 question/code 边界

关键发现:
  "Token-level 定位机制是必要的工程实践。
   原始问题中的 : 和 ? 必须预处理，否则干扰占位符定位。"
```

### E9: Predictor 架构消融

```
研究问题: 正确性预测用单层 Linear 还是多层 MLP？
假设验证: ✅ 部分成功 — 多层 MLP 略优，但 question-only pooling 是关键

实验变量:
  - 单层: Linear(4096→1) + Xavier init
  - 多层: 4096→512→64→1 + Kaiming init + ReLU
  - configs.predictor_multilayer: True/False

关键发现:
  "Question-only pooling（mask 掉 answer hidden states）
   比 predictor 架构选择更重要。避免了信息泄露。"
```

### E10: Loss Function 消融

```
研究问题: 正确性预测用 BCE 还是 CrossEntropy？
假设验证: ⚠️ 差异不显著

实验变量:
  - BCE: BCEWithLogitsLoss → Linear(4096,1) → sigmoid
  - CE: CrossEntropyLoss → Linear(4096,2) → softmax
  - configs.binary_loss_fn: 'BCE' / 'CE'

关键发现:
  "BCE 的 sigmoid 输出可直接用于 KC mastery 估计。
   AUC 差异不大，但 BCE 的 calibration 更好。"
```

### E11: 跨数据集验证

```
研究问题: KCGen-KT 能否跨语言泛化？
假设验证: ✅ 成功 — 框架通用，细节需适配

实验变量:
  - CodeWorkout: Java, 18 baseline KC, CSEDM 数据
  - Falcon: Python, 独立 baseline KC, 不同 few-shot 示例
  - CodeBLEU lang: 'java' / 'python'

关键发现:
  "方法框架跨语言通用，但 KC 生成的 few-shot 示例
   和聚类粒度需要按语言调整。"
```

---

## 实验间关系图

```
E1 (KC 生成)
  ├── E4 (粒度消融) — E1 的输出经聚类后需要确定最优粒度
  ├── E6 (采样策略) — E1 的输入代码如何选择
  ├── E7 (Baseline 对比) — E1 的自动 KC vs 人工 KC
  ├── E11 (跨数据集) — E1 的跨语言泛化验证
  └── 输出 KC → E2, E3, E5, E8

E2 (Mastery 嵌入)
  ├── 与 E3 联合 — mastery 注入方式影响多任务训练效果
  └── 依赖 E8 — prompt 中占位符设计决定嵌入插入位置

E3 (多任务训练)
  ├── 依赖 E2 — 需要 mastery 嵌入方案
  ├── 与 E5 联合 — transition layer 影响多任务损失平衡
  ├── E9 (Predictor 消融) — predictor 架构影响预测任务
  └── E10 (Loss 消融) — loss function 影响训练动态

E5 (Transition)
  └── 依赖 E4 — KC 数量决定是否需要 transition

E6 (采样策略)
  └── 影响 E1 — 采样质量决定 KC 生成质量

E8 (Prompt 构造)
  └── 影响 E2 — 占位符定位决定 mastery 嵌入位置
```

---

## 假设验证矩阵

| 假设 | 支持实验 | 验证结果 | 可信度 |
|------|---------|---------|--------|
| LLM 可自动生成高质量 KC | E1, E6, E7 | ✅ 成立 | 高 |
| 错误代码提升 KC 质量 | E1 | ✅ 成立 | 高 |
| True/False 嵌入优于数值注入 | E2, E8 | ✅ 成立 | 中高 |
| 多任务训练带来双向增益 | E3, E9 | ✅ 成立 | 高 |
| 存在最优 KC 粒度 | E4 | ✅ 成立 | 中高 |
| Bottleneck 防止过拟合 | E5 | ✅ 成立 | 中 |
| 多样性采样 > 随机采样 | E6 | ✅ 成立 | 中高 |
| 自动 KC 优于人工 KC | E7 | ✅ 成立 | 高 |
| KC 位置影响 attention | E8 | ✅ 成立 | 中 |
| 多层 Predictor 优于单层 | E9 | ⚠️ 略优 | 中 |
| BCE 优于 CrossEntropy | E10 | ⚠️ 差异不显著 | 低 |
| 方法跨语言可迁移 | E11 | ✅ 成立 | 中高 |
