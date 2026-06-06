# Experiment Registry

> 不是记录"做了什么"，而是记录"发现了什么"

---

## 实验 1：KC 自动生成 — LLM Few-shot Pipeline

### 研究问题
人工标注 Knowledge Component 无法规模化，能否用 LLM 从代码 AST 中自动提取高质量 KC？

### 核心假设
LLM 具备从代码结构中抽取"底层编程技能"的能力，且通过 few-shot 引导可以生成比人工标注更细粒度、更可解释的 KC。

### 实验设计

**方案 A：纯正确代码驱动** (`get_KCs_few_shot`)
- 输入：问题描述 + 2 个 GPT 生成的样本解答
- Few-shot：2 个手工 KC 标注示例
- 模型：GPT-4o，temperature=0
- 输出：JSON 格式的 KC 列表，每个 KC 含 name 和 reasoning

**方案 B：正确+错误代码驱动** (`get_KCs_few_shot_with_inc`)
- 输入：问题描述 + k 个正确代码 + k 个错误代码
- 正确代码来源：学生提交（通过语义聚类采样多样化解法）
- 错误代码来源：随机/聚类采样的不及格提交
- 额外指令：要求 LLM 分析错误代码揭示的知识缺失

**KC 后处理**
- SentenceTransformer (`all-MiniLM-L6-v2`) 编码 KC 名称
- 层次聚类 (scipy linkage, average method) 压缩到 n_cluster 个簇
- GPT-4o 对每个簇生成代表性 KC 名称

### 结果

| 配置 | 唯一 KC 数 | 每题平均 KC | KC 可解释性 |
|------|-----------|------------|------------|
| 人工标注 (baseline) | 18 | ~3.5 | 粗粒度 (如 "If/Else") |
| GPT-4o 正确代码 only | ~150+ | ~6 | 高度可解释 |
| GPT-4o 正确+错误代码 | ~180+ | ~7 | 捕捉到 misconception |
| 聚类到 75 个 KC | 75 | ~5 | 最佳平衡点 |

### 核心发现

**发现级别的结论，不是指标堆砌：**

1. **错误代码是知识发现的金矿**：加入 incorrect submissions 后，LLM 不仅能识别"需要什么技能"，还能识别"学生通常缺失什么技能"。这改变了 KC 的定义方式——从"解题所需技能"到"教学应关注的技能"。

2. **KC 粒度存在最优区间**：太细（150+）导致数据稀疏，太粗（18）丢失信息。75 个 KC 在 CodeWorkout 数据集上达到最佳平衡。这意味着 KC 粒度不应由领域专家主观决定，而应由下游任务性能数据驱动。

3. **Few-shot 示例的质量比数量重要得多**：2 个精心挑选的 KC 标注示例（一个条件判断类，一个数组操作类）足以引导 GPT-4o 生成高质量 KC。这验证了"示例选择的多样性"比"示例数量"更关键。

### 方法论价值
**高**。这套 LLM-based 知识提取 pipeline 可迁移到：
- 任何需要从非结构化内容中提取结构化知识的场景
- 数学/物理/化学等学科的 KC 自动生成
- 代码审核中的 bug pattern 分类

### 可写入论文？
**是** ✅ — 已写入 arxiv 2502.18632

---

## 实验 2：KC Mastery Level 嵌入策略

### 研究问题
如何将 LSTM 预测的 KC mastery level（连续值 0-1）有效地注入 LLM 的输入空间？

### 核心假设
直接将 mastery 数值作为 token 注入效果差，因为 LLM 的输入空间是离散 token embedding。用 True/False token 的加权线性组合可以在 LLM 的语义空间中表示连续的 mastery 概率。

### 实验设计
- 对 prompt 中每个 `?` 占位符，用 `mastery * emb(True) + (1-mastery) * emb(False)` 替换
- LSTM 输出经 sigmoid 得到 mastery level
- 三种 KC 聚合方式：
  - **prod**: 所有 KC mastery 的乘积（严格：任一 KC 未掌握则整体低）
  - **mean**: 所有 KC mastery 的均值（宽松）
  - **geo_mean**: 几何平均（折中）

### 结果
- prod 方式使模型对"知识缺口"更敏感，在正确性预测任务上效果最好
- mean 方式在代码生成任务上更稳定
- True/False 加权嵌入显著优于直接数值注入（后者破坏了 LLM 的 token embedding 分布）

### 核心发现

1. **利用 LLM 已有的语义空间比创造新表示更有效**：True/False 是 LLM 预训练时已经充分学习的概念，用它们的加权组合来表示不确定性是一种"借力"策略。

2. **KC mastery 聚合方式反映了不同的教育假设**：prod 假设"所有 KC 都必须掌握"，mean 假设"整体水平重要"，geo_mean 是折中。没有绝对优劣，取决于教育目标。

3. **连续值到离散空间的映射是一个普遍性问题**：这种 True/False 加权方案可迁移到任何需要在 LLM 输入中表示概率/连续值的场景。

### 方法论价值
**高**。"利用 LLM 已有 token embedding 表示连续概率"这一方法可复用于：
- 情感强度的连续表示
- 置信度注入
- 多标签概率的 soft embedding

### 可写入论文？
**是** ✅ — 已作为 KCGen-KT 论文的核心贡献之一

---

## 实验 3：多任务联合训练（Code Generation + Correctness Prediction）

### 研究问题
知识追踪（预测学生是否答对）和代码生成（预测学生会写什么代码）这两个任务能否互相增益？

### 核心假设
代码生成任务提供了丰富的序列级信号，可以帮助模型更深入地理解"知识状态"。反过来，正确性预测任务提供了 binary 监督信号，可以校准代码生成的方向。

### 实验设计
- 代码生成 Loss：标准 cross-entropy（LLM causal LM loss）
- 正确性预测 Loss：BCE / CrossEntropy（从 hidden states 池化后预测）
- KC 识别 Loss：BCE（LSTM 输出的 mastery level 与真实 score 对比）
- 总 Loss 加权：`alpha * (gen_loss + pred_loss) + (1-alpha) * kc_loss`
- Loss 归一化：`loss / loss.detach()` 避免不同 loss 量级的干扰
- Alpha 默认值：0.8（代码生成主导）

### 结果
- 多任务训练显著提升了正确性预测的 AUC（相比单任务 KT baseline）
- 代码生成质量（CodeBLEU）不受多任务训练的负面影响
- Loss 归一化对训练稳定性至关重要——不归一化时，kc_loss 量级远小于 gen_loss，被淹没

### 核心发现

1. **知识追踪和代码生成是同一硬币的两面**：正确性预测是对学生知识状态的"压缩"判断，代码生成是对知识状态的"展开"表达。它们共享底层表示，联合训练是自然的。

2. **Loss 归一化是多任务学习的关键工程实践**：不同任务的 loss 量级可能相差数个数量级。`loss / loss.detach()` 将所有 loss 归一化到 ~1 的量级，使 alpha 权重的含义更直观。

3. **Hidden state 的池化方式影响预测质量**：只用 question 部分的 hidden states（通过 prompt mask 过滤）比用全部 hidden states 更好，因为 answer 部分的 hidden states 包含了"已知答案"的信息泄露。

### 方法论价值
**高**。多任务 loss 归一化和 selective hidden state pooling 适用于所有多任务 LLM 应用。

### 可写入论文？
**是** ✅ — 已写入论文

---

## 实验 4：KC 聚类粒度消融

### 研究问题
自动生成的 KC 经过聚类压缩后，什么粒度最适合下游知识追踪？

### 核心假设
存在一个最优的 KC 数量，在信息保留（细粒度）和统计可靠性（粗粒度，每个 KC 有足够样本）之间取得平衡。

### 实验设计
- 固定 KC 生成参数（5 correct solutions, GPT-4o）
- 变量：n_cluster ∈ {20, 30, 50, 75, 100, 150}
- 聚类方法：层次聚类 (average linkage, cosine distance)
- 评估：下游 KT 模型的 CodeBLEU, Acc, AUC, F1

### 核心发现

1. **KC 粒度不是越细越好**：太细粒度的 KC 导致每个 KC 的训练样本不足，LSTM 无法稳定学习 mastery level。

2. **数据集规模决定了最优 KC 粒度**：CodeWorkout 这种中等规模数据集，~75 个 KC 是最优。更大的数据集可以支撑更细的粒度。

3. **聚类方法本身引入了信息损失**：语义相近但教学意义不同的 KC 可能被错误合并。这提示需要一种"教学感知"的聚类方法，而不是纯语义聚类。

### 方法论价值
**中**。KC 粒度消融的方法论可复用，但具体数值不可迁移。

### 可写入论文？
**是** ✅ — 已作为消融实验写入论文

---

## 实验 5：Transition Layer 消融

### 研究问题
当 LSTM 的隐藏维度与 KC 数量不匹配时，是否需要一个中间线性变换层？

### 核心假设
LSTM hidden dim 和 KC 数量是两个独立的设计选择。当它们不相等时（如 transition_dim=64, KC数=75），transition layer 可以在两者之间建立灵活映射。

### 实验设计
- 无 transition：LSTM hidden dim = KC 数量
- 有 transition：LSTM hidden dim = transition_dim (64)，后接 Linear(64, KC数)

### 核心发现

1. **Transition layer 在 KC 数量较大时有明显收益**：当 KC > 50 时，直接让 LSTM 输出维度等于 KC 数量会导致参数爆炸和过拟合。Transition layer 充当了 bottleneck。

2. **Bottleneck 维度 64 是一个合理默认值**：KC mastery 的"内在维度"远小于 KC 总数，因为很多 KC 的 mastery 是相关的（如掌握了"for loop"的学生通常也掌握了"while loop"）。

### 方法论价值
**中**。Bottleneck 设计在 KT 以外的序列建模任务中也常见。

### 可写入论文？
**是** ✅ — 已作为消融实验写入论文

---

## 实验 6：正确代码采样策略对 KC 质量的影响

### 研究问题
用于 KC 生成的正确代码样本如何选择？随机采样 vs 多样性聚类采样有何区别？

### 核心假设
编程问题通常有多种解法（如迭代 vs 递归、数组 vs 链表）。多样化的代码样本能揭示更全面的 KC 集合。

### 实验设计
- 随机采样：从正确提交中随机选 k 个
- 聚类采样：用 GraphCodeBERT 编码代码 → AgglomerativeClustering → 每簇选一个代表
- 变量：n_correct ∈ {1, 2, 5}

### 核心发现

1. **代码多样性比数量更重要**：聚类采样 5 个代码生成的 KC 质量优于随机采样 5 个，因为随机采样容易选到结构相似的解法。

2. **即使只有 1 个代码也能生成合理 KC**：这说明 LLM 对编程问题有足够的先验知识，代码主要起到"锚定"和"验证"的作用。

3. **GraphCodeBERT 的代码嵌入足够区分不同解法风格**：cosine distance + average linkage 聚类可以有效分离"迭代 vs 递归"、"简洁 vs 啰嗦"等代码风格差异。

### 方法论价值
**高**。"代码多样性采样"方法可迁移到代码审核、代码推荐等任务。

### 可写入论文？
**是** ✅ — 已写入论文
