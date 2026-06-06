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
**是** ✅ — 已写入论文（多任务训练部分）

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

---

## 实验 7：Baseline KC 对比 — 人工粗粒度 vs 自动细粒度

### 研究问题
人工标注的 18 类粗粒度 KC（如 "If/Else"、"For"、"ArrayIndex"）在下游 KT 任务上表现如何？自动生成的细粒度 KC 能否超越？

### 核心假设
人工标注的 KC 粒度太粗，无法区分同一大类下的不同子技能（如 "String indexing" vs "String length" 都被归为 "String" 类），导致 LSTM 的 mastery level 预测失去分辨力。

### 实验设计
- **Baseline KC**: `prompt_concept.csv` 中的 18 列 binary 标注（If/Else, NestedIf, While, For, NestedFor, Math+-*/, Math%, LogicAndNotOr, LogicCompareNum, LogicBoolean, StringFormat, StringConcat, StringIndex, StringLen, StringEqual, CharEqual, ArrayIndex, DefFunction）
- **自动 KC**: `problem_kc.json` 中 GPT-4o 生成的细粒度 KC（聚类后归为更高层类别如 "Conditional Logic and Evaluation", "String Manipulation Techniques", "Array Management in Java" 等）
- 训练配置切换: `configs.baseline = True/False`
- 控制变量: 模型结构（LSTM + LLaMA）、训练参数完全相同

### 结果
- 自动 KC 在正确性预测和代码生成上均优于 baseline
- 18 类 KC 中存在大量 co-occurrence（如 "If/Else" 和 "LogicCompareNum" 几乎总是同时出现），导致 LSTM 难以学到独立的 mastery signal

### 核心发现

1. **人工 KC 的粒度问题不是"不够细"，而是"维度冗余"**: 18 个 KC 中有些几乎完全相关（如条件判断相关的 4 个 KC），实际有效维度远小于 18，反而不如 75 个自动 KC 的信息量大。

2. **KC 的定义来源决定了其上限**: 人工 KC 来自"教材知识点列表"，自动 KC 来自"代码结构分析"。前者反映的是"应该教什么"，后者反映的是"解题实际需要什么"。这两者之间的差距就是教育理论与教学实践的鸿沟。

3. **Baseline 实验的真正价值不是证明"我们更好"，而是揭示"人工标注的失效模式"**: 通过分析 baseline 失败的 case，可以反向发现自动 KC 的关键优势在哪里。

### 方法论价值
**中高**。任何需要对比"自动 vs 人工"的实验，都应关注"人工方法的失效模式"而非仅报告指标差异。

### 可写入论文？
**是** ✅ — 已作为 baseline 对比写入论文

---

## 实验 8：Prompt 构造实验 — KC-aware Input Template 设计

### 研究问题
如何设计输入 prompt 模板，使得 KC 信息和 mastery level 能有效地与问题描述和代码混合？

### 核心假设
prompt 的结构化设计（KC 信息的位置、mastery level 占位符的放置方式）直接影响 LLM 对 KC mastery 的利用程度。

### 实验设计
最终采用的 prompt 模板:
```
Question: {问题描述}
 KC 1: {KC名称}. The student's mastery level on {KC名称} is ?
 KC 2: {KC名称}. The student's mastery level on {KC名称} is ?
...
 Student written code: {代码}
```

关键设计决策:
- `?` 作为 mastery level 占位符，在嵌入层被 True/False 加权替换
- KC 信息放在 question 和 code 之间，作为"桥梁"
- 使用 `written` token 作为 question/code 的分界标记（`delimiter_token_id`）
- 使用 `?` token 定位 mastery level 插入位置（`level_token_id`）

### 核心发现

1. **Token-level 的定位机制是必要的工程实践**: 不能靠字符串匹配找到 `?` 的位置，必须在 token id 层面精确定位，因为 tokenizer 可能将 `?` 与前后字符合并。

2. **KC 信息的位置影响 attention 分布**: 将 KC 放在 question 后、code 前，使 LLM 在生成代码时能同时 attend 到 KC mastery 和 question context。

3. **Prompt 构造中的特殊字符处理不可忽视**: 原始问题描述中的 `:` 和 `?` 需要替换，否则会干扰 KC mastery 占位符的定位。

### 方法论价值
**中**。这类 token-level prompt engineering 的经验在 LLM 应用中普遍适用，但具体模板不可直接迁移。

### 可写入论文？
**部分** — 作为方法论实现细节

---

## 实验 9：预测器架构消融 — 单层 vs 多层 Predictor

### 研究问题
从 LLM hidden states 预测学生正确性时，简单线性层 vs 多层 MLP 哪个更好？

### 核心假设
LLM 最后一层 hidden states 已经编码了足够丰富的语义信息，简单线性层可能足够。但更深的 predictor 可能捕捉到非线性模式。

### 实验设计
- **单层**: `Linear(4096, 1)` + Xavier 初始化
- **多层**: `Linear(4096, 512) → ReLU → Linear(512, 64) → ReLU → Linear(64, 1)` + Kaiming 初始化
- Pooling 策略: question-only mean pooling（用 prompt mask 过滤掉 answer 部分的 hidden states）
- 配置开关: `configs.predictor_multilayer`

### 核心发现

1. **多层 MLP 在数据充足时优于单层线性**: Kaiming 初始化 + 逐层降维的 MLP 能学到更丰富的从 hidden state 到 correctness 的映射。

2. **Question-only pooling 是关键**: 只对 question embedding 做 mean pooling（mask 掉 answer 部分），避免了信息泄露。

3. **Predictor 的初始化策略影响训练稳定性**: Xavier (单层) vs Kaiming (多层+ReLU) 的选择不是随意的——Kaiming 专为 ReLU 网络设计，避免了梯度消失。

### 方法论价值
**中**。Hidden state pooling + predictor 设计是多任务 LLM 的通用子问题。

### 可写入论文？
**部分** — 作为消融实验

---

## 实验 10：Binary Loss Function 消融 — BCE vs CrossEntropy

### 研究问题
正确性预测任务应该使用 BCEWithLogitsLoss 还是 CrossEntropyLoss？

### 核心假设
BCE 将正确性视为回归问题（预测一个概率值），CrossEntropy 将其视为 2-class 分类问题。两种 framing 可能导致不同的预测行为。

### 实验设计
- **BCE**: `BCEWithLogitsLoss` → `Linear(4096, 1)` → sigmoid → threshold 0.5
- **CrossEntropy**: `CrossEntropyLoss` → `Linear(4096, 2)` → softmax → argmax
- 配置开关: `configs.binary_loss_fn = 'BCE'` 或其他值

### 核心发现

1. **BCE 更自然**: 正确性本身就是一个连续概率（"几乎正确"vs"完全错误"），BCE 的输出（sigmoid 值）可以直接解释为 correctness probability。

2. **CrossEntropy 的 2-class 设计引入了不必要的参数**: 输出维度从 1 变成 2，多了一倍的 predictor 参数，但信息量相同。

3. **两者在 AUC 上差异不大，但在 calibration 上 BCE 更好**: BCE 的 sigmoid 输出直接可用于下游的 KC mastery 估计，不需要额外的 temperature scaling。

### 方法论价值
**低**。这是一个常见的工程选择，结论并不意外。

### 可写入论文？
**否** — 差异不显著，不值得独立报告

---

## 实验 11：跨数据集验证 — CodeWorkout (Java) vs Falcon (Python)

### 研究问题
KCGen-KT 的方法是否具有跨数据集、跨编程语言的泛化能力？

### 核心假设
KC 生成和 KT 的方法不应依赖于特定数据集或编程语言。如果方法具有通用性，应该在 Java (CodeWorkout) 和 Python (Falcon) 数据集上都有效。

### 实验设计
- **CodeWorkout**: Java 编程题，18 类人工 KC baseline，CSEDM 学生数据
- **Falcon**: Python 编程题，不同的人工 KC 标注方案（`problems_falcon_4.csv`），few-shot 示例需换为 Python 题目
- KC 生成时的 few-shot 示例需要按语言分别选择
- 评估时的 CodeBLEU 需要设置 `lang='java'` 或 `lang='python'`

### 核心发现

1. **方法框架通用，但 KC 生成的 prompt 需要语言适配**: system prompt 中的语言描述、few-shot 示例的选择都需要根据目标语言调整。这不是一个根本性限制，但说明完全 zero-shot 的跨语言 KC 生成尚不成熟。

2. **Python 代码通常比 Java 更短，KC 粒度需要相应调整**: Python 的简洁语法使得同一问题产生的 KC 数量较少，聚类数也应相应减少。

3. **CodeBLEU 的语言依赖性**: AST 匹配和数据流匹配依赖于语言解析器（tree-sitter），跨语言评估需要切换解析器。

### 方法论价值
**高**。跨数据集验证是可信度的关键来源。

### 可写入论文？
**是** ✅ — 已作为泛化性实验写入论文
