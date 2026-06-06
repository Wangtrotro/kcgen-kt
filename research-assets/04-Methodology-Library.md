# Methodology Library

> 从项目经验中沉淀的可复用方法论

---

## Method 1: Error-Driven Knowledge Extraction

### 定义
通过系统性地分析失败案例（错误代码、审核失误、Agent 失败等），提取隐含的结构化知识。核心信念：错误比成功更有信息量。

### 适用场景
- 从学生代码中提取 Knowledge Components
- 从 LLM 输出错误中优化 Prompt
- 从 Agent 执行失败中改进 tool design
- 从审核漏判中发现审核标准的盲区

### 步骤
1. **收集失败案例**: 系统性地记录所有失败/次优结果，保留完整上下文
2. **归因分析**: 对每个失败案例标注 root cause（知识缺失?设计缺陷?数据问题?）
3. **模式提取**: 聚类相似的失败模式，提取共性规律
4. **知识编码**: 将发现的规律转化为显式的规则/约束/Prompt/代码
5. **验证闭环**: 在新数据上验证改进是否解决了目标失败模式

### 成功案例
- **KCGen-KT**: 加入 incorrect student submissions 后，KC 生成质量显著提升。错误代码揭示了学生的 misconception，这些 misconception 对应了"容易被忽略的 KC"
- **评论审核 Prompt 优化**: 收集审核失败的评论 → 分析漏判原因 → 在 Prompt 中添加对应的审核维度 → 漏判率下降

### 失败案例
- 当失败模式过于多样化（长尾分布）时，归因分析的成本过高
- 当 root cause 是数据质量问题而非方法问题时，error-driven 方法可能导致过拟合噪声

### 可迁移项目
- 任何涉及 LLM 输出质量优化的项目
- 教育 AI 中的 misconception discovery
- 代码审核中的 bug pattern 识别
- Agent evaluation 中的 failure taxonomy

---

## Method 2: Structured Output Engineering

### 定义
通过严格定义 LLM 输出的结构、格式和约束，将"开放式生成"转变为"约束内生成"。核心信念：缩小输出空间比优化输入更高效。

### 适用场景
- 需要从 LLM 获得可解析、可审计的结构化输出
- 多步骤 pipeline 中，上游输出需要作为下游输入
- 需要保证输出一致性和可复现性的场景

### 步骤
1. **定义输出 Schema**: 明确每个字段的名称、类型、取值范围和语义
2. **选择约束机制**: 
   - OpenAI: `response_format={"type": "json_object"}` 
   - 或在 system prompt 中嵌入 JSON template
3. **添加 reasoning 字段**: 要求 LLM 先 reasoning 再 output，提高输出质量
4. **验证与修复**: 对输出做 schema validation，对不合规输出做 retry 或 fallback
5. **迭代 Schema**: 根据下游任务反馈调整 schema 设计

### 成功案例
- **KC 生成**: JSON 格式包含 `name` 和 `reasoning` 字段，确保每个 KC 有可追溯的理由
  ```json
  {
    "KC 1": {"reasoning": "...", "name": "Conditional Logic"},
    "KC 2": {"reasoning": "...", "name": "Array Indexing"}
  }
  ```
- **KC 聚类总结**: 要求 LLM 输出 `representative_kc` 或 `summary_name`（二选一），用 null 标记未选的选项，消除歧义

### 失败案例
- 过于严格的 schema 约束可能限制 LLM 表达有价值但超出 schema 的洞察
- JSON schema 对于需要长文本推理的任务不合适

### 可迁移项目
- 所有需要 LLM 输出结构化数据的场景
- Agent 的 tool use 设计（定义 tool 的输入输出 schema）
- 教育评估的评分标准输出

---

## Method 3: Semantic Clustering Pipeline for Knowledge Deduplication

### 定义
用语义嵌入 + 层次聚类 + LLM 摘要的三阶段管线，将大量语义相近但措辞不同的文本条目去重并归纳为代表性条目。

### 适用场景
- 大规模文本分类/标签体系的构建和精简
- 从多个来源收集的知识条目的去重
- 开放式 LLM 输出（如 KC、标签、摘要）的后处理

### 步骤
1. **语义编码**: 用 SentenceTransformer（如 `all-MiniLM-L6-v2`）将每个条目编码为向量
2. **距离计算**: 计算 pairwise cosine distance
3. **层次聚类**: 用 scipy linkage (average method)，设定目标簇数 n_cluster
4. **簇内摘要**: 对每个簇，用 LLM 选择代表性条目或生成摘要名称
5. **映射回溯**: 建立原始条目 → 代表性条目的映射字典

### 成功案例
- **KC 去重**: 将 ~150 个原始 KC 压缩到 75 个，保留了语义多样性的同时消除了冗余。如 "Using if statements" 和 "Implementing conditional logic" 被合并为 "Conditional Logic and Evaluation"

### 失败案例
- 语义相近但教学意义不同的 KC 可能被错误合并（如 "for loop" 和 "while loop"）
- 当目标簇数设得太小时，过度合并导致信息损失

### 可迁移项目
- 教育内容标签体系的构建
- 文献调研中的研究主题聚类
- 用户反馈/评论的主题提取
- Agent 日志的 error pattern 聚类

---

## Method 4: LLM Semantic Space Injection

### 定义
利用 LLM 预训练已学习的 token embedding（如 True/False、Yes/No）的语义对立性，通过加权线性组合来在 LLM 输入空间中表示连续概率值。

### 适用场景
- 需要在 LLM 输入中注入连续值信息（如概率、置信度、强度）
- LLM 的 token embedding 空间中存在语义对立的 anchor token pair
- 直接数值注入破坏了 LLM embedding 分布的场景

### 步骤
1. **选择 Anchor Token Pair**: 找到语义对立的 token pair（如 True/False, Yes/No）
2. **获取嵌入**: `true_emb = model.embed_tokens(tokenize("True"))`, 同理 false_emb
3. **加权组合**: `injected_emb = p * true_emb + (1-p) * false_emb`
4. **替换占位符**: 在输入序列中将特定占位符 token 的嵌入替换为 injected_emb
5. **端到端训练**: 让梯度从 LLM loss 回传到 p 的生成模块

### 成功案例
- **KC Mastery Level 注入**: LSTM 输出的 KC mastery level (0-1) 通过 True/False 加权嵌入注入 LLaMA-3-8B，模型能据此调整代码生成的内容

### 失败案例
- 尚未在 True/False 之外的 anchor pair 上验证
- 当 p 接近 0.5 时，嵌入可能落入 LLM embedding 空间的"无人区"

### 可迁移项目
- 情感分析中的强度注入（positive/negative 的加权）
- 多标签分类的 soft label 注入
- 不确定性感知的 LLM 生成

---

## Method 5: Multi-task Loss Normalization

### 定义
在多任务学习中，对每个任务的 loss 做 `loss / loss.detach()` 归一化，使不同量级的 loss 在同一尺度上参与加权平衡。

### 适用场景
- 多任务学习中不同任务 loss 量级相差数个数量级
- 需要通过超参数 alpha 来精确控制任务间的权重
- 训练不稳定，某个 loss 主导梯度方向的场景

### 步骤
1. **计算各任务 loss**: L_1, L_2, ..., L_n
2. **归一化**: `L_i_norm = L_i / L_i.detach()`，使每个归一化后的 loss ≈ 1.0
3. **加权求和**: `total_loss = α * (L_1_norm + L_2_norm) + (1-α) * L_3_norm`
4. **反向传播**: 对 total_loss 做 backward

### 成功案例
- **KCGen-KT**: 代码生成 loss (~2.0) + KC loss (~0.01) + 预测 loss (~0.5)，不归一化时 KC loss 完全被淹没。归一化后 alpha=0.8 有明确的物理含义："80% 的梯度来自生成和预测，20% 来自 KC 识别"

### 失败案例
- 当 loss 值接近 0 时，`loss.detach()` 接近 0 导致归一化后的值爆炸。需要加 epsilon: `loss / (loss.detach() + eps)`
- 过度归一化可能导致训练初期（loss 很大）和后期（loss 很小）的梯度量级相同，无法自然衰减

### 可迁移项目
- 任何多任务 LLM 训练
- 多目标优化问题
- GAN 训练中 G/D loss 的平衡

---

## Method 6: Diversity-Aware Sampling for Few-shot Learning

### 定义
在构建 few-shot 示例集时，先对候选示例做聚类，然后从每个簇中选取代表性示例，确保示例集的多样性最大化。

### 适用场景
- LLM few-shot learning 的示例选择
- 训练数据的代表性子集选择
- 代码审核/评估的 benchmark 构建

### 步骤
1. **候选池构建**: 收集所有可能作为 few-shot 示例的候选
2. **语义编码**: 用合适的 encoder（文本用 SentenceTransformer，代码用 GraphCodeBERT）编码候选
3. **聚类**: AgglomerativeClustering 或 HDBSCAN，设定目标簇数 = 目标示例数
4. **代表选择**: 从每个簇中选择最接近簇心的候选作为代表
5. **人工检查**: 确认选出的示例在语义上确实覆盖了不同类型

### 成功案例
- **KC 生成**: 选择 1 个条件判断类 + 1 个数组操作类的 KC 标注示例，比随机选择效果更好
- **代码采样**: 通过 GraphCodeBERT + 聚类从正确代码中采样多样化解法

### 失败案例
- 当候选池本身就缺乏多样性时，聚类无法创造多样性
- 聚类的质量依赖 encoder 的表示能力

### 可迁移项目
- Eval 数据集的构建
- RAG 中的检索结果多样性优化
- Active Learning 的样本选择

---

## 方法论依赖关系图

```
Error-Driven Knowledge Extraction
    ├── 输入给 → Structured Output Engineering（编码提取结果）
    ├── 结合 → Semantic Clustering Pipeline（聚类错误模式）
    └── 驱动 → Diversity-Aware Sampling（选择多样化错误案例）

Structured Output Engineering
    └── 输出给 → Semantic Clustering Pipeline（结构化数据做聚类）

LLM Semantic Space Injection
    └── 配合 → Multi-task Loss Normalization（多任务训练中使用）

Diversity-Aware Sampling
    └── 依赖 → Semantic Clustering Pipeline（聚类作为采样基础）
```
