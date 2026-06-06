# 路径感知认知状态驱动的编程题推荐：仓库现状与扩展路径分析

## 1. 仓库现状总结

本仓库实现了论文 *Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems* (arXiv:2502.18632) 的核心系统，包含两个主要管线：

### 1.1 KC 自动生成管线 (`KC_gen.py`)

| 步骤 | 实现方式 | 对应函数 |
|------|----------|----------|
| 样本解法获取 | 从学生正确提交中聚类采样 **或** GPT-4o-mini 生成 | `student_solution_construct()`, `get_sample_solution()` |
| 代码聚类/策略发现 | GraphCodeBERT embedding + Agglomerative Clustering | `find_solution_by_cluster()` |
| KC 抽取 | GPT-4o few-shot prompting（从解法代码中推断 KC） | `get_KCs_few_shot()`, `get_KCs_few_shot_with_inc()` |
| KC 聚类/去重 | SentenceTransformer + scipy linkage 层次聚类 | `KC_cluster_linkage()` |
| KC 摘要命名 | GPT-4o 对聚类内 KC 做代表性命名 | `KC_cluster_summarize()` |

**产出**: `problem_kc.json` — 每道题映射到一组 `[KC描述, KC类别]` 对。

### 1.2 知识追踪管线 (`main_kc_okt.py` + `trainer.py`)

```
学生做题序列 (prompt embedding 4096d + ASTNN embedding 200d = 4296d)
    │
    ▼
LSTM(4296 → |KC|)  →  sigmoid  →  每个 KC 的掌握概率
    │
    ▼
将 prompt 中 "?" token 的 embedding 替换为 mastery-weighted True/False embedding
    │
    ▼
Llama-3-8B-Instruct (8-bit + LoRA r=128)
    │
    ├─→ 代码生成 (next-token prediction loss)
    ├─→ KC mastery loss (BCE: predicted mastery product vs actual score)
    └─→ 正确性预测 (BCE: hidden state → linear → binary)
```

### 1.3 数据集

- **CodeWorkout / CSEDM**: 大学 Java 编程提交，公开可获取
- 数据包含: `SubjectID`, `ProblemID`, `prompt`, `Code`, `Score`, `ServerTimestamp`, `prompt-embedding`, `input`(4296d)

---

## 2. 与"路径感知认知状态驱动推荐"研究方向的对接分析

### 2.1 仓库中已具备的基础设施

| 研究报告中的环节 | 仓库现有对应 | 完备程度 |
|-----------------|-------------|----------|
| **代码提交** | `data_loader.py` 加载 `dataset_time.pkl`，含完整提交序列 | ✅ 完整 |
| **解题路径识别** | `find_solution_by_cluster()` 用 GraphCodeBERT + 聚类 | ⚠️ 仅用于 KC 生成时的参考解采样，未用于运行时学生提交分类 |
| **KC 生成/标注** | `KC_gen.py` 全自动 LLM 管线 | ✅ 完整（题目级） |
| **KC 掌握追踪** | LSTM → sigmoid → per-KC mastery | ✅ 完整（题目级激活） |
| **路径级 KC 激活** | 无 — 当前所有 KC 对同题统一激活 | ❌ 缺失 |
| **认知图扩散** | 无 — LSTM 输出是平面向量 | ❌ 缺失 |
| **编程题推荐** | 无 — 系统仅做 KT + 代码生成 | ❌ 缺失 |

### 2.2 关键发现：已有"路径识别"的原型代码

`KC_gen.py` 第 430-483 行的 `find_solution_by_cluster()` 函数已经实现了：

```python
def find_solution_by_cluster(model_path, code_ls, k, file_path):
    # GraphCodeBERT mean-pooling → normalize → AgglomerativeClustering(cosine)
    embeddings_norm = np.load(f"code_embedding/embeddings_problem_{file_path}.npy")
    clusterer = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
    labels = clusterer.fit_predict(embeddings_norm)
    # 每个簇采样一个代表解
    ...
```

同时 `student_solution_construct()` (第 486-511 行) 根据 `random=False` 参数选择是随机采样还是聚类采样：

```python
def student_solution_construct(df, k, random=True):
    if random:
        solution_dict = df.groupby('prompt').apply(lambda x: x.sample(n=k)['Code'].tolist())
    else:
        # cluster correct codes by semantic and structure
        selected_sol = find_solution_by_cluster("microsoft/graphcodebert-base", code_ls, k, ...)
```

**这说明仓库作者已经意识到"同一道题存在多种不同策略的解法"，并且已经用 GraphCodeBERT + 聚类来发现它们——但目前这个能力仅被用于 KC 生成阶段的多样化采样，而非运行时的学生路径识别。**

### 2.3 从现有代码到"路径感知"的具体差距

#### 差距 1: 路径识别仅在离线 KC 生成时使用

当前流程：
```
离线: 正确解聚类 → 采样多样解法 → LLM 生成 KC → problem_kc.json
运行时: 学生提交 → 直接按题目查 KC → 全部 KC 统一激活
```

需要变为：
```
离线: 正确解聚类 → 建立每题的 path prototype bank
运行时: 学生提交 → 与 path prototypes 比对 → 识别所走路径 → 仅激活该路径相关 KC
```

#### 差距 2: KC 映射是题目级而非路径级

`problem_kc.json` 的结构是 `{problem_text: [[kc_desc, category], ...]}` — 每道题一组 KC，所有学生无论用哪种解法都激活同一组。

需要变为类似：
```json
{
  "problem_text": {
    "path_0": ["KC_A", "KC_B", "KC_C"],
    "path_1": ["KC_A", "KC_D", "KC_E"],
    "path_2": ["KC_B", "KC_F"]
  }
}
```

#### 差距 3: LSTM 状态更新无图结构

当前 `predict_mastery_level()` (`trainer.py` 第 231-248 行):
```python
def predict_mastery_level(padded_inputs, padded_kc, lstm, trans, trans_linear):
    ks, hidden = lstm(padded_inputs)  # 平面序列建模
    ks = torch.sigmoid(ks)            # 独立 KC sigmoid
    result_kc = torch.gather(ks, 2, safe_indices)  # 按题目取对应 KC
```

- 没有 KC 之间的图传播
- 没有 KC 之间的先决/共现关系建模
- 每个 KC 的状态独立更新

#### 差距 4: 无推荐模块

当前系统输出是：
- KC 掌握向量 (用于控制代码生成)
- 正确性预测 (二分类)
- 生成代码 (CodeBLEU 评估)

没有"给定当前状态，选择下一道最合适的题"这一推荐决策。

---

## 3. 扩展路线图：从 kcgen-kt 到 SP-CogKT + Recommendation

### Phase 1: 路径识别器 (Path Recognizer)

**目标**: 将 `find_solution_by_cluster` 的离线能力扩展为运行时学生提交分类器。

**基于仓库的实现路径**:

1. **建立 Path Prototype Bank**
   - 复用 `find_solution_by_cluster()` 对每道题的正确提交做 k-means/层次聚类
   - 保存每个簇的质心 embedding 作为 path prototype
   - 用 LLM (复用 `get_KCs_few_shot()` 的 prompt 框架) 为每个簇生成语义标签

2. **运行时路径分类**
   - 新学生提交 → GraphCodeBERT/CodeBERT embedding → 与 prototype cosine 相似度 → soft assignment
   - 对错误提交：用"最近正确簇 + 编辑距离"做 soft label
   - 输出: `path_posterior ∈ R^{K_paths}` (每条路径的概率)

3. **与现有 data_loader.py 的集成点**
   - 在 `make_pytorch_dataset()` 中新增 `'path_id'` 字段
   - 在 `CollateForKC.__call__()` 中新增路径信息的 padding 与 batching

### Phase 2: 路径级 KC 激活 (Path-conditioned KC Activation)

**目标**: 让 Q-matrix 从 `problem → KC_set` 变为 `(problem, path) → KC_subset`。

**基于仓库的实现路径**:

1. **扩展 `problem_kc.json` 的结构**
   - 方案 A (硬): 对每条路径独立调用 `get_KCs_few_shot()`，只喂该路径簇内的代表解
   - 方案 B (软): 保持现有 problem-level KC，新增一个 `path_kc_weight` 矩阵，表示每条路径对每个 KC 的激活权重

2. **修改 `trainer.py` 中的 `update_input_weight()`**
   
   当前逻辑 (第 252-319 行):
   ```python
   result_kc, kc = predict_mastery_level(padded_inputs, padded_kc, lstm, ...)
   # result_kc shape: (B, T, max_kc_len) — 所有 KC 统一激活
   ```
   
   需要变为:
   ```python
   result_kc, kc = predict_mastery_level(padded_inputs, padded_kc, lstm, ...)
   path_weight = path_recognizer(current_code_embedding)  # (B, T, K_paths)
   path_kc_activation = torch.einsum('btk,kc->btc', path_weight, path_kc_matrix)
   result_kc = result_kc * path_kc_activation  # 路径条件化激活
   ```

### Phase 3: 认知图扩散 (Cognitive Graph Propagation)

**目标**: 替代平面 LSTM，在 KC 图上做状态传播。

**基于仓库的实现路径**:

1. **构建 KC 关系图**
   - 利用现有 `KC_cluster_linkage()` 的层次聚类树，提取 KC 之间的语义邻近关系
   - 从学生做题序列中统计 KC 共现/先决关系
   - 图节点 = KC，边权 = 语义相似度 + 共现频率

2. **GNN 状态传播模块**
   - 在 LSTM 输出 KC mastery 后，加一层 GAT/GCN 在 KC 图上扩散
   - 修改 `model.py` 中的模型工厂，新增 `create_cognitive_graph()` 函数
   - 保留 LSTM 的序列建模能力，GNN 仅负责 KC 间的关系推断

3. **与现有训练流程的集成**
   - `main_kc_okt.py` 中增加 GNN optimizer
   - `trainer.py` 中的 `predict_mastery_level()` 输出后接 GNN propagation

### Phase 4: 编程题推荐 (Exercise Recommendation)

**目标**: 基于路径感知认知状态，推荐下一道最优题目。

**基于仓库的实现路径**:

1. **候选题表征**
   - 复用 `problem_kc.json` + path-level KC structure
   - 每题表征 = KC 覆盖向量 + 难度 + 路径多样性

2. **推荐策略**
   - 输入: 学生当前 path-aware mastery state (来自 Phase 2-3)
   - 输出: 候选题排序 (mastery gap matching / RL-based selection)
   - 最简版: 选择能最大化"薄弱路径相关 KC"覆盖的题目

3. **评估**
   - Replay-style next-item ranking (Recall@K, NDCG@K)
   - 后续 KC 表现提升 (learning gain proxy)

---

## 4. 仓库代码中可直接复用的组件

| 组件 | 文件 | 可复用方向 |
|------|------|-----------|
| GraphCodeBERT 代码聚类 | `KC_gen.py:find_solution_by_cluster()` | → Path Recognizer 的核心 |
| LLM KC 生成 prompt | `KC_gen.py:get_KCs_few_shot()` | → 路径级 KC 标注 |
| KC 层次聚类 + 摘要 | `KC_gen.py:KC_cluster_linkage()` + `KC_summarize()` | → KC 关系图构建 |
| LSTM 掌握追踪 | `trainer.py:predict_mastery_level()` | → 基础 KT baseline + 状态初始化 |
| Mastery → Token embedding 注入 | `trainer.py:update_input_weight()` | → 可扩展为路径条件化注入 |
| 时间排序数据划分 | `data_loader.py:read_data()` | → 保证时间不泄漏的实验设计 |
| CodeBLEU 评估 | `evaluator/CodeBLEU/` | → 代码生成质量的辅助指标 |
| 多任务训练框架 | `trainer.py:generator_step()` | → 可扩展为 KT + 推荐联合训练 |

---

## 5. 最小可行实验 (MVP) 设计

### 5.1 基于本仓库的最短路径实验

**假设**：仅使用 CodeWorkout 数据和仓库现有代码，验证"路径变量是否有用"。

**Step 1**: 路径聚类（复用 `find_solution_by_cluster`）
```
对每道题的所有正确提交 → GraphCodeBERT embedding → k 簇 (k=3~5)
保存: path_prototypes.pkl, student_path_assignments.pkl
```

**Step 2**: 路径级 KC 标注（复用 `get_KCs_few_shot`）
```
对每个簇的代表代码单独调用 KC 生成 → path_problem_kc.json
对比: 不同路径簇是否产生不同的 KC 子集
```

**Step 3**: 路径条件化 KT（修改 `trainer.py`）
```
修改 update_input_weight():
  - 根据学生当前提交的 path_id，仅激活对应路径的 KC 子集
  - 对比: 全 KC 激活 vs 路径条件化激活 在 AUC/ACC 上的差异
```

**Step 4**: 核心消融 — real path vs random path
```
保持模型结构完全一致，仅将 path_id 在同题内随机打乱
如果 real path > random path: 证明路径身份本身携带诊断信号
```

### 5.2 预期结果与论文叙事

如果 Step 3-4 成功:
- 主贡献: "路径级 KC 激活改善编程知识追踪" → 投 EDM/LAK/AIED
- 如果再加推荐模块: "路径感知认知状态驱动编程题推荐" → 投 KBS/Recsys/CIKM

---

## 6. 风险评估与仓库特有约束

### 6.1 数据限制

- `dataset_time.pkl` 需要下载（Google Drive），且仅含 CodeWorkout 数据
- 公开数据中没有路径真值标签 — 需要自行通过聚类构造
- 首次提交筛选 (`first_ast_convertible: true`) 会丢失多次提交信息

### 6.2 模型限制

- 当前系统重度依赖 GPU (Llama-3-8B + 8-bit)，路径识别实验可先不涉及 LLM 代码生成
- LSTM hidden dim = |KC| (约 75)，如果 KC 数大幅增加需要重调架构
- `transition` 模块 (linear projection) 可以缓解维度问题

### 6.3 实验设计注意事项

1. **时间排序已保证** — `read_data()` 中 `sort_values(by=['SubjectID', 'ServerTimestamp'])` ✅
2. **路径聚类必须只用训练集** — 当前 `find_solution_by_cluster` 未做 train/test 隔离，需要修改
3. **KC 生成不能用测试集统计** — 当前 KC 生成是离线的，但需确认用的是哪些学生的代码
4. **`first_ast_convertible=True` 意味着每人每题仅保留首次提交** — 这对路径研究可能是限制，因为学生可能在多次提交中切换策略

---

## 7. 建议的开题定位（结合仓库）

### 核心叙事

> 本仓库已实现"LLM 自动 KC 生成 + KC 掌握追踪"的完整管线，并且在 KC 生成阶段已经利用代码聚类发现了同题多策略现象。但这一"策略/路径"信息在运行时被丢弃——所有学生无论用哪种解法，激活的 KC 集合都相同。
> 
> 本研究的核心扩展是：**将离线 KC 生成时已发现的"同题多路径"能力，转化为运行时的路径感知认知状态估计，并验证该状态对 KT 预测和编程题推荐的增益。**

### 创新点与仓库的关系

| 创新点 | 在仓库中的定位 |
|--------|--------------|
| 路径级 KC 激活 | 从 `problem_kc.json` 的题目级 → `path_problem_kc.json` 的路径级 |
| 运行时路径识别 | 将 `find_solution_by_cluster` 从离线工具变为在线分类器 |
| 认知图扩散 | 在 LSTM 输出后新增 GNN 层，利用 KC 聚类树构建关系图 |
| 推荐模块 | 在现有多任务框架上新增推荐 head |

### 最关键的实验设计

1. **same-problem, different-path 分析**: 证明同题不同路径学生在后续表现上有显著差异
2. **real path vs random path**: 证明路径身份本身携带不可替代的诊断信号  
3. **path-aware state vs item-level state**: 控制住所有其他变量，仅改变 KC 激活方式
4. **oracle path vs predicted path**: 量化路径识别误差对下游的影响上界

---

## 8. 总结

本仓库是一个非常好的起点：

- ✅ 已有完整的 LLM-based KC 生成管线
- ✅ 已有代码聚类/策略发现的原型代码 (`find_solution_by_cluster`)
- ✅ 已有 LSTM-based KT + 多任务训练框架
- ✅ 使用 CodeWorkout 数据集（文献中公认的 PKT benchmark）
- ✅ 已做时间排序、学生级划分等基本实验规范

需要新增的核心模块：
- ❌ 运行时路径分类器
- ❌ 路径级 KC 激活矩阵 (`path × KC` weight matrix)
- ❌ KC 关系图 + GNN 传播
- ❌ 推荐模块

从工程量看，**Phase 1-2 (路径识别 + 路径级 KC 激活) 可以在现有代码基础上相对快速实现**，因为核心组件（代码聚类、KC 生成 prompt、LSTM KT）都已存在。Phase 3 (认知图) 和 Phase 4 (推荐) 是增量扩展，可以根据 Phase 1-2 的实验结果决定是否推进。

**建议优先验证的假设**：同题不同路径的学生，其后续 KC 表现轨迹是否存在统计显著差异。如果这个前提成立，后续所有模块才有意义。
