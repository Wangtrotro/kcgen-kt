# Research Findings

> 跨实验归纳：哪些实验实际上验证了同一个研究假设？

---

## 发现 1：错误案例驱动优于经验驱动的知识提取

**结论**：在从数据中提取结构化知识时，分析"失败案例"比分析"成功案例"更有信息量。这是一条跨项目、跨领域的通用原则。

**证据来源**：

| 实验 | 场景 | 具体表现 |
|------|------|---------|
| KC 生成实验（方案 B vs A） | Knowledge Tracing | 加入错误代码后，LLM 生成的 KC 能捕捉"学生典型知识缺口"，而非仅仅列举"解题所需技能" |
| 评论审核 Prompt 优化 | Education AI | 用失败审核案例驱动 Prompt 改进，比凭经验修改 Prompt 更高效 |
| Agent 开发 Error-driven Iteration | Agent Engineering | Agent 的改进方向应由失败 case 的归因结果决定，而非由开发者直觉决定 |

**核心洞察**：
- 成功案例告诉你"什么是对的"，但错误案例告诉你"什么是关键的"
- 在教育场景中，这对应于从"知识清单"到"知识缺口清单"的转变
- 在 Prompt Engineering 中，这对应于从"正面示例"到"反面示例"的驱动力转变

**可信度**: **高** — 在 3 个独立场景中一致验证

**可论文化方向**：可写为 position paper "Error-Driven Knowledge Discovery in Educational AI"

---

## 发现 2：结构化约束比优化措辞更重要

**结论**：在使用 LLM 处理教育场景任务时，定义清晰的输出结构和评估维度，比反复优化 Prompt 的措辞更能提升系统性能。

**证据来源**：

| 实验 | 场景 | 具体表现 |
|------|------|---------|
| KC 生成 JSON Schema | Knowledge Tracing | 使用 `response_format={"type": "json_object"}` + 明确的 JSON template 后，KC 输出的格式合规率从 ~80% 提升到 ~99% |
| KC 聚类总结 | Knowledge Tracing | 给 LLM 明确的"选择代表性 KC 还是总结"的二选一结构，消除了模糊输出 |
| 评论审核结构化 | Education AI | 定义具体审核维度（事实性、建设性、专业性）比优化"请认真审核"的措辞有效得多 |

**核心洞察**：
- LLM 的"不稳定性"很大程度上来自输出空间的模糊。缩小输出空间比优化输入更高效
- Structured Output 不仅是一个工程选择，更是一种"约束即规范"的认识论
- 这与 Agent Engineering 中的 Tool Use 设计理念一致：清晰定义 tool 的输入输出 schema 比写更好的 system prompt 更重要

**可信度**: **高** — 在多个 LLM 应用场景中验证

**可论文化方向**：可融入 Prompt Engineering 最佳实践综述

---

## 发现 3：知识追踪本质是条件生成问题

**结论**：传统知识追踪将任务定义为 P(correct | history)，但编程知识追踪更自然的定义是 P(code | knowledge_state)。知识追踪和代码生成不是两个独立任务，而是同一问题的两个视角。

**证据来源**：

| 实验 | 具体表现 |
|------|---------|
| 多任务联合训练 | 代码生成和正确性预测共享 hidden representation，联合训练带来双向增益 |
| Mastery Level 嵌入 | KC mastery 作为 LLM 的输入条件，直接影响生成代码的内容和风格 |
| 评估指标设计 | CodeBLEU 和 Acc/AUC 的同时评估揭示了"生成质量"和"预测准确性"的关联性 |

**核心洞察**：
- 这一发现重新定义了 KT 的任务边界：KT 不只是预测"对错"，更是建模"学生会怎么做"
- 这为 KT 领域引入了 generative AI 的视角，可能催生一类新的"Generative Knowledge Tracing"方法
- 实际应用价值：基于 knowledge state 生成个性化代码 hint / scaffolding

**可信度**: **高** — 有实验数据支撑，且与教育学理论（Zone of Proximal Development）一致

**可论文化方向**：可写为专题论文 "Generative Knowledge Tracing: Bridging Knowledge Assessment and Code Generation"

---

## 发现 4：KC 粒度应该由数据驱动而非专家决定

**结论**：知识组件的最佳粒度不是一个固定的教育学问题，而是一个依赖于数据集规模和下游任务的统计问题。

**证据来源**：

| 实验 | 具体表现 |
|------|---------|
| KC 聚类粒度消融 | 不同 n_cluster 值在下游 KT 任务上性能差异显著，75 最优 |
| Baseline 对比（18 KC） | 人工标注的 18 个粗粒度 KC 在大数据集上表现不佳 |
| Transition Layer 消融 | 当 KC 数量与 LSTM 容量不匹配时需要 bottleneck，说明存在"内在 KC 维度" |

**核心洞察**：
- 传统教育学中 KC 粒度由专家定义（如 Bloom's Taxonomy 的层级）。本研究证明可以用数据驱动的方式找到最优粒度
- KC 的"内在维度"（~64）远小于表面 KC 数量（75），说明 KC 之间存在大量相关性
- 这为"自适应 KC 粒度"开辟了可能：不同学生/不同学习阶段可以使用不同粒度的 KC 表示

**可信度**: **中高** — 仅在 1-2 个数据集上验证，但结论的方向性可靠

**可论文化方向**：可作为 KCGen-KT 后续工作的核心研究问题

---

## 发现 5：LLM 语义空间可以表示连续概率

**结论**：LLM 的 token embedding 空间具有足够的线性性，使得 `p * emb(True) + (1-p) * emb(False)` 可以有效表示概率 p 对应的语义含义。

**证据来源**：

| 实验 | 具体表现 |
|------|---------|
| Mastery Level 嵌入 | True/False 加权嵌入显著优于直接数值注入或 soft token |
| KC 正确性预测 | 模型能从嵌入中恢复出 binary correctness，说明嵌入保留了概率信息 |

**核心洞察**：
- 这利用了 LLM 预训练中 True/False 的语义对立性
- 可以推广：任何二元概念的连续中间状态都可以用这种方式表示（如 positive/negative sentiment 的强度）
- 这是一种"借用 LLM 已有知识"来编码新信息的低成本策略

**可信度**: **中** — 理论基础直觉上合理，但缺乏在其他场景的验证

**可论文化方向**：可作为 technical insight 写入 LLM 应用技巧论文

---

## 发现 6：Few-shot 示例的多样性比数量更关键

**结论**：在 LLM few-shot 学习中，选择覆盖不同"类型"的示例，比增加同类型示例的数量更能提升输出质量。

**证据来源**：

| 实验 | 具体表现 |
|------|---------|
| KC 生成 few-shot 设计 | 选 1 个条件判断类 + 1 个数组操作类问题作为 example，效果优于 5 个同类问题 |
| 代码采样策略 | 聚类采样 5 个多样化代码 > 随机采样 5 个代码 |

**核心洞察**：
- 多样性保证了 LLM 学到的是"通用规则"而非"特定模式"
- 这与 active learning 中的"信息增益最大化"策略一致
- 实际应用：在设计 few-shot prompt 时，应先对候选示例做聚类，然后从每个簇中选取代表

**可信度**: **中高** — 在 KC 生成场景中验证，与 few-shot learning 理论一致

**可论文化方向**：可融入 Prompt Engineering 最佳实践综述
