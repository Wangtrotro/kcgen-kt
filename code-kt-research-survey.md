# Code-KT 前沿研究调研报告

## 一、当前 Code-KT 领域最新进展（2025-2026）

### 1.1 核心方向：LLM 增强的编程知识追踪

| 论文/框架 | 会议/期刊 | 核心创新 |
|-----------|-----------|----------|
| **DPKT** (Difficulty-aware PKT via LLMs) | Scientific Reports 2025 | 利用LLM提取题目文本理解难度和知识概念难度，结合注意力机制动态更新学生知识状态 |
| **CQD-PKT** (Code Quality & Difficulty Aware) | ADMA 2025 | 用DeepSeek自动评分代码质量，结合cross-attention评估代码质量对编程能力的影响 |
| **CFKT** (Code-driven Feature Fusion KT) | Information Fusion 2026 | LLM嵌入题目语义+代码语义，构建PP模块(编程预测)和CP模块(代码预测)的融合框架 |
| **KGNN-KT** | ICIC 2025 | LLM提取编程概念构建知识图谱 + GNN建模概念关系 + 多模态架构融合 |
| **TIKTOC** (Test Case-Informed KT) | arXiv 2024 | 将测试用例级别信息引入KT，从二元正误扩展到细粒度test case通过率 |
| **Thinking-KT** | ACL ARR 2026 Jan | **免训练框架**，利用Test-Time Scaling使1-2B小模型实现KT预测+反馈+推荐统一输出 |

### 1.2 重要发现：评估可靠性问题

最新论文 *"Ensuring Reliability in Programming Knowledge Tracing"* (arXiv 2605.04727, 2026) 揭示：
- 现有PKT模型(Code-DKT, ECKT)的性能提升对评估协议和超参数选择高度敏感
- 基线模型DKT在标准化设置下被严重低估
- Code-DKT中代码表征的有效性是**任务依赖**的，而非普适的
- 提出了更可靠的rectified evaluation protocol

---

## 二、可以承接KT研究的前沿方向

### 方向一：Reasoning-Enhanced Code-KT（推理增强的编程知识追踪）

**核心思路**：将Test-Time Scaling / Chain-of-Thought推理引入Code-KT

**关键论文参考**：
- **Thinking-KT** (ACL ARR 2026)：证明了TTS是LLM-based KT性能提升的主要因素
- **Min-Seek** (EACL Findings 2026)：稳定的sequential test-time scaling方法
- **TOPS** (Thinking-Optimal Scaling)：自适应推理长度控制

**可行研究角度**：

1. **Code-Specific Reasoning Traces for KT**
   - Thinking-KT 目前在通用教育数据上验证，尚未针对编程领域的特殊性（如代码执行逻辑、调试过程、算法理解）设计专门的reasoning template
   - **Gap**: 编程KT需要对代码结构进行推理（变量追踪、控制流、递归理解等），现有reasoning traces过于通用
   - **提出**: 设计 Code-Aware Reasoning Template，让模型在推理过程中显式追踪代码执行路径

2. **Execution-Guided Reasoning for KT**
   - 结合RLVR中"执行反馈"的思想，让KT推理过程能"模拟执行"学生代码
   - 不仅预测学生做对/做错，还能推理出学生的具体misconception

---

### 方向二：Multi-Agent Code Tutoring + KT（多智能体编程辅导与知识追踪一体化）

**核心思路**：将KT作为多智能体编程教育系统的核心"学习者建模"模块

**关键论文参考**：
- **IntelliCode** (EACL Demo 2026)：多智能体+集中式学习者建模，用版本化知识状态图数据库
- **GenMentor** (WWW 2025 Oral)：目标导向的多智能体ITS框架
- **CodeEdu** (arXiv 2025)：多智能体协作编程教育平台
- **GraphMASAL** (arXiv 2025)：图基多智能体自适应学习系统
- **ACE-TA** (arXiv 2026)：Agentic编程教学助手

**可行研究角度**：

1. **KT-Guided Multi-Agent Code Tutoring**
   - 现有多智能体编程教育系统(CodeEdu, ACE-TA)缺乏深度的KT集成
   - IntelliCode做了centralized learner modeling但用的是简单规则
   - **Gap**: 将深度KT模型（如Thinking-KT的推理能力）嵌入多智能体tutoring system作为"诊断Agent"
   - **提出**: KT-Agent统一框架，诊断Agent用reasoning-based KT，规划Agent用KT输出做路径优化

2. **Simulated Student Agent for Code-KT Data Augmentation**
   - 参考 Agent4Edu 的思路，用LLM Agent模拟不同水平的编程学生行为
   - 生成大规模合成编程交互数据来增强KT模型训练
   - **Gap**: 现有Code-KT数据集(CodeWorkout: 413学生, 69K交互)规模有限

---

### 方向三：Cross-Lingual/Cross-Task Knowledge Transfer in Code（代码跨语言/跨任务知识迁移）

**核心思路**：将Code领域的知识迁移研究与KT中的"知识迁移"概念连接

**关键论文参考**：
- **Parallel-SFT** (arXiv 2025)：功能等价的parallel programs实现跨编程语言zero-shot迁移
- **CodePivot** (arXiv 2025)：以Python为中间表示的多语言transpilation
- **Agent Distillation** (NeurIPS 2025 Spotlight)：将LLM Agent的tool-use能力蒸馏到小模型

**可行研究角度**：

1. **Cross-Language Programming Knowledge Tracing**
   - 学生学完Python再学Java，KT如何建模跨语言的知识迁移？
   - **Gap**: 现有Code-KT全部是单语言(Java-only via CodeWorkout)，没有跨语言KT的研究
   - **提出**: 利用Parallel-SFT的思路，构建functionality-centric的skill representation，使KT模型能跨语言追踪编程技能掌握

2. **Skill-Level Knowledge Distillation for Code-KT**
   - 将大模型(如GPT-4)的编程能力评估能力蒸馏到小模型用于高效KT
   - 结合Agent Distillation + Thinking-KT的免训练框架
   - **提出**: 用大模型生成高质量reasoning traces，蒸馏到小模型实现高效Code-KT部署

---

### 方向四：RLVR-Inspired Code-KT（可验证奖励驱动的编程知识追踪）

**核心思路**：将RLVR中"可验证奖励信号"的思想引入Code-KT

**关键论文参考**：
- **ADR** (Combinatorial Synthesis, arXiv 2025)：原子分解+重组生成可验证代码任务
- **VeRPO/StepCodeReasoner**: 白盒RL，奖励中间推理步骤
- **Murphy**: Multi-turn RLVR with feedback-conditioned rollout

**可行研究角度**：

1. **Process-Level KT with Verifiable Signals**
   - 传统KT只看最终结果(对/错)，RLVR的process reward思想可以引入到KT
   - 利用编程的天然可验证性（编译、测试用例），构建Process-Level KT
   - **提出**: 用test case通过率作为细粒度可验证信号，构建编程过程中的knowledge state estimation

2. **Self-Evolving Code-KT via RL**
   - KT模型通过RL自动学习最优的knowledge tracing策略
   - 奖励信号：预测准确性 + 推荐有效性（学生是否因推荐而进步）
   - **Gap**: 现有KT全部是监督学习范式，缺少RL-based自适应KT

---

### 方向五：Trace-Based Programming Learning Modeling（基于编程痕迹的学习建模）

**核心思路**：利用学生编程过程中的rich traces建模学习行为

**关键论文参考**：
- **Pencil Code Dataset** (arXiv 2025)：3.8M编程推理痕迹，训练language model建模学生编程行为
- **TIKTOC** (Test Case-Informed KT)：测试用例级别的细粒度KT

**可行研究角度**：

1. **Edit-Trace-Aware Code-KT**
   - 不仅看最终提交，而是建模学生的完整编辑历史（如何一步步修改代码）
   - **Gap**: 现有Code-KT仅使用最终提交结果，浪费了丰富的过程信息
   - **提出**: 利用code edit traces + LLM embedding构建temporal knowledge state model

2. **Style-Aware Personalized Code Assessment**
   - 不同学生有不同编程风格，KT应该考虑个体差异
   - 参考Pencil Code论文中"many properties of code are properties of individual students"
   - **提出**: 构建student-aware code representation，使KT模型能区分"真正掌握"vs"模仿性通过"

---

## 三、如何把故事讲圆：推荐的研究路线

### 路线A：Reasoning-Enhanced Code-KT（最直接，最易出成果）

**故事线**: "现有Code-KT模型缺乏推理能力，无法理解代码的执行语义" → "引入TTS/CoT使KT模型具备代码推理能力" → "实现更精准的编程能力追踪+可解释的诊断"

**承接关系**:
- 承接KT领域：Thinking-KT (2026) → 你的工作（Code-Specific Reasoning-KT）
- 承接Code领域：Code Reasoning Survey (ACL ARR 2026) 中提到的inference-time reasoning技术
- 衔接点：编程天然适合reasoning-based追踪（代码执行本身就是推理过程）

**预期贡献**:
1. 设计编程领域专用的reasoning template（代码追踪、调试推理、算法分析）
2. 提出execution-grounded reasoning for KT（推理过程可通过代码执行验证）
3. 在CodeWorkout等数据集上证明code-specific reasoning显著优于generic reasoning

---

### 路线B：Multi-Agent KT for Programming Education（系统级贡献）

**故事线**: "现有编程教育系统的学习者建模与教学决策割裂" → "统一KT诊断与多智能体教学" → "实现闭环的个性化编程教育"

**承接关系**:
- 承接KT领域：LPReKL (2025, KT+LLM闭环) → 你的工作
- 承接Agent领域：IntelliCode, CodeEdu, GenMentor → 你的工作
- 衔接点：KT作为Agent系统的"感知模块"

**预期贡献**:
1. 设计KT-Agent架构（KT模型输出驱动Agent的教学决策）
2. 实现KT与内容生成的bidirectional feedback loop
3. 在真实编程教育场景中验证系统效果

---

### 路线C：Process-Level Code-KT with Verifiable Rewards（理论创新+实践价值）

**故事线**: "传统KT仅建模最终结果，忽略编程过程信息" → "利用编程天然的可验证性(test cases)构建过程级KT" → "实现从'知道学生对不对'到'理解学生为什么错'的跨越"

**承接关系**:
- 承接KT领域：TIKTOC (2024, test-case level KT) → 你的工作
- 承接Code/RL领域：RLVR + process reward → 你的工作
- 衔接点：编程是少有的同时具备"可验证性"和"丰富过程信息"的教育场景

**预期贡献**:
1. 提出process reward inspired KT loss function
2. 利用test case作为细粒度验证信号的multi-granularity KT
3. 证明过程级建模显著优于结果级建模

---

### 路线D：Cross-Lingual Programming Knowledge Transfer（开拓新问题）

**故事线**: "学生从一门编程语言迁移到另一门时，哪些技能能迁移？" → "提出跨语言编程知识追踪问题" → "利用functionality-centric code representation实现跨语言skill tracking"

**承接关系**:
- 承接KT领域：knowledge transfer in KT → 你的工作
- 承接Code领域：Parallel-SFT, CodePivot (跨语言code transfer) → 你的工作
- 衔接点：编程技能本质上是跨语言的，但现有KT无法建模这一点

**预期贡献**:
1. 定义Cross-Lingual Programming KT新问题
2. 构建跨语言编程学习数据集
3. 提出functionality-centric skill representation使KT跨语言泛化

---

## 四、综合建议

### 最推荐路线：路线A（Reasoning-Enhanced Code-KT）

**理由**：
1. **时机最好**：Thinking-KT刚发表(2026年1月)，在Code上的特化是自然延伸
2. **门槛适中**：不需要构建大系统或收集新数据集，可在现有CodeWorkout上做
3. **故事最圆**：完美连接"code reasoning"大趋势和"KT"核心方向
4. **可验证性强**：编程领域的reasoning可以通过代码执行来验证
5. **扩展空间大**：做完后可以自然延伸到路线B(加入Agent)或路线C(加入process-level)

### 如果想要更大创新性：路线C + A组合

"Process-Level Reasoning-Enhanced Code-KT"
- 同时利用过程信息（edit traces / test cases）和推理增强
- 提出"execution-verifiable reasoning for programming KT"
- 既有方法创新（reasoning for KT），又有问题创新（process-level KT）

---

## 五、关键参考文献汇总

### Code-KT 核心文献
1. Thinking-KT (ACL ARR 2026): https://arxiv.org/html/2601.01708v1
2. DPKT (Scientific Reports 2025): https://www.nature.com/articles/s41598-025-96540-3
3. CFKT (Information Fusion 2026): https://www.sciencedirect.com/science/article/abs/pii/S1566253526000448
4. CQD-PKT (ADMA 2025): https://link.springer.com/chapter/10.1007/978-981-95-3459-3_6
5. KGNN-KT (ICIC 2025): https://link.springer.com/chapter/10.1007/978-981-96-9986-5_12
6. TIKTOC (arXiv 2024): https://arxiv.org/html/2410.10829v3
7. PKT Reliability (arXiv 2026): https://arxiv.org/html/2605.04727v1

### Code Reasoning & LLM
8. Code Reasoning Survey (ACL ARR 2026): https://openreview.net/forum?id=pmb3YKhSVu
9. How Does LLM Reasoning Work for Code (arXiv 2025): http://www.arxiv.org/pdf/2506.13932
10. Parallel-SFT (arXiv 2025): https://arxiv.org/html/2604.20835v2
11. CodePivot (arXiv 2025): https://arxiv.org/html/2604.18027
12. ADR/Combinatorial Synthesis (arXiv 2025): https://arxiv.org/html/2605.31058v1

### 教育智能体
13. IntelliCode (EACL Demo 2026): https://aclanthology.org/2026.eacl-demo.10.pdf
14. GenMentor (WWW 2025): https://arxiv.org/abs/2501.15749
15. CodeEdu (arXiv 2025): https://arxiv.org/html/2507.13814v1
16. LPReKL (Electronics 2025): https://www.mdpi.com/2079-9292/14/22/4385
17. GraphMASAL (arXiv 2025): https://arxiv.org/html/2511.11035v1
18. ACE-TA (arXiv 2026): https://arxiv.org/pdf/2604.09572

### KT + LLM 通用
19. Pencil Code Traces (arXiv 2025): https://arxiv.org/html/2510.05056v2
20. CoTutor (arXiv 2025): https://arxiv.org/html/2509.23996v1
21. LMM-based KC Extraction (EDM 2025): https://educationaldatamining.org/EDM2025/proceedings/2025.EDM.long-papers.170/

### 知识蒸馏与迁移
22. Agent Distillation (NeurIPS 2025): https://arxiv.org/pdf/2505.17612
23. Multi-Step KD (EACL SRW 2026): https://aclanthology.org/2026.eacl-srw.13.pdf

### 数据集
24. CodeWorkout: https://codeworkout.cs.vt.edu/ (413 students, 69K interactions, Java)
25. CSEDM Data Challenge: https://sites.google.com/ncsu.edu/csedm-dc-2021/dataset
26. Pencil Code: https://github.com/meghabyte/pencilcode-public (3.8M traces)

---

## 六、基于 KCGen-KT 项目的论文故事线设计

### 项目回顾：KCGen-KT 做了什么

KCGen-KT (arXiv:2502.18632) 的核心贡献：
1. **KC自动生成**：GPT-4o few-shot → 多样性采样 → SentenceBERT聚类 → GPT-4o摘要 → 50个中粒度KC
2. **Soft Token注入**：将KC mastery以可微分的soft token形式注入Llama-3提示
3. **多任务学习**：同时预测正确性 + 生成代码
4. **核心发现**：LLM生成的KC显著优于人工标注的KC

### KCGen-KT 现有局限（即你的下一篇论文的motivation）

| 局限 | 描述 |
|------|------|
| KC生成是静态的 | KC一次性生成后固定，无法根据学生表现自适应调整 |
| KC粒度控制靠启发式 | 聚类数目(50/60)是超参数，缺乏理论指导 |
| 不理解"为什么错" | 只追踪mastery分数，不建模学生的misconception |
| 代码表征是浅层的 | ASTNN 200-dim + LLM 4096-dim拼接，缺少执行语义 |
| 缺少推理过程 | 模型直接预测，不解释为什么认为学生会对/错 |
| 数据规模受限 | CodeWorkout仅246学生，10K交互 |

---

## 故事线方案一（最推荐）：Reasoning-Grounded KCGen-KT

### 标题方向
**"From Knowledge Components to Knowledge Reasoning: Execution-Grounded Reasoning for Interpretable Programming Knowledge Tracing"**

或中文：基于执行推理的可解释编程知识追踪

### 故事逻辑

```
[Problem] KCGen-KT生成了好的KC，但KT过程是"黑盒"的
     ↓ 缺什么？
[Gap] 模型知道学生mastery=0.3，但不知道为什么=0.3
     ↓ 怎么解决？
[Method] 引入Reasoning Module：让模型在预测前先"推理"
     ↓ 关键创新点
[Innovation] 推理过程可以通过代码执行来验证（execution-grounded）
     ↓ 效果
[Results] 更准确的KT + 可解释的诊断 + 个性化反馈生成
```

### 具体方法设计

```
┌──────────────────────────────────────────────────────────────┐
│              Reasoning-KCGen-KT Architecture                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [Stage 1: KC-Aware Reasoning Prompt]                        │
│  ┌──────────────────────────────────────────┐               │
│  │ Student history: {problems, codes, scores} │               │
│  │ Current KCs: {kc1: mastery_1, ...}        │               │
│  │ Next problem: {...}                        │               │
│  │                                            │               │
│  │ REASONING TEMPLATE:                        │               │
│  │ 1. 分析学生在每个KC上的表现模式            │               │
│  │ 2. 识别可能的misconception                 │               │
│  │ 3. 模拟学生可能写出的代码逻辑              │               │
│  │ 4. 预测代码在各test case上的表现           │               │
│  │ 5. 给出最终预测和诊断                      │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  [Stage 2: Execution Verification]                           │
│  - 模型推理出的"学生可能犯的错" 可以通过                    │
│    实际执行test case来验证                                   │
│  - 形成self-verification loop                                │
│                                                              │
│  [Stage 3: Reasoning-Enhanced Mastery Update]                │
│  - 推理结论反馈到LSTM的mastery更新                           │
│  - 不再是纯数值更新，而是有语义支撑的更新                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 为什么这个故事讲得圆

1. **自然承接KCGen-KT**：你已经有了KC生成和soft token注入，下一步自然是"让模型理解KC"
2. **连接Code Reasoning大趋势**：2026年最热门方向
3. **连接Thinking-KT**：但你做的是CODE-SPECIFIC reasoning（有execution verification），比Thinking-KT的generic reasoning更有针对性
4. **保持可验证性**：编程的独特优势——推理可以通过执行来验证
5. **解决实际痛点**：教育场景需要可解释的诊断，不只是一个分数

### 实验设计

| 实验 | 验证什么 |
|------|----------|
| KCGen-KT vs. Reasoning-KCGen-KT | 推理是否提升KT准确率 |
| Generic Reasoning vs. Code-Specific Reasoning | 编程专用推理模板是否优于通用模板 |
| With/Without Execution Verification | 执行验证是否提升推理质量 |
| Reasoning Trace Analysis | 模型是否真正识别了student misconception |
| Thinking Budget Ablation | 推理长度对性能的影响 |
| Generated Feedback Quality | 推理过程是否产生更好的个性化反馈 |

### 承接关系图

```
KCGen-KT (你的已发表工作)
    ↓ "KC生成好了，但KT不够智能"
Thinking-KT (ACL ARR 2026, 别人的工作)
    ↓ "证明了TTS对KT有效，但没考虑代码特殊性"
你的新工作: Reasoning-KCGen-KT
    = KCGen-KT的KC表示 + Thinking-KT的TTS思想 + Code执行验证(你的创新)
```

---

## 故事线方案二：Adaptive KC Evolution with Feedback Loop

### 标题方向
**"Self-Evolving Knowledge Components: Closing the Loop Between KC Generation and Knowledge Tracing in Programming Education"**

### 故事逻辑

```
[Problem] KCGen-KT的KC是静态的，生成一次后固定不变
     ↓
[Observation] 不同学生群体需要不同粒度/类型的KC
     ↓
[Gap] KC生成和KT是割裂的两个阶段，没有反馈闭环
     ↓
[Method] 让KT的表现反馈回KC生成，形成迭代优化
     ↓
[Innovation] KT-guided KC refinement：
     - KT性能差的KC → 需要细分
     - KT性能好但区分度低的KC → 可以合并
     - 学生频繁混淆的概念 → 需要新KC
```

### 具体方法

```
┌─────────────────────────────────────────────────────────┐
│            Self-Evolving KCGen-KT                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Cycle 1]                                              │
│  KC Generation (GPT-4o) → KCGen-KT Training → Eval     │
│       ↑                                          │      │
│       │         ← Feedback Signal ←              │      │
│       │                                          ↓      │
│  [Cycle 2]                                              │
│  KC Refinement:                                         │
│  - 分析哪些KC的mastery预测误差最大                      │
│  - 识别被频繁混淆的KC pairs                             │
│  - 生成refinement prompt给GPT-4o                        │
│  - 重新聚类/分裂/合并KC                                 │
│       ↓                                                 │
│  Updated KCs → Re-train → Eval → ...                    │
│                                                         │
│  [Stop Criterion]                                       │
│  - KT performance converges                             │
│  - KC数量稳定                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 为什么这个故事讲得圆

1. **直接解决KCGen-KT的核心局限**：静态KC → 动态KC
2. **连接LPReKL的闭环思想**：KT和LLM的双向交互
3. **有Education理论支撑**：KC理论本身强调"right grain size"
4. **实验自然对比**：1次生成 vs. N次迭代的消融实验
5. **可以讲增量贡献**：每一轮KC进化带来多少性能提升

---

## 故事线方案三：Process-Level KCGen-KT

### 标题方向
**"Beyond Final Submissions: Process-Aware Knowledge Component Generation and Tracing for Programming Education"**

### 故事逻辑

```
[Problem] KCGen-KT只看最终正确代码生成KC
     ↓
[Observation] 学生的代码编辑过程包含丰富的learning signal
     ↓
[Gap] 从"正确答案"生成的KC无法描述"常见错误模式"
     ↓
[Method] 从编辑过程(process)中生成KC
     - 错误代码→正确代码的diff中蕴含misconception
     - 多次提交的演变反映知识获取过程
     ↓
[Innovation] Process-KC：描述"学生容易犯什么错"的KC
     + Process-Level KT：追踪学生是否克服了特定misconception
```

### 具体方法

```
Phase 1: Process-Aware KC Generation
- 输入: (错误代码, 正确代码, diff) pairs
- LLM分析: "这个错误反映了什么misconception?"
- 聚类: 生成Misconception-KCs (M-KCs)
- 结果: 每个problem有两类KC
  - Skill-KCs (from correct solutions, 原KCGen)
  - Misconception-KCs (from error patterns, 新增)

Phase 2: Dual-Track KT
- Skill Mastery Track: 追踪正向技能掌握 (原有)
- Misconception Track: 追踪错误概念是否被克服 (新增)
- 融合: 综合两个track预测下一次表现
```

### 为什么这个故事讲得圆

1. **解决KCGen-KT"只看正确答案"的局限**
2. **连接Pencil Code Traces (2025)的过程建模思想**
3. **连接TIKTOC的test-case级别细粒度建模**
4. **教育意义明确**：misconception detection是tutoring系统的核心需求
5. **数据利用率高**：CodeWorkout本身就包含错误提交，现在被浪费了

### 关键发现可以预期
- KCGen-KT原论文发现"包含错误代码生成KC会hurt performance"
- 但那是因为把error和skill混在一起了
- 分开建模(dual-track)应该能解决这个问题 → 这就是你的contribution

---

## 故事线方案四：Multi-Granularity Reasoning KCGen

### 标题方向
**"Hierarchical Knowledge Reasoning for Programming Knowledge Tracing: From Atomic Operations to Algorithmic Thinking"**

### 故事逻辑

```
[Problem] KCGen-KT用固定粒度的KC (50个cluster)
     ↓
[Observation] 编程知识天然是分层的:
     - 语法层: for循环, if语句
     - 操作层: 数组遍历, 字符串操作
     - 算法层: 分治, 动态规划
     - 思维层: 问题分解, 边界处理
     ↓
[Gap] 单一粒度的KC无法刻画这种层次结构
     ↓
[Method] 层次化KC + 层次化推理
     - 自底向上: 语法掌握 → 操作能力 → 算法理解
     - 诊断时自顶向下: 算法不行 → 哪个操作不行 → 具体语法问题
```

### 为什么讲得圆

1. 直接扩展KCGen-KT的聚类层次（现在是flat的50个KC）
2. 连接知识图谱方向（KGNN-KT的层次结构思想）
3. 连接Code Reasoning的"从简单到复杂推理"
4. 教育学理论支撑（Bloom's taxonomy层次）

---

## 综合推荐

| 方案 | 创新性 | 可行性 | 故事连贯性 | 预期venue |
|------|--------|--------|------------|-----------|
| **方案一: Reasoning-KCGen-KT** | ★★★★ | ★★★★ | ★★★★★ | ACL/EMNLP/NeurIPS |
| 方案二: Self-Evolving KC | ★★★ | ★★★★★ | ★★★★ | AAAI/IJCAI |
| **方案三: Process-Level KCGen** | ★★★★ | ★★★★ | ★★★★ | EDM/LAK/KDD |
| 方案四: Hierarchical KC | ★★★ | ★★★ | ★★★ | CIKM/ECML |

### 最终推荐：方案一（Reasoning）或方案三（Process-Level）

- **方案一**适合投NLP/AI顶会：因为连接了reasoning大趋势
- **方案三**适合投教育数据挖掘/应用场景：因为直接解决KCGen-KT原文的遗留问题（错误代码hurt KC质量）

如果时间和精力允许，**方案一+三的组合**（Process-Aware Reasoning KCGen-KT）可以同时解决"不理解错误"和"不会推理"两个问题，冲击最高venue。
