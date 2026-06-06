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
