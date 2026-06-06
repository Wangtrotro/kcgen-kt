# Research Assets — 研究知识体系

> 从"实验记录"升级为"研究资产"

本目录包含从当前仓库 (KCGen-KT) 和相关研究方向中系统性提炼的研究资产。每份文档都是 Obsidian 兼容的 Markdown 格式，可直接作为 Obsidian Vault 使用。

---

## 文件索引

| 文件 | 内容 | 对应任务 |
|------|------|---------|
| [[00-Research-Map]] | 按研究主题组织的全景知识地图 | Part 1: 研究地图 |
| [[01-Experiment-Registry]] | 所有实验的深度分析（假设→设计→发现→方法论价值） | Part 2: 实验重构 |
| [[02-Research-Findings]] | 跨实验归纳的研究发现 | Part 3: 实验归纳 |
| [[03-Paper-Review]] | 论文阅读记录（价值导向，非 summary 导向） | Part 4: 论文重构 |
| [[04-Methodology-Library]] | 从项目中沉淀的可复用方法论 | Part 5: 方法论库 |
| [[05-Dashboard-Research]] | 研究方向、关键发现、论文分布总览 | Part 6: Dashboard |
| [[06-Dashboard-Experiments]] | 实验假设、设计、发现、状态总览 | Part 6: Dashboard |
| [[07-Dashboard-Papers]] | 论文核心思想、复现价值、关联度总览 | Part 6: Dashboard |
| [[08-Dashboard-Methodology]] | 方法论来源、成熟度、复用场景总览 | Part 6: Dashboard |

---

## 如何使用

### 作为 Obsidian Vault
1. 将 `research-assets/` 目录作为 Obsidian Vault 打开
2. 使用 `[[wikilink]]` 导航不同文档
3. 使用 Obsidian Graph View 查看文档间关联

### 作为参考文档
- **开始新项目前**: 查看 [[00-Research-Map]] 定位项目在研究版图中的位置
- **设计实验前**: 查看 [[04-Methodology-Library]] 选择可复用的方法论
- **写论文时**: 查看 [[02-Research-Findings]] 提取可写论文的结论
- **读论文时**: 参考 [[03-Paper-Review]] 的格式记录新论文

---

## 核心仓库信息

**项目**: KCGen-KT — Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems

**论文**: [arxiv 2502.18632](https://arxiv.org/abs/2502.18632)

**核心贡献**:
1. 基于 LLM (GPT-4o) + Solution AST 的全自动 KC 生成管线
2. KC mastery level 驱动的 Generative Knowledge Tracing 框架
3. LSTM (KC tracking) + LLaMA-3-8B (Code Generation) 的多任务架构
