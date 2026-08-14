# MiniClaudeCode 升级方案（v5.1 → v6.x）

> **执行状态（2026-08-14）**：v6.0 的 D1（测量基建 + jitter + Recovery
> Rate + 版本同步 + pytest 修复）与 D2（五阶段状态机 + Verify + trace 补全）
> 已完成；D3（工具激活 + 语义 skill 路由 + 路由评估）已完成并全量回归通过
> （路由命中率 keyword/hybrid 26/26 →
> `reports/skill-routing-2026-08-14-3863285d31.json`）；D4（4 层上下文压缩 +
> read 缓存）、D5（并发 dispatch）、D6（--resume 接线 + provider_state 修复 +
> Rich CLI + 流式 + Anthropic 适配器）已完成；v6.1 策略级自进化核心
> （`evaluation/evolution.py`：策略空间生成 → 分层训练/holdout → 晋升/回滚）
> 与 MCP stdio 适配器（`miniclaude/mcp/`）已完成。全量 pytest 全绿；离线 26/26 →
> `reports/benchmark-validate-2026-08-14-0dd53601b6.json`；
> 演化 dry-run 产物 `reports/evolution-v1-dry-run-2026-08-14-4118225c61.json`。

> 目标：把简历里 24 条可核查声称全部变成"能指到代码/产物"的真实能力。
> 原则：面试零说谎。每条声称三选一——**做实（implement）/ 改口径（rephrase）/ 删（delete）**。
> 版本：v6.0 补漏（把 ⚠️ 全部变 ✅），v6.1 做头牌（策略级自进化），v7 可选扩展。

## 0. 现状基线（2026-08-14 实测）

- 离线 26 任务验证 **26/26 通过**（`python -m evaluation.coding.runner --validate-only`），这是全项目最硬的事实之一。
- pytest 共收集 122 个测试；Windows 上全套跑 >3 分钟未完成（另有 `.pytest_cache` WinError 183 缓存写入告警），CI 健康度需要修。
- `pyproject.toml` 版本号仍为 5.0.0，CHANGELOG 已到 5.1.0，未同步。
- 全仓库 grep 确认：无 asyncio/await、无 anthropic、无 mcp、无 rich 代码引用、无 evolution/holdout/promotion/rollback 关键词。
- 真实资产：有界状态机（`controller.py`）、双协议归一化（Responses + DeepSeek Chat）、安全漏斗（ALLOW/ASK/DENY + 精确调用缓存 + argv 风险分析 + 路径边界）、诚实隔离报告（`RuntimeInfo.isolated`）、原子会话持久化、workspace_diff、trace/metrics 体系、3 个 SKILL、26 任务 benchmark。

---

## 1. 审计 24 条 → 处理策略总表

| # | 简历声称 | 判定 | 处理策略 | 主要文件 | 工作量 |
|---|---------|------|---------|---------|--------|
| 1 | 自进化 Coding Agent Runtime / Self-Evolving | ❌ | **做实（v6.1）**：策略级自进化（Benchmark-Driven Strategy Evolution），不声称模型自进化 | `evaluation/evolution/`（新）、`miniclaude/config.py` | L (2-3d) |
| 2 | Planning → Tool Calling → Context → Skill → Evaluation → Evolution 闭环 | ⚠️ | v6.0 把 Planning 做成真实阶段；v6.1 Evolution 做实 | `controller.py`、`driver.py` | 随 1、3 |
| 3 | 自研 Plan → Act → Observe → Reflect → Verify Agent Loop | ⚠️ | **做实**：状态机加 Verify（有编辑则跑验证再终答）与 Reflect（失败归因事件）阶段 | `controller.py`、`driver.py`、`models.py` | M (1-2d) |
| 4 | 技术栈 Anthropic/OpenAI API | ❌ | **做实（推荐）**：加 Anthropic Messages 适配器，证明"双协议归一化"抽象；同时保留 Responses + DeepSeek 双路径 | `miniclaude/llm/anthropic_provider.py`（新） | M (0.5-1d) |
| 5 | 技术栈 AsyncIO | ❌ | **删除**。并发用线程池（见 #10），不引入 asyncio，面试讲清取舍 | — | S |
| 6 | 技术栈 CLI（Rich） | ❌ | **做实**：rich 已在 requirements，零新依赖；CLI 输出 metrics 表格、approval 渲染、流式打印 | `miniclaude/cli.py` | S (0.5d) |
| 7 | 支持流式输出 | ❌ | **做实**：provider 加 `complete_stream()`（Responses/chat 两路径），CLI `--stream` | `llm/base.py`、`openai_provider.py`、`cli.py` | M (0.5-1d) |
| 8 | 指数退避 + 抖动重试 | ⚠️ | **做实**：加 jitter（full jitter 或 equal jitter），测试断言随机性 | `openai_provider.py`、`tests/test_retry.py` | S (0.5d) |
| 9 | Checkpoint 恢复 | ⚠️ | **做实**：CLI 接 `--resume`；修复 Chat 路径 restore 丢 `tool_call_id` 的问题；中断自动存 checkpoint | `cli.py`、`session.py`、`openai_provider.py` | M (0.5-1d) |
| 10 | 多工具并发执行 | ❌ | **做实（限定口径）**：线程池并发 dispatch 独立调用；ASK 需人工确认的调用串行；不声称 DAG 依赖分析 | `driver.py`、`tools.py`、`metrics.py` | M (0.5-1d) |
| 11 | 13 个核心工具 / MCP Tools | ❌ | **做实（v6.1）**：内置保持 10 个 + MCP stdio 客户端适配器（tools/list → ToolDefinition），口径改为"10 内置 + MCP 可插拔" | `miniclaude/mcp/`（新） | M (1-2d) |
| 12 | 延迟激活降低无关工具上下文开销 | ❌ | **做实**：ToolRegistry 支持按任务激活工具子集（关键词 + skill 映射 + 已用工具扩展，兜底全量），benchmark 上 A/B 实测 | `tools.py`、`driver.py`、`context.py` | M (1d) |
| 13 | 前 5 轮平均节省 Token 7K+ | ❌ | **数字删除**；先做测量基建（版本化报告 + compare 脚本），A/B 实测后才有资格写数字 | `evaluation/reporting.py`（新） | S (0.5d) |
| 14 | 复验 26 组任务，完成率 72% → 88% | ❌（数字无支撑） | **数字删除**；跑一次 live 基线出 artifact，简历只引"离线 26/26 + live 基线 X%" | `evaluation/coding/runner.py`、`reports/` | M (需一次 live 跑) |
| 15 | 多工具任务平均执行时间降低约 15% | ❌ | **删除**；等 #10 落地后用 compare 脚本实测，无产物不写 | — | S |
| 16 | 按任务语义路由加载 skill | ⚠️ | **做实**：TF-IDF/余弦语义检索 + 关键词兜底（无网络依赖），在 26 任务上做路由命中率评估 | `skills.py`、`evaluation/` | M (0.5-1d) |
| 17 | 失败归因、自动生成候选 Prompt/Skill/Routing 策略、Validation/Holdout 回归、版本晋升或回滚 | ❌ | **做实（v6.1 头牌）**：策略注册表 + 候选生成 + holdout 回归 + 晋升/回滚 + 报告 | `evaluation/evolution/`（新） | L (2-3d) |
| 18 | Multi-Agent：Blackboard / Coordinator / Specialist / Reviewer / Critic… | ❌ | **删除**。本周期不做；v7 可选"Reviewer 二次校验"（复用 code-review skill + offline checkers）作为诚实替身 | — | — |
| 19 | 4 层渐进式上下文压缩 | ❌ | **做实**：① budget truncation（已有）② stale snip（旧快照/旧 diff 被更新版本取代）③ micro-compact（长输出首尾保留 + 标记）④ auto-compact（LLM 摘要，默认关）。每层可开关、可测 | `context.py`、`metrics.py`、`tests/test_context.py` | M (1d) |
| 20 | Working / Episodic / Semantic Memory、sideQuery、异步预取、freshness、跨会话记忆 | ❌ | **大部分删除**；留一个诚实替身：read_file 按 (path, mtime) 缓存 + freshness 校验，直接降 repeated_reads 并可测 | `miniclaude/memory.py`（新）、`runtime_tools.py`、`metrics.py` | M (0.5-1d) |
| 21 | 18 组长会话压测：-31% / -43% / 91% | ❌ | **数字删除**；压测基建：合成长会话生成器 + compare 报告 | `evaluation/stress/`（新） | M (1d) |
| 22 | 记录模型决策、Tool Call、Observation、文件修改、Checkpoint 与最终结果 | ⚠️ | **做实**：trace 补 `file_modified`（由 write/replace observation 派生）与 `checkpoint_saved` 事件 | `trace.py`、`driver.py`、`session.py` | S (0.5d) |
| 23 | 统一评测指标含 Recovery Rate | ❌ | **做实**：定义并实现 recovery rate（失败调用后同任务后续恢复的比例），进 RunMetrics + CodingReport | `metrics.py`、`evaluation/coding/models.py` | M (0.5-1d) |
| 24 | 对 Prompt/Skill/Memory/Routing 做 Validation/Holdout 离线回归 | ⚠️ | **做实（v6.1）**：与 #17 合并；validation（可解性，已有）+ holdout 子集 + 策略回归 | `evaluation/evolution/` | 随 17 |

**工作量估算**：v6.0（#2-16、19-23 中标注 v6.0 的项）约 3-4 个工作日（并行可压缩）；v6.1（#1、11、17、24）约 2-3 个工作日。

---

## 2. 版本路线图

### v6.0 "诚实化补漏"（先做，3-4 天）

目的：把"部分支持"全部变成"完全支持"，并建立测量基建，让任何数字都有产物。

1. 测量基建：`evaluation/reporting.py`（版本化报告 + `compare` 脚本 + `reports/` 目录），所有 benchmark 输出落盘。
2. 状态机升级：Plan → Act → Observe → Reflect → Verify 真阶段。
3. trace 补全 + jitter + recovery rate + read 缓存 + 工具激活 + 语义路由 + 4 层压缩。
4. 并发 dispatch + CLI `--resume` 接线 + Rich CLI + 流式输出 + Anthropic 适配器。
5. 项目卫生：版本号同步、pytest 全套可 3 分钟内跑完（修 cache 告警、必要时加 `pytest-timeout`）、README 与简历口径对齐。

### v6.1 "策略级自进化"（头牌，2-3 天）

把简历最大的虚构概念做成**真实的、有界定的**能力：Agent 本体不变，进化发生在"策略空间"（skill 选择、context 预算、重试、工具激活阈值、prompt 版本）。

1. `StrategyConfig` 策略注册表 + 版本化。
2. 候选生成：模板变异 + 可选 LLM 建议（人工确认后才进评估池）。
3. 评估：live 子集 + 固定 holdout 子集；另支持**trace 回放**模式——用已记录的工具调用序列离线估算 token/上下文收益，不花钱。
4. 晋升/回滚：holdout 得分提升且无回归才 promote；否则 rollback；全过程写 `reports/evolution/`。
5. MCP stdio 适配器（10 内置 + MCP 可插拔）。

### v7 "可选扩展"（不做也安全）

- Reviewer 二次校验（用 code-review skill + offline checkers 在终答前复验 diff）。
- 压测报告正式产物（合成长会话 × 内存/记忆缓存 A/B）。
- LLM 摘要压缩（auto-compact 默认开，需基准数据支撑）。

---

## 3. 关键设计

### 3.1 状态机升级（#2、#3）

- `LoopDecision` 增加 `phase` 字段（枚举：`PLAN / ACT / OBSERVE / REFLECT / VERIFY / FINALIZE`），`AgentController.run` 按 phase 推进并记 trace。
- **Plan**：首轮把任务指令中显式要求"先给计划"，driver 将模型首条输出记为 `plan` 事件（CompatibilityLoopDriver 已有 plan 事件可对齐）。
- **Verify**：当本轮含 write/replace 且模型声明完成时，harness 执行配置的验证命令（默认 `pytest -q`，或任务 checker），结果回填给模型；验证通过才允许 terminal。
- **Reflect**：工具失败/策略阻断时生成结构化归因事件（错误类型、工具、是否后续恢复），供 recovery rate 和进化回路消费。
- 验收：`tests/test_agent_loop.py` 增加 verify-phase 用例；CompatibilityLoopDriver 仍 3 步跑通。

### 3.2 工具激活（#12）

- `ToolRegistry` 增加 `activated` 状态：`schemas()` 默认只发激活子集；激活策略 = 任务关键词 + 已选 skill 声明依赖 + 已用工具扩展，未命中任何工具时兜底全量。
- 指标：每轮发送工具数、schema 字符数、兜底触发次数；benchmark 用 `--ab tool-gating` 跑 A/B，产出 `reports/compare-tool-gating.json`。
- 口径：**"按任务激活工具子集，降低无关工具上下文开销（实测见 reports/）"**，不提 7K+ 这类无产物数字。

### 3.3 并发 dispatch（#10）

- `driver.py` 用 `ThreadPoolExecutor` 并发执行独立调用（读文件、grep、diff 等 I/O 密集操作收益明显）。
- 安全：`ApprovalManager` 加锁；含 ASK 决策的批次串行等待人工确认，确认后余下调用可并行。
- 指标：并发批次次数、最大并发度、批耗时；`execute_command` 默认仍单独串行（避免进程树竞争）。
- 口径：**"同轮独立工具调用并发执行（线程池），无 DAG 依赖分析"**——诚实限定，面试官追问也站得住。

### 3.4 语义 skill 路由（#16）

- 实现 `SemanticSkillSelector`：TF-IDF 向量化 skill 文档与任务文本，余弦相似度排序，与关键词分数加权融合；无网络依赖、可单测。
- 在 26 任务上建立"任务分类 → 期望 skill"映射表，评估命中率并写报告。

### 3.5 4 层上下文压缩（#19）

`ContextManager.snapshot()` 依次应用（每层独立开关）：
1. **budget truncation**（已有，字符预算回填）；
2. **stale snip**：删除"同工具、参数相同、已被更新结果取代"的旧 tool 消息（如旧 workspace_diff）；
3. **micro-compact**：超长 tool 输出保留首尾各 N 字符 + `[truncated …]` 标记，并计 metric；
4. **auto-compact**：可选调用 LLM 对最旧消息做摘要（默认关，配置开启）。
- 每层记录 `context_compression: {layer, removed_chars, messages_dropped}` 进 metrics。

### 3.6 read 缓存 + freshness（#20 的诚实替身）

- `FileCache`：key=(path, mtime, size)，命中直接返回缓存内容；写入/替换工具自动失效相关条目。
- metrics 增加 `cache_hits`；benchmark 对比开启前后的 `repeated_read_rate`。
- 口径：**"文件读取缓存（mtime 校验）降低重复读取"**，删掉 Working/Episodic/Semantic Memory、sideQuery、异步预取等全部虚构词。

### 3.7 策略级自进化（#1、#17、#24，v6.1 头牌）

```
StrategyConfig(version) ──候选生成──> candidates[]
      ▲                                      │
      │                               evaluation (live 子集 / trace 回放)
      │                                      │
  晋升/回滚 ◄── holdout 回归（无回归才晋升）◄──┘
      │
      └──> reports/evolution/<version>.json + production strategy 指针
```

- `StrategyConfig`：skill top_k / 路由阈值、context 预算分配、重试参数（含 jitter）、工具激活阈值、system prompt 版本、verify 命令。
- 候选来源：模板变异（预算 ±20%、top_k 1↔2、阈值 0.3↔0.5…）+ 可选 LLM 生成（人工确认）。
- 评估集划分：**固定训练子集（调参用） + 固定 holdout 子集（晋升判定用）**，两者从 26 任务中按类别分层抽样并写死在配置里，防止"偷看 holdout"。
- 晋升规则：holdout 上 success rate 不降 且 (tokens 或 latency 或 recovery rate) 有提升；否则回滚到上一 promoted 版本。
- trace 回放：用已记录的 tool_outputs 序列重放不同 context 预算/激活策略，离线估算 token 收益，成本为零。
- 验收：用 FakeProvider 写一个"候选 A 更优 → promote，候选 B 更差 → rollback"的确定性测试。

### 3.8 MCP 适配器（#11，v6.1）

- `miniclaude/mcp/client.py`：stdio 启动子进程 → `initialize` → `tools/list` → 把 MCP 工具 JSON Schema 转成 `ToolDefinition`。
- 安全：MCP 工具默认 `risk=MUTATING`，全部走既有 ALLOW/ASK/DENY 漏斗；连接参数来自白名单配置。
- 演示：接入一个 filesystem MCP server 做 demo。
- 口径：**"10 个内置工具 + MCP 工具可插拔接入"**，不再写 13。

### 3.9 测量基建（#13、#14、#15、#21）

- `evaluation/reporting.py`：
  - `save_report(name, payload)`：按日期+hash 写入 `reports/`；
  - `compare(a, b)`：输出 delta JSON + Markdown（成功率的提升/回落、tokens、延迟、repeated_read_rate…）。
- 所有 live benchmark 必须 `--output reports/…`；简历上**每一个数字必须能指到一个 reports 文件**。
- 建议做一次 live 基线（26 任务 × 你面试用的模型，预算内跑完），产出 `reports/baseline-2026-xx.json`。

---

## 4. 简历口径与面试弹药

### 新口径示例（每条都能指到代码/产物）

| 简历条目（新） | 指向 |
|---|---|
| 自研有界状态机（Plan/Act/Observe/Reflect/Verify 五阶段） | `controller.py`、`driver.py` |
| 双协议归一化：OpenAI Responses + DeepSeek Chat + Anthropic Messages 三适配器 | `llm/*.py` |
| 安全漏斗：JSON Schema 校验 → ALLOW/ASK/DENY → 精确调用缓存 → 路径边界 | `security/`、`tools.py` |
| 诚实隔离：本地进程非沙箱（isolated=False）、Docker 网络隔离（isolated=True） | `runtime/` |
| 26 组 Coding Tasks 离线验证 26/26；live 基线 X%（见 reports/） | `evaluation/coding/`、`reports/` |
| 策略级自进化：候选生成 → holdout 回归 → 晋升/回滚（不训练模型权重） | `evaluation/evolution/` |
| 按任务激活工具 + 语义 skill 路由，A/B 实测上下文收益（reports/） | `tools.py`、`skills.py` |
| 4 层渐进式上下文压缩（逐层可开关、可测） | `context.py` |
| 同轮独立工具并发执行（线程池），多工具任务耗时 A/B（reports/） | `driver.py` |
| 中断恢复：CLI `--resume`，Responses 走 previous_response_id，Chat 走历史重建 | `cli.py`、`session.py` |

### 预判面试追问

- **"你说 Self-Evolving，怎么个进化法？"**
  答：进化发生在策略空间而非模型权重——候选策略在固定训练子集上评估，holdout 子集上做回归判定，达标才晋升，不达标自动回滚；全程有报告和可复现实验。一句话："我在做工程级的配置自优化，不是伪装的 RL。"
- **"Token 到底省了多少？"**
  答：不背数字，直接指 `reports/compare-tool-gating-*.json` 的 A/B 结果；没有产物的数字不上简历。
- **"为什么不用 asyncio？"**
  答：同步边界让状态机和安全漏斗可单测、可审计；I/O 并发用线程池处理，取舍是"可测性优先"。这是有意识的架构决定。
- **"Multi-Agent 呢？"**
  答：本轮刻意不做——黑板上多 Agent 的价值在这个规模下小于其复杂度；我用 Reviewer 二次校验 + 离线 checker 达到同样的"终答前复验"效果。诚实比堆概念更能过深挖。
- **"72% → 88% 怎么来的？"**
  答：这是旧简历的虚构数字，已删除。真实数据是离线 26/26 + live 基线 X%（reports 可查）；提升对比只出现在 A/B 实验报告里。

---

## 5. 不做清单（避免再造 vaporware）

- 不做模型级自进化（无 RL 数据与算力，声称可被一眼拆穿）。
- 不做全量 asyncio 改造（收益低、风险高）。
- 不做真 Multi-Agent 框架（除非 v7 且时间充足）。
- 不做向量数据库语义记忆（数据量撑不起，且是另一个项目）。
- 不做"13 个工具"这种凑数口径；内置工具数就是 10。
- 任何量化数字必须等到 `reports/` 里有对应 A/B 或基线产物之后才允许写进简历。

---

## 6. 两周执行排期（建议）

| 天 | 内容 |
|---|------|
| D1 | 测量基建（reporting + reports/）+ 项目卫生（版本号、pytest 修复）+ jitter + recovery rate |
| D2 | 状态机五阶段 + trace 补全（file_modified / checkpoint_saved） |
| D3 | 工具激活 + 语义 skill 路由 + 路由命中率评估 |
| D4 | 4 层上下文压缩 + read 缓存（freshness）+ metrics 扩展 |
| D5 | 并发 dispatch + 安全锁 + 并发指标 |
| D6 | CLI：--resume 接线 + Rich 输出 + --stream（流式） |
| D7 | Anthropic 适配器 + 三协议归一化测试 |
| D8 | 跑一次 live 基线 → reports/baseline-*.json，A/B 工具激活 → compare 报告 |
| D9-D11 | v6.1 策略注册表 + 候选生成 + trace 回放 + holdout 晋升/回滚（含 FakeProvider 测试） |
| D12 | MCP stdio 适配器 + demo |
| D13 | 简历/README 按新口径重写，每个声称锚定代码/产物 |
| D14 | 全量回归：pytest + 离线 26/26 + 一次 live 复跑，冻结报告 |

---

## 7. 先做什么（如果你只想动第一刀）

第一步做 **D1：测量基建 + jitter + recovery rate**。它成本最低（0.5-1 天），却立刻改变面试形态：所有数字开始"有出处"，并且 `recovery rate` 是唯一一个把"失败归因/恢复"做成可量化指标的增量，直接支撑后面 v6.1 的进化闭环。
