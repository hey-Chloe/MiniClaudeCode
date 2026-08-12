# MiniClaudeCode

面向 Harness Engineering 的轻量级 Coding Agent CLI。

MiniClaudeCode 是一个针对 Coding 场景设计的 Agent Runtime，实现了 Coding Agent 的核心执行闭环，包括 Agent Loop、Tool Calling、Context Management、Permission Control 和 Evaluation Framework。

项目参考 Claude Code、OpenHands 等 Coding Agent 架构思想，将复杂 Agent 系统中的核心机制进行轻量化实现，探索如何构建一个可控、可扩展、可评测的代码智能体执行框架。

---

# 项目概述

Coding Agent 的核心问题并不是简单调用大模型生成代码，而是如何让 Agent 在真实工程环境中完成稳定、可控的任务执行。

主要解决：

- 理解用户任务并拆解目标
- 制定任务执行计划
- 调用外部工具完成实际操作
- 根据执行结果进行持续决策
- 管理长期任务上下文
- 控制自动化执行过程中的安全风险

MiniClaudeCode 围绕以上问题设计了一套轻量级 Coding Agent Runtime，实现从任务理解、工具调用、执行反馈到结果验证的完整闭环。

## 核心架构


# 技术栈

- Python 3.12+
- Custom Agent Runtime
- Agent Loop / Tool Calling
- OpenAI Compatible API（DeepSeek）
- AsyncIO
- Rich CLI
- Pytest / Benchmark Evaluation
- Docker


---

# 🚀 快速开始

## 1. 克隆项目

```bash
git clone https://github.com/hey-Chloe/MiniClaudeCode.git

cd MiniClaudeCode
```

## 2. 创建虚拟环境

```bash
python -m venv venv
```

Windows:

```bash
.\venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 4. 运行 Demo

执行：

```bash
python main.py "fix calculator bug"
```

示例输出：

```text
{
 event: task,
 detail: fix calculator bug
}

{
 event: planning,
 detail: create plan
}

{
 event: tool_selection,
 detail: pytest
}

{
 event: verification,
 detail: passed
}
```

Agent 执行流程：

```
任务接收

↓

任务规划

↓

工具选择

↓

工具执行

↓

结果验证
```

---

# 设计目标

MiniClaudeCode 主要探索：

- 如何构建稳定的 Coding Agent Runtime
- 如何通过 Tool Calling 扩展 Agent 能力
- 如何控制 Agent 自动执行风险
- 如何评估 Agent 在真实工程任务中的执行效果

# 项目结构

```
MiniClaudeCode
│
├── miniclaude
│   ├── agent.py          # Agent Runtime核心逻辑
│   ├── cli.py            # CLI入口
│   ├── tools.py          # Tool Registry
│   └── trace.py          # Agent执行轨迹记录
│
├── runtime
│   └── sandbox.py        # Sandbox执行抽象
│
├── security
│   └── approval.py       # 权限控制与审批机制
│
├── git
│   └── diff.py           # Diff生成模块
│
├── evaluation
│   └── benchmark.json    # Agent评测任务集
│
├── main.py               # 项目入口
└── requirements.txt      # 项目依赖
```
MiniClaudeCode 主要探索：

Coding Agent 如何完成多步骤任务执行
LLM 如何通过 Tool Calling 使用外部能力
如何设计安全可控的 Agent Runtime
如何建立 Agent Evaluation Framework
