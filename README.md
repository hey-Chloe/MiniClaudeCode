# MiniClaudeCode
# 用python代码完成最小闭环！
面向 Harness Engineering 的轻量级 Coding Agent Runtime

MiniClaudeCode 是一个针对 Coding 场景设计的轻量级 Agent Runtime。参考 Claude Code、OpenHands 等 Coding Agent 的架构思想。

模型负责决策

Tool 描述可执行能力

Security 决定是否允许

Runtime 负责真实执行

Observation 把真实结果反馈给模型

demo:https://www.bilibili.com/video/BV1ZSgj6uEsy/?spm_id_from=333.1387.homepage.video_card.click&vd_source=102fb68c5f80c92499d6704930157555
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Runtime](https://img.shields.io/badge/Runtime-Local%20%7C%20Docker-green)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20Compatible-orange)
![CI](https://img.shields.io/github/actions/workflow/status/hey-Chloe/MiniClaudeCode/ci.yml?branch=main)
![License](https://img.shields.io/badge/License-TBD-lightgrey)

# 和普通 LLM API 调用的区别

  普通调用是 User → LLM → Answer：模型是纯函数，输入 prompt 输出文本，它无法改变世界，也无法感知改变后的世界。让它修
  bug，它只能"猜"哪里错了、"假设"测试会过，因为它看不到文件、跑不了测试。

  MiniClaudeCode 把模型放进一个 Harness（围栏+脚手架） 里，变成 User → LLM → Action → Environment → Observation → LLM
  

# 核心能力

多轮 Agent Loop 与最大轮次控制

OpenAI-compatible / DeepSeek LLM Provider

JSON Schema 驱动的 Tool Calling

Coding Tools：目录浏览、Glob、Grep、文件读写、精确文本修改、命令执行、Git Diff

ALLOW / ASK / DENY 三态权限策略

plan / default / accept-edits / bypass 权限模式

Workspace 路径边界与危险命令分析

Local Runtime / Docker Runtime

Context Management 与项目指令加载

JSONL Trace 与 Session Audit

离线 Evaluation 与真实 Agent Fixture

Windows / Linux GitHub Actions CI


<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/e51e18d0-c1fd-4b36-9e83-9529755736da" />



## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/hey-Chloe/MiniClaudeCode.git
cd MiniClaudeCode
```

### 2. 创建虚拟环境

Windows：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux / macOS：

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
python -m pip install -e ".[dev]"
```

### 4. 配置模型

项目支持通过 `.env` 文件配置模型。

在项目根目录创建：

```text
.env
```

以 DeepSeek 为例：

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.deepseek.com
MINICLAUDE_MODEL=deepseek-chat
```

`.env` 已被 Git 忽略，请不要提交真实 API Key。

### 5. 运行 MiniClaudeCode

只读分析模式：

```powershell
python main.py "分析当前项目结构" --mode v5 --runtime local --permission-mode plan
```

默认权限模式：

```powershell
python main.py "分析并改进当前项目" --mode v5 --runtime local --permission-mode default
```

Docker Runtime：

```powershell
python main.py "分析当前项目结构" --mode v5 --runtime docker --permission-mode plan
```





# MiniClaudeCode 的设计原则：

Model makes decisions

模型负责选择下一步动作。


Harness controls execution

所有真实执行都必须经过 Tool、Security 和 Runtime。


Observations are first-class data

工具执行结果以结构化 Observation 返回模型。


Security before execution

权限判断发生在真实副作用之前。


Isolation must be explicit

Local Runtime 不伪装成 Sandbox，只有真正隔离的 Runtime 才报告 isolated=True。


Everything should be testable

Agent Loop、Tool、Security、Runtime 和 Context 都应该可以在不依赖真实 LLM 的情况下单独测试。



License

License TBD.
