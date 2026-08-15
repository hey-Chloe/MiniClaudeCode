# Changelog

## 5.3.0 - 2026-08-15

- The loop now records the full Plan -> Act -> Observe -> Reflect -> Verify
  progression on tool rounds: the driver emits an explicit `tool_calls` trace
  event and advances `AgentResult.phases` through `act -> observe` (adding
  `reflect` when any tool call fails) inside a single decision
  (`LoopDecision.extra_phases` / `extra_events`), so turn accounting and
  checkpoint semantics are unchanged.
- MCP is wired end to end: a bundled demo server
  (`python -m miniclaude.mcp.demo_server`) exposes read-only and mutating
  tools, the CLI attaches it with `--mcp-demo` (or any stdio server with
  `--mcp NAME=COMMAND [ARGS...]`), server tools declare
  `annotations.readOnlyHint` to map to `READ_ONLY` risk, and every MCP call
  still passes through the ALLOW/ASK/DENY funnel. `MCPServerConfig` gained
  `activation_keywords` for tool gating.
- New offline benchmarks with versioned artifacts:
  - `python -m evaluation.stress` replays a deterministic synthetic long
    session through the real context-compression layers
    (`reports/stress-compression-*.json`);
  - `python -m evaluation.ab_tool_gating` measures tool-gating schema savings
    on all 26 benchmark tasks (`reports/ab-tool-gating-*.json`);
  - `python -m evaluation.ab_concurrency` micro-benchmarks sequential vs
    pooled read-only dispatch (`reports/ab-concurrency-*.json`);
  - `python -m evaluation.ab_read_cache` A/Bs the freshness read cache on a
    repeated-read session (`reports/ab-read-cache-*.json`).
- New console scripts: `miniclaude-stress`, `miniclaude-ab-gating`,
  `miniclaude-ab-concurrency`, `miniclaude-ab-read-cache`.
- `safety_block_rate` is now a real metric on `RunMetrics` and the coding
  benchmark report (denied calls / tool calls).
- The evolvable strategy space now includes `routing_mode`
  (keyword/hybrid/semantic) and `read_cache_enabled`, both wired into the
  live runner and the evolution candidate generator.
- Failure attribution is real: `evaluation/attribution.py` derives structured
  attribution from a run trace (failed tools, error kinds, phase, recovery,
  policy denials) and `--attribution-trace` seeds evolution with
  attribution-driven candidates (`attr-recovery-hint`, `attr-skill_top_k-2`,
  `attr-retry_max_retries-3`).
- Bounded memory is real: `WorkingMemory` (in-run, TTL, keyword retrieval,
  eviction) and `PersistentMemory` (JSONL-backed cross-session store with TTL
  and freshness checks). `Agent(memory=...)` injects relevant prior outcomes
  at the start of a run and persists a run summary afterwards.
- The Anthropic provider now implements `complete_stream`, so streaming is
  supported on all three backends (OpenAI Responses, DeepSeek chat,
  Anthropic Messages).
- Bounded Multi-Agent is real: `miniclaude/reviewer.py` provides an
  LLM-backed Reviewer second pass (`build_review_verifier`); the CLI wires it
  with `--review`. A reviewer rejection is fed back into the main loop (like
  a failed verification) and the agent gets a bounded chance to address the
  comments before finalizing.
- AsyncIO is real at the I/O and orchestration boundaries (the core loop
  stays synchronous for determinism):
  - `AsyncOpenAIProvider` (`miniclaude/llm/async_openai_provider.py`) drives
    Responses / chat endpoints through `openai.AsyncOpenAI` with `acomplete`
    and `acomplete_stream`;
  - `RunInLoopProvider` bridges an async provider to the sync agent contract
    with a per-instance event loop;
  - `python -m evaluation.coding.async_runner` runs the live benchmark with
    bounded asyncio concurrency (`--concurrency`, Semaphore + `to_thread`),
    producing a `live_async` report.
- The built-in toolset grew to 13 with three read-only, workspace-confined
  tools: `file_tree` (depth-limited recursive listing), `todo_scan`
  (TODO/FIXME/HACK markers), and `file_stat` (metadata and line counts).
- Real Multi-Agent collaboration (`miniclaude/agents/`):
  - `CollaborationBlackboard`: thread-safe shared evidence store with
    provenance, content dedup, query, and verification marks;
  - `SpecialistAgent`: one agent scoped to a subtask with its own prompt and
    tool subset, publishing answers and tool observations to the blackboard;
  - `CoordinatorAgent`: deterministic task decomposition, concurrent
    specialist execution, workspace cross-check of evidence, synthesized
    final answer, and an optional `Critic` second pass;
  - CLI `--multi-agent` runs the pipeline end to end.

## 5.2.0 - 2026-08-14

- The bounded loop now records an explicit phase per decision
  (`AgentResult.phases`: plan / act / reflect / verify / finalize).
- Real Plan stage: by default the first model text response is treated as a
  plan (`plan_first=True`), so the loop runs Plan -> Act -> Verify ->
  Finalize; disable it for single-shot completion.
- Harness-driven Verify stage: when files were edited and a verifier is
  configured, the driver runs verification before finalizing. A failed
  verification is returned to the model as a `verification` decision; a
  passed verification finalizes without an extra model round trip. Tool
  rounds with failures are marked with the `reflect` phase (their attribution
  data feeds recovery rate).
- Trace now derives explicit `file_modified` events from successful
  write/replace observations and records `checkpoint_built` when a
  checkpoint is captured.
- CLI `--verify` runs `pytest` before finalizing when the workspace was
  edited during the run.
- Tool activation: tools declare `activation_keywords`; the driver activates
  a task-matched subset (plus tools declared by selected skills) and only
  sends those schemas, with a full-toolset fallback and used-tool expansion.
  `RunMetrics.tools_sent` / `average_tools_per_turn` quantify the context
  saved; disable with `--no-tool-gating` or `Agent(tool_gating=False)`.
- Skill routing upgraded to keyword recall + TF-IDF cosine reranking
  (`SkillRegistry.select(task, mode="hybrid")`), with a pure `semantic` mode
  and a `keyword` mode for comparison; skills can declare `tools:` front
  matter used by tool activation.
- Added `evaluation.skill_routing` (`miniclaude-skill-eval`): measures
  keyword vs hybrid routing hit rates on the 26-task benchmark and persists a
  versioned report under `reports/`. Skill trigger vocabulary was calibrated
  against the benchmark corpus: keyword and hybrid both route 26/26
  (`reports/skill-routing-2026-08-14-3863285d31.json`).
- Progressive context compression, applied per layer and measured in
  `RunMetrics.context_compression`: `stale_snip` (drops superseded
  snapshot-type tool outputs), `micro_compact` (head + tail of oversized
  outputs with a marker), and optional `auto_compact` (a `summarizer`
  callback folds the oldest tool outputs into a summary).
- Freshness-checked file read cache (`miniclaude/memory.py`): reads are
  served from memory keyed by path + mtime + size and invalidated by
  harness writes; `read_file` now returns
  `{path, content, cache_hit}` and `RunMetrics.cache_hits` /
  `cache_hit_rate` quantify the effect.
- Concurrent tool dispatch: read-only batches run on a bounded thread pool
  (max 4), while batches containing mutating calls stay sequential so
  approvals and write ordering remain deterministic. No DAG dependency
  analysis is claimed. `ApprovalManager` is now thread-safe;
  `RunMetrics.parallel_batches` / `max_parallelism` record the effect.
- Session checkpoints now carry provider-local state (`provider_state`),
  which preserves assistant `tool_calls` and tool `tool_call_id` links on the
  DeepSeek chat resume path. The CLI gained `--resume` (loads a checkpoint
  for `--session-id`) and auto-saves a checkpoint whenever a run ends without
  completing.
- Third provider adapter: `AnthropicProvider` (Messages API) normalizes text
  and `tool_use` blocks into the same `LLMResponse` boundary; select it with
  `--provider anthropic` (`ANTHROPIC_API_KEY`).
- Provider-level streaming: `OpenAIProvider.complete_stream()` yields text
  deltas for both the Responses and chat paths. The agent loop keeps its
  synchronous `complete()` contract so the state machine stays unit-testable.
- Rich CLI: v5 non-JSON output renders a metrics table (falls back to plain
  JSON when Rich is unavailable).
- Benchmark-driven strategy evolution (`evaluation/evolution.py`,
  `miniclaude-evolve`): the agent's behavior is parametrized by a versioned
  `StrategyConfig` (skill top-k, context budget, micro-compaction, retry
  policy, tool gating, plan-first, system prompt) that is actually wired into
  the live benchmark runner. Candidates are generated deterministically,
  scored on a fixed category-stratified training split, and promoted only
  when they improve a fixed holdout split without regressing task success;
  otherwise the run keeps/rolls back to the base. A dry-run mode estimates
  per-turn context cost offline (`reports/evolution-*-dry-run-*.json`).
- Workspace snapshots now exclude `.git`, `.venv`, `venv`, and
  `node_modules`, so `workspace_diff` and tool-creation scans stay fast and
  clean.
- Minimal MCP stdio client (`miniclaude/mcp/`): JSON-RPC
  initialize/tools/list/tools/call over stdio. MCP tools are exposed as
  `ToolDefinition`s and default to `MUTATING` risk, so every call still goes
  through the existing ALLOW/ASK/DENY funnel. Tool inventory is now "10
  built-in + MCP pluggable".
- Retry backoff now applies full jitter by default
  (`OpenAIProviderConfig.retry_jitter`, default true); set it to false for
  deterministic backoff delays.
- Added Recovery Rate to per-run metrics and the coding benchmark report:
  the share of failed tool calls followed by a later successful call to the
  same tool (`recoverable_failures` / `recovered_failures` / `recovery_rate`
  in `RunMetrics`, `CodingCaseResult`, and `CodingReport`).
- Added versioned report storage and A/B comparison (`evaluation/reporting.py`
  plus the `miniclaude-report` CLI): every benchmark run keeps a
  content-addressed copy and a `latest-` pointer under `reports/`, and any
  two reports can be diffed as JSON or Markdown.
- The coding benchmark runner now persists a versioned copy of every run
  (live and `--validate-only`) into `reports/` automatically.
- Version metadata synchronized: package version is now 5.2.0 (pyproject and
  `miniclaude.__version__` were still at 5.0.0 while the changelog described
  unreleased 5.1.0 work).
- Pytest no longer uses the on-disk cache provider
  (`-p no:cacheprovider`) to avoid Windows cache-path conflicts.

## 5.1.0 - 2026-08-13

- Added per-run metrics (turns, tool success rate, policy decisions, token
  usage, latency, repeated reads, context truncation) assembled from trace and
  provider usage into `AgentResult.metrics`.
- Added optional cost estimation via `CostCalculator` with per-model pricing.
- Added on-demand skills (`skills/<name>/SKILL.md`): discovered by
  `SkillRegistry`, selected by task keywords, and injected into context only
  when matched; loaded skills are recorded in `AgentResult.skills` and trace.
- Added `arguments` to `ToolObservation` for auditability and repeated-read
  metrics (kept out of the model-visible observation output).
- Added a repo-level coding benchmark (`evaluation.coding`): 26 tasks across
  7 categories with deterministic offline validation and an opt-in live
  runner reporting task success, tool success, first-pass, turns, tokens,
  latency, cost, approval accuracy, repeated reads, and truncation rate.
- CLI/JSON/session output now include metrics and skills; pricing is
  configurable through `MINICLAUDE_INPUT_PRICE_PER_1M` /
  `MINICLAUDE_OUTPUT_PRICE_PER_1M`.
- Added retry/backoff to the OpenAI provider for transient failures
  (timeouts, connection errors, HTTP 408/429/5xx) with configurable attempts
  via `MINICLAUDE_MAX_RETRIES` (default 2, exponential backoff).
- Added a `workspace_diff` tool that returns the unified diff of files changed
  since the session started (cache artifacts excluded), enabling diff-aware
  context without requiring a git repository.
- Added session resume: `SessionCheckpoint` +
  `SessionStore.save_checkpoint/load_checkpoint`, `ContextManager.restore`,
  and `Agent.build_checkpoint/resume` continue an interrupted run across
  processes. The Responses API path resumes through `previous_response_id`;
  the DeepSeek chat path seeds its provider-side message history.

## 5.0.0 - 2026-08-12

- Added a bounded, structured agent loop.
- Added a provider-neutral LLM interface and OpenAI Responses API adapter.
- Added schema-validated tools and structured observations.
- Added allow/ask/deny security policy and exact-call approvals.
- Added workspace-confined local and Docker command runtimes.
- Added file exploration, exact editing, command, and read-only Git tools.
- Added project instructions, context trimming, detailed traces, and session records.
- Added deterministic offline evaluation and regression thresholds.
- Preserved the v4.1 `python main.py "task"` fallback when no model is configured.

