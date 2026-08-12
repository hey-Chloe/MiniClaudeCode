# Migrating from v4.1 to v5.0

The positional entry point remains valid:

```shell
python main.py "your task"
```

Without `MINICLAUDE_MODEL`, auto mode preserves the v4.1 compatibility trace.
Configure a model to activate the real v5 agent loop:

```powershell
$env:OPENAI_API_KEY = "..."
$env:MINICLAUDE_MODEL = "your-model-id"
python main.py "inspect this repository" --mode v5
```

Important behavioral changes:

- Tool parameters are schema validated.
- File paths are confined to the selected workspace.
- Side effects are denied or require approval according to the permission mode.
- Local process execution is not an operating-system sandbox.
- Docker mode defaults to no network and applies CPU, memory, and PID limits.
- The historic `SandboxRuntime` name now honestly reports `isolated=False`.

