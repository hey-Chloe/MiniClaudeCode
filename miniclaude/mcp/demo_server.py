"""Bundled demo MCP server (stdio JSON-RPC) for MiniClaudeCode.

Run it standalone with ``python -m miniclaude.mcp.demo_server`` or attach it
to the CLI with ``--mcp-demo``. It exposes three tools:

- ``demo_echo`` (read-only): echoes the supplied text back;
- ``demo_read_file`` (read-only): reads a text file inside the demo root;
- ``demo_append_note`` (mutating): appends a line to ``notes.md``.

The server declares ``annotations.readOnlyHint`` on its read-only tools, so
the client maps them to ``ToolRisk.READ_ONLY``; everything else keeps the
client-configured risk (``MUTATING`` by default) and still passes through the
ALLOW/ASK/DENY funnel.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "2024-11-05"
_DEMO_ROOT = Path(
    os.environ.get("MINICLAUDE_DEMO_ROOT") or Path.cwd()
).resolve()


def _tool(
    name: str, description: str, parameters: dict[str, Any], read_only: bool
) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "name": name,
        "description": description,
        "inputSchema": parameters,
    }
    if read_only:
        tool["annotations"] = {"readOnlyHint": True, "destructiveHint": False}
    else:
        tool["annotations"] = {"readOnlyHint": False, "destructiveHint": True}
    return tool


def _tools() -> list[dict[str, Any]]:
    return [
        _tool(
            "demo_echo",
            "Echo the supplied text back to the caller.",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            read_only=True,
        ),
        _tool(
            "demo_read_file",
            "Read a UTF-8 text file inside the demo root.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            read_only=True,
        ),
        _tool(
            "demo_append_note",
            "Append one line to notes.md inside the demo root.",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            read_only=False,
        ),
    ]


def _resolve(path: str) -> Path:
    target = (_DEMO_ROOT / path).resolve()
    if target != _DEMO_ROOT and _DEMO_ROOT not in target.parents:
        raise ValueError(f"path escapes the demo root: {path}")
    return target


def _handle(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "miniclaude-demo", "version": "1.0.0"},
        }
    if method == "tools/list":
        return {"tools": _tools()}
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name == "demo_echo":
            return {
                "content": [
                    {"type": "text", "text": "echo:" + str(arguments.get("text", ""))}
                ]
            }
        if name == "demo_read_file":
            target = _resolve(str(arguments.get("path", "")))
            if not target.is_file():
                raise ValueError(f"not a file: {arguments.get('path')}")
            content = target.read_text(encoding="utf-8", errors="replace")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"path": str(arguments.get("path")), "content": content},
                            ensure_ascii=False,
                        ),
                    }
                ]
            }
        if name == "demo_append_note":
            target = _DEMO_ROOT / "notes.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            line = str(arguments.get("text", "")).strip()
            if not line:
                raise ValueError("note text must not be empty")
            with target.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"path": "notes.md", "appended": True},
                            ensure_ascii=False,
                        ),
                    }
                ]
            }
        raise ValueError(f"unknown tool: {name}")
    raise ValueError(f"unknown method: {method}")


def main() -> int:
    for raw in sys.stdin:
        if not raw.strip():
            continue
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            continue
        request_id = message.get("id")
        try:
            result = _handle(
                message.get("method", ""),
                message.get("params") or {},
            )
            response: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": str(exc)},
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
