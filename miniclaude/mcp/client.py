"""Minimal Model Context Protocol (MCP) client over stdio.

Implements just enough JSON-RPC to ``initialize`` a server, list its tools,
and call them. MCP tools are exposed through the same ``ToolDefinition``
boundary and default to ``MUTATING`` so every call still passes through the
existing ALLOW/ASK/DENY funnel.
"""

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Callable

from miniclaude.tools import ToolDefinition, ToolRisk


class MCPError(RuntimeError):
    """Raised when the MCP server cannot be reached or returns an error."""


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    """How to launch one MCP server as a subprocess."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    risk: ToolRisk = ToolRisk.MUTATING


class MCPClient:
    """Owns one stdio subprocess and speaks JSON-RPC 2.0 to it."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._tools: dict[str, ToolDefinition] = {}

    def start(self) -> "MCPClient":
        if self._process is not None:
            return self
        environment = os.environ.copy()
        for key, value in self.config.env:
            environment[key] = value
        try:
            self._process = subprocess.Popen(
                [self.config.command, *self.config.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env=environment,
            )
        except OSError as exc:
            raise MCPError(
                f"MCP server '{self.config.name}' could not start: {exc}"
            ) from exc
        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "miniclaude", "version": "5.2.0"},
            },
        )
        return self

    def list_tools(self) -> list[ToolDefinition]:
        """Discover the server's tools as validated ToolDefinitions."""
        self.start()
        result = self._request("tools/list", {}) or {}
        tools: list[ToolDefinition] = []
        for item in result.get("tools", []):
            name = item.get("name")
            if not name:
                continue
            tool = ToolDefinition(
                name=str(name),
                description=str(item.get("description") or ""),
                parameters=(
                    item.get("inputSchema")
                    or {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    }
                ),
                handler=self._handler(str(name)),
                risk=self.config.risk,
            )
            self._tools[str(name)] = tool
            tools.append(tool)
        return tools

    def _handler(self, name: str) -> Callable[..., Any]:
        def call(**arguments: Any) -> Any:
            result = self._request(
                "tools/call",
                {"name": name, "arguments": arguments},
            ) or {}
            content = result.get("content") or []
            return "".join(
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            )

        return call

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            process.stdin.close()
        except OSError:
            pass
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()

    def __enter__(self) -> "MCPClient":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        process = self._process
        if process is None:
            raise MCPError("MCP client is not started")
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            payload: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
            }
            if params:
                payload["params"] = params
            if process.stdin is None or process.stdout is None:
                raise MCPError("MCP server streams are unavailable")
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()
            while True:
                line = process.stdout.readline()
                if not line:
                    raise MCPError(
                        f"MCP server '{self.config.name}' closed the stream"
                    )
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MCPError(
                        f"MCP server sent invalid JSON: {line!r}"
                    ) from exc
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    error = message["error"] or {}
                    raise MCPError(
                        f"MCP request '{method}' failed: "
                        f"{error.get('message', 'unknown error')}"
                    )
                return message.get("result")
