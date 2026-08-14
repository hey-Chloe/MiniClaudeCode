"""Minimal MCP stdio client that exposes MCP tools as ToolDefinitions."""

from miniclaude.mcp.client import (
    MCPClient,
    MCPError,
    MCPServerConfig,
)

__all__ = ["MCPClient", "MCPError", "MCPServerConfig"]
