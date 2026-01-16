"""MCP (Model Context Protocol) client implementation for Bro.

This module provides a unified interface for connecting to MCP servers
(both stdio and HTTP transports) and exposing their tools to AI models.
"""

import asyncio
import json
import logging
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPClient(ABC):
    """Abstract base class for MCP clients."""

    def __init__(self, name: str):
        self.name = name
        self._tools: List[Dict[str, Any]] = []
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the MCP connection and fetch available tools."""
        pass

    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the MCP connection."""
        pass

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools from this MCP server."""
        return self._tools


class StdioMCPClient(MCPClient):
    """MCP client that communicates via stdio (standard input/output)."""

    def __init__(self, name: str, command: List[str], env: Optional[Dict[str, str]] = None):
        """
        Initialize a stdio-based MCP client.

        Args:
            name: Name of this MCP server
            command: Command to start the MCP server (e.g., ['uvx', 'workspace-mcp', '--tools', 'gmail'])
            env: Optional environment variables to pass to the server
        """
        super().__init__(name)
        self.command = command
        self.env = env or {}
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0

    async def initialize(self) -> None:
        """Start the MCP server process and fetch available tools."""
        if self._initialized:
            return

        logger.info(f"Starting MCP server '{self.name}' with command: {' '.join(self.command)}")

        # Start the subprocess
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**self.env},
        )

        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "bro", "version": "1.0.0"},
            },
        }

        response = await self._send_request(init_request)
        logger.info(f"MCP server '{self.name}' initialized: {response}")

        # Send initialized notification
        initialized_notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        await self._send_notification(initialized_notification)

        # Fetch available tools
        tools_request = {"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list", "params": {}}

        tools_response = await self._send_request(tools_request)

        if "result" in tools_response and "tools" in tools_response["result"]:
            self._tools = tools_response["result"]["tools"]
            logger.info(f"MCP server '{self.name}' has {len(self._tools)} tools: {[t['name'] for t in self._tools]}")

        self._initialized = True

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server via stdio."""
        if not self._initialized:
            raise RuntimeError(f"MCP client '{self.name}' not initialized")

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }

        response = await self._send_request(request)

        if "error" in response:
            raise RuntimeError(f"MCP tool call failed: {response['error']}")

        return response.get("result")

    async def close(self) -> None:
        """Close the MCP server process."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        self._initialized = False

    def _next_id(self) -> int:
        """Generate next request ID."""
        self._request_id += 1
        return self._request_id

    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("MCP process not running")

        # Send request
        request_line = json.dumps(request) + "\n"
        self._process.stdin.write(request_line)
        self._process.stdin.flush()

        # Read response
        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("MCP server closed connection")

        return json.loads(response_line)

    async def _send_notification(self, notification: Dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("MCP process not running")

        notification_line = json.dumps(notification) + "\n"
        self._process.stdin.write(notification_line)
        self._process.stdin.flush()


class MCPManager:
    """Manages multiple MCP client connections."""

    def __init__(self):
        self._clients: Dict[str, MCPClient] = {}
        self._tool_to_client: Dict[str, str] = {}  # Map tool name to client name

    async def add_client(self, client: MCPClient) -> None:
        """Add and initialize an MCP client."""
        await client.initialize()
        self._clients[client.name] = client

        # Map tools to this client
        for tool in client.get_tools():
            tool_name = tool["name"]
            if tool_name in self._tool_to_client:
                logger.warning(
                    f"Tool '{tool_name}' already registered by client '{self._tool_to_client[tool_name]}', overriding with '{client.name}'"
                )
            self._tool_to_client[tool_name] = client.name

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get combined list of all tools from all MCP clients, converted to OpenAI format."""
        all_tools = []
        for client in self._clients.values():
            for tool in client.get_tools():
                # Convert MCP tool format to OpenAI function format (old style to match existing tools)
                openai_tool = {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {}),
                }
                all_tools.append(openai_tool)
        return all_tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by routing to the appropriate MCP client."""
        if name not in self._tool_to_client:
            raise ValueError(f"Tool '{name}' not found in any MCP client")

        client_name = self._tool_to_client[name]
        client = self._clients[client_name]

        return await client.call_tool(name, arguments)

    async def close_all(self) -> None:
        """Close all MCP client connections."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
        self._tool_to_client.clear()


# Synchronous wrapper for use in non-async code
class SyncMCPManager:
    """Synchronous wrapper around MCPManager for use in non-async contexts."""

    def __init__(self):
        self._manager = MCPManager()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def add_client(self, client: MCPClient) -> None:
        """Add and initialize an MCP client (synchronous)."""
        loop = self._get_loop()
        loop.run_until_complete(self._manager.add_client(client))

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get combined list of all tools from all MCP clients."""
        return self._manager.get_all_tools()

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by routing to the appropriate MCP client (synchronous)."""
        loop = self._get_loop()
        return loop.run_until_complete(self._manager.call_tool(name, arguments))

    def close_all(self) -> None:
        """Close all MCP client connections (synchronous)."""
        loop = self._get_loop()
        loop.run_until_complete(self._manager.close_all())
