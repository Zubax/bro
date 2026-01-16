"""Google Workspace MCP integration for Bro.

This module provides access to Google Workspace services (Gmail, Calendar, Drive, etc.)
through the Model Context Protocol (MCP).
"""

import json
import logging
import os
import sys
from typing import Any

from bro.mcp.client import SyncMCPManager, StdioMCPClient

_logger = logging.getLogger(__name__)


class GoogleWorkspaceClient:
    """
    Client for accessing Google Workspace services through MCP.

    This client uses the workspace-mcp server (https://github.com/taylorwilsdon/google_workspace_mcp)
    to provide access to Gmail, Calendar, Drive, and ot her Google Workspace services.

    Setup:
    1. Install workspace-mcp: rye add workspace-mcp
    2. Create OAuth credentials in Google Cloud Console (Desktop Application)
    3. Download client_secret.json and place it in one of:
       - Current directory as client_secret.json
       - ~/.google_workspace_mcp/client_secret.json
       - Or set GOOGLE_CLIENT_SECRET_PATH environment variable
    4. Alternatively, set GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET env vars
    5. On first use, follow OAuth flow to authorize (credentials stored in ~/.google_workspace_mcp/credentials/)

    The client will automatically manage the MCP server lifecycle and tool routing.
    """

    def __init__(
        self,
        services: list[str] | None = None,
        tool_tier: str = "core",
        oauth_client_id: str | None = None,
        oauth_client_secret: str | None = None,
    ) -> None:
        """
        Initialize Google Workspace MCP client.

        Args:
            services: List of services to enable (e.g., ["gmail", "calendar", "drive"])
                     If None, only Gmail is enabled by default.
            tool_tier: Tool tier level - "core", "extended", or "complete"
            oauth_client_id: Google OAuth client ID (auto-loaded from client_secret.json or env vars if not provided)
            oauth_client_secret: Google OAuth client secret (auto-loaded from client_secret.json or env vars if not provided)
        """
        self._services = services or ["gmail"]
        self._tool_tier = tool_tier
        self._mcp_manager = SyncMCPManager()

        # Load OAuth credentials from environment or client_secret.json
        if not oauth_client_id or not oauth_client_secret:
            # Check for client_secret.json in standard locations
            client_secret_paths = [
                os.getenv("GOOGLE_CLIENT_SECRET_PATH"),
                "client_secret.json",
                os.path.expanduser("~/.google_workspace_mcp/client_secret.json"),
            ]

            for path in client_secret_paths:
                if path and os.path.exists(path):
                    with open(path, "r") as f:
                        oauth_data = json.load(f)
                        if "installed" in oauth_data:
                            oauth_client_id = oauth_data["installed"]["client_id"]
                            oauth_client_secret = oauth_data["installed"]["client_secret"]
                            break
                        elif "web" in oauth_data:
                            oauth_client_id = oauth_data["web"]["client_id"]
                            oauth_client_secret = oauth_data["web"]["client_secret"]
                            break

        if not oauth_client_id or not oauth_client_secret:
            raise ValueError(
                "OAuth credentials not found. Please set GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET "
                "environment variables, or provide a client_secret.json file."
            )

        # Get workspace-mcp executable path
        venv_bin = os.path.dirname(sys.executable)
        workspace_mcp_path = os.path.join(venv_bin, "workspace-mcp")

        if not os.path.exists(workspace_mcp_path):
            raise FileNotFoundError(
                f"workspace-mcp not found at {workspace_mcp_path}. Install it with: rye add workspace-mcp"
            )

        # Build command
        command = (
            [
                workspace_mcp_path,
                "--tool-tier",
                tool_tier,
                "--tools",
            ]
            + self._services
            + [
                "--single-user",
                "--transport",
                "stdio",
            ]
        )

        # Initialize MCP client
        workspace_client = StdioMCPClient(
            name="google-workspace",
            command=command,
            env={
                "GOOGLE_OAUTH_CLIENT_ID": oauth_client_id,
                "GOOGLE_OAUTH_CLIENT_SECRET": oauth_client_secret,
                "GOOGLE_MCP_CREDENTIALS_DIR": os.path.expanduser("~/.google_workspace_mcp/credentials"),
            },
        )

        try:
            self._mcp_manager.add_client(workspace_client)
            tools = self._mcp_manager.get_all_tools()
            _logger.info(
                f"Google Workspace MCP client initialized with {len(tools)} tools from services: {self._services}"
            )
        except Exception as e:
            _logger.error(f"Failed to initialize Google Workspace MCP client: {e}")
            raise

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Get all available tools from Google Workspace MCP server.

        Returns:
            List of tool definitions in OpenAI function format
        """
        return self._mcp_manager.get_all_tools()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Call a Google Workspace tool.

        Args:
            name: Tool name (e.g., "search_gmail_messages")
            arguments: Tool arguments

        Returns:
            Tool result (text content extracted from MCP response)
        """
        try:
            mcp_result = self._mcp_manager.call_tool(name, arguments)

            # Extract text from MCP result format
            if isinstance(mcp_result, dict) and "content" in mcp_result:
                content = mcp_result["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", str(mcp_result))
                else:
                    return str(mcp_result)
            else:
                return str(mcp_result)
        except Exception as e:
            error_msg = f"Failed to call Google Workspace tool '{name}': {e}"
            _logger.error(error_msg)
            return error_msg

    def close(self) -> None:
        """Close the MCP client and cleanup resources."""
        try:
            self._mcp_manager.close_all()
            _logger.info("Google Workspace MCP client closed")
        except Exception as e:
            _logger.warning(f"Error closing Google Workspace MCP client: {e}")
