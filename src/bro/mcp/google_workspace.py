"""Google Workspace MCP integration for Bro.

This module provides access to Google Workspace services (Gmail, Calendar, Drive, etc.)
through the Model Context Protocol (MCP).
"""

import json
import logging
import os
import sys
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from bro.mcp.client import SyncMCPManager, StdioMCPClient

_logger = logging.getLogger(__name__)


class GoogleWorkspaceClient:
    def __init__(
        self,
        oauth_client_id: str,
        oauth_client_secret: str,
        credentials_dir: str,
        default_user_email: str,
        services: list[str] | None = None,
        tool_tier: str = "core",
    ) -> None:
        """
        Initialize Google Workspace MCP client.

        Args:
            oauth_client_id: Google OAuth client ID
            oauth_client_secret: Google OAuth client secret
            credentials_dir: Directory to store OAuth credentials
            default_user_email: Default user email for accessing mailbox
            services: List of services to enable (e.g., ["gmail", "calendar", "drive"])
                     If None, only Gmail is enabled by default.
            tool_tier: Tool tier level - "core", "extended", or "complete"
        """
        self._services = services or ["gmail"]
        self._tool_tier = tool_tier
        self._default_user_email = default_user_email

        # Keep legacy MCP manager for backward compatibility
        self._mcp_manager = SyncMCPManager()

        # Get workspace-mcp executable path
        venv_bin = os.path.dirname(sys.executable)
        workspace_mcp_path = os.path.join(venv_bin, "workspace-mcp")

        if not os.path.exists(workspace_mcp_path):
            raise FileNotFoundError(
                f"workspace-mcp not found at {workspace_mcp_path}. Install it with: rye add workspace-mcp"
            )

        # Build command and args
        command = workspace_mcp_path
        args = (
            [
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

        # Initialize legacy MCP client for backward compatibility
        _logger.info(f"Setting USER_GOOGLE_EMAIL to: {default_user_email}")
        workspace_client = StdioMCPClient(
            name="google-workspace",
            command=[command] + args,
            env={
                "GOOGLE_OAUTH_CLIENT_ID": oauth_client_id,
                "GOOGLE_OAUTH_CLIENT_SECRET": oauth_client_secret,
                "GOOGLE_MCP_CREDENTIALS_DIR": credentials_dir,
                "USER_GOOGLE_EMAIL": default_user_email,
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

        # Initialize LangChain MultiServerMCPClient
        self._langchain_client = MultiServerMCPClient(
            {
                "google-workspace": {
                    "transport": "stdio",
                    "command": command,
                    "args": args,
                    "env": {
                        "GOOGLE_OAUTH_CLIENT_ID": oauth_client_id,
                        "GOOGLE_OAUTH_CLIENT_SECRET": oauth_client_secret,
                        "GOOGLE_MCP_CREDENTIALS_DIR": credentials_dir,
                        "USER_GOOGLE_EMAIL": default_user_email,
                    },
                }
            }
        )
        _logger.info("LangChain MultiServerMCPClient initialized for Google Workspace")

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
            # Automatically inject or replace user_google_email
            arguments["user_google_email"] = self._default_user_email

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

    def get_langchain_client(self) -> MultiServerMCPClient:
        """
        Get the LangChain MultiServerMCPClient instance.

        Returns:
            MultiServerMCPClient instance for use with LangChain agents
        """
        return self._langchain_client

    def close(self) -> None:
        """Close the MCP client and cleanup resources."""
        try:
            self._mcp_manager.close_all()
            _logger.info("Google Workspace MCP client closed")
        except Exception as e:
            _logger.warning(f"Error closing Google Workspace MCP client: {e}")
