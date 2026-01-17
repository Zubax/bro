"""Shopify MCP integration for Bro.

This module provides access to Shopify Admin API through the Model Context Protocol (MCP).
"""

import logging
import os
from typing import Any

from bro.mcp.client import SyncMCPManager, StdioMCPClient

_logger = logging.getLogger(__name__)


class ShopifyClient:
    def __init__(
        self,
        access_token: str,
        domain: str,
    ) -> None:
        """
        Initialize Shopify MCP client.

        Args:
            access_token: Shopify Admin API access token
            domain: Shopify store domain (e.g., your-store.myshopify.com)
        """
        self._mcp_manager = SyncMCPManager()

        # Find mcp-shopify executable
        import shutil

        mcp_shopify_path = shutil.which("mcp-shopify")
        if not mcp_shopify_path:
            raise FileNotFoundError("mcp-shopify not found. Install it with: npm install -g @akson/mcp-shopify")

        # Build command
        command = [mcp_shopify_path]

        # Initialize MCP client
        shopify_client = StdioMCPClient(
            name="shopify",
            command=command,
            env={
                "SHOPIFY_ACCESS_TOKEN": access_token,
                "SHOPIFY_DOMAIN": domain,
            },
        )

        try:
            self._mcp_manager.add_client(shopify_client)
            tools = self._mcp_manager.get_all_tools()
            _logger.info(f"Shopify MCP client initialized with {len(tools)} tools for domain: {domain}")
        except Exception as e:
            _logger.error(f"Failed to initialize Shopify MCP client: {e}")
            raise

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Get all available tools from Shopify MCP server.

        Returns:
            List of tool definitions in OpenAI function format
        """
        return self._mcp_manager.get_all_tools()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Call a Shopify tool.

        Args:
            name: Tool name (e.g., "get_orders", "search_products")
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
            error_msg = f"Failed to call Shopify tool '{name}': {e}"
            _logger.error(error_msg)
            return error_msg

    def close(self) -> None:
        """Close the MCP client and cleanup resources."""
        try:
            self._mcp_manager.close_all()
            _logger.info("Shopify MCP client closed")
        except Exception as e:
            _logger.warning(f"Error closing Shopify MCP client: {e}")
