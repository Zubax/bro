"""Shopify MCP integration for Bro.

This module provides access to Shopify Admin API through the Model Context Protocol (MCP).
"""

import logging
import os
from typing import Any

from bro.mcp.client import SyncMCPManager, StdioMCPClient

_logger = logging.getLogger(__name__)


class ShopifyClient:
    """
    Client for accessing Shopify Admin API through MCP.

    This client uses the shopify-mcp-server (https://github.com/antoineschaller/shopify-mcp-server)
    to provide access to Shopify store data including products, orders, customers, and inventory.

    Setup:
    1. Install mcp-shopify: npm install -g @akson/mcp-shopify
    2. Create a custom app in your Shopify admin with appropriate API scopes:
       - read_products, write_products
       - read_orders, write_orders
       - read_customers, write_customers
       - etc.
    3. Set environment variables:
       - SHOPIFY_ACCESS_TOKEN: Your Admin API access token
       - SHOPIFY_DOMAIN: Your store domain (e.g., your-store.myshopify.com)

    The client will automatically manage the MCP server lifecycle and tool routing.
    """

    def __init__(
        self,
        access_token: str | None = None,
        domain: str | None = None,
    ) -> None:
        """
        Initialize Shopify MCP client.

        Args:
            access_token: Shopify Admin API access token (auto-loaded from SHOPIFY_ACCESS_TOKEN env var if not provided)
            domain: Shopify store domain (auto-loaded from SHOPIFY_DOMAIN env var if not provided)
        """
        self._mcp_manager = SyncMCPManager()

        # Load credentials from environment if not provided
        access_token = access_token or os.getenv("SHOPIFY_ACCESS_TOKEN")
        domain = domain or os.getenv("SHOPIFY_DOMAIN")

        if not access_token:
            raise ValueError(
                "Shopify access token not found. Please set SHOPIFY_ACCESS_TOKEN environment variable "
                "or provide it as an argument."
            )

        if not domain:
            raise ValueError(
                "Shopify domain not found. Please set SHOPIFY_DOMAIN environment variable "
                "or provide it as an argument. Format: your-store.myshopify.com"
            )

        # Find mcp-shopify executable
        mcp_shopify_path = "/opt/homebrew/bin/mcp-shopify"
        if not os.path.exists(mcp_shopify_path):
            # Try to find it in PATH
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
