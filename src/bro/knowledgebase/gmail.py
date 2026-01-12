"""
Gmail MCP client for accessing Gmail through Model Context Protocol.

This module provides functionality to search, read, and send emails using the
Gmail MCP server (https://github.com/GongRzhe/Gmail-MCP-Server).
"""

import asyncio
import json
import logging
import os
from typing import Any

from bro.knowledgebase import KnowledgeBase

_logger = logging.getLogger(__name__)

_GMAIL_INSTRUCTIONS = """Use these functions to access Gmail for searching and reading emails, and sending messages.

IMPORTANT - Memory Integration:
1. Before searching for emails, use recall() to check if you already have relevant email information 
   from previous searches. Query memory with sectors ["semantic", "episodic"].
   Example: recall(query="email from John about project", sectors=["semantic", "episodic"])

2. After successfully finding relevant emails, use remember() to store important information 
   (but NOT full email content - only key facts, file locations, or references).
   Example: remember(text="Project proposal email from John received on 2024-01-05, discussion about budget in gmail",
                     tags=["semantic", "episodic", "email", "john", "project", "proposal"])

3. For emails with attachments or important documents, store the reference/location in memory, not the content.

IMPORTANT - Knowledge Retrieval Priority:
When asked about company information, procedures, or technical documentation:
1. FIRST check the Wiki using wiki_search() - Wiki is the primary source for official documentation
2. ONLY use Gmail search if Wiki doesn't have the information OR if looking for recent communications
3. Gmail is best for: recent conversations, specific correspondence, follow-ups, and time-sensitive info

Examples of searching emails:

```
{
    "query": "from:john@example.com subject:budget",
    "max_results": 10
}
```

Examples of reading a specific email:

```
{
    "message_id": "18d1a2b3c4d5e6f7"
}
```

Examples of sending an email:

```
{
    "to": "colleague@example.com",
    "subject": "Project Update",
    "body": "Here's the latest update on the project..."
}
```

Note: Use Gmail search operators for precise results (from:, to:, subject:, after:, before:, has:attachment, etc.)
"""

tools = [
    {
        "type": "function",
        "name": "gmail_search",
        "description": _GMAIL_INSTRUCTIONS
        + "\n\nBefore using this function, check memory first using recall() to see if you already have the information. "
        + "After finding results, use remember() to store key facts (not full email content) for future reference."
        + "\n\nIMPORTANT: Check Wiki FIRST for official documentation. Only search Gmail for recent communications.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Gmail search query using search operators (e.g., 'from:user@example.com subject:invoice')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "gmail_read",
        "description": _GMAIL_INSTRUCTIONS
        + "\n\nAfter reading an email with important information, use remember() to store key facts "
        + "(but not the full email content) for future reference.",
        "parameters": {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "The Gmail message ID to read",
                },
            },
            "required": ["message_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "gmail_send",
        "description": _GMAIL_INSTRUCTIONS
        + "\n\nUse this to send emails when you need to communicate with humans or respond to requests.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line",
                },
                "body": {
                    "type": "string",
                    "description": "Email body content (plain text or HTML)",
                },
                "cc": {
                    "type": "string",
                    "description": "CC recipients (comma-separated)",
                },
                "bcc": {
                    "type": "string",
                    "description": "BCC recipients (comma-separated)",
                },
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class GmailClient(KnowledgeBase):
    """
    Client for accessing Gmail through the Model Context Protocol (MCP) server.

    The Gmail MCP server (https://github.com/GongRzhe/Gmail-MCP-Server) should be
    installed on the same machine. This client spawns and communicates with it via stdio.

    Setup:
    1. Install Node.js and npm
    2. Install Gmail MCP: npm install -g @gongrzhe/server-gmail-mcp
    3. Run OAuth setup once: npx @gongrzhe/server-gmail-mcp (interactive)
    4. Bro will automatically spawn and manage the server process

    Environment Variables:
    - BRO_GMAIL_MCP_COMMAND: Custom command to run MCP server (optional)
      Default: "npx @gongrzhe/server-gmail-mcp"
    """

    def __init__(self, mcp_command: str | None = None) -> None:
        """
        Initialize the Gmail MCP client.

        Args:
            mcp_command: Command to run the MCP server. If not provided, uses default.
        """
        self._mcp_command = mcp_command or os.getenv("BRO_GMAIL_MCP_COMMAND", "npx @gongrzhe/server-gmail-mcp")
        self._session: Any = None
        self._read_stream: Any = None
        self._write_stream: Any = None

        _logger.info(f"Initialized GmailClient with MCP command: {self._mcp_command}")

    def _ensure_connection(self) -> None:
        """Ensure MCP session is connected, creating it if necessary."""
        if self._session is not None:
            return

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as e:
            raise RuntimeError("MCP SDK not available. Install with: pip install mcp") from e

        # Parse command into executable and args
        command_parts = self._mcp_command.split()
        if not command_parts:
            raise ValueError("Empty MCP command")

        server_params = StdioServerParameters(
            command=command_parts[0],
            args=command_parts[1:] if len(command_parts) > 1 else [],
            env=None,
        )

        async def connect() -> None:
            """Async connection function."""
            self._read_stream, self._write_stream = stdio_client(server_params)
            self._session = ClientSession(self._read_stream, self._write_stream)
            await self._session.initialize()
            _logger.info("Gmail MCP session initialized successfully")

        # Run async connection in new event loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(connect())
        finally:
            # Don't close the loop, we'll reuse it for subsequent calls
            pass

    def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Call an MCP tool synchronously.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result

        Raises:
            RuntimeError: If MCP session is not initialized or call fails
        """
        self._ensure_connection()

        if self._session is None:
            raise RuntimeError("Gmail MCP session not initialized")

        async def call() -> Any:
            """Async call function."""
            result = await self._session.call_tool(tool_name, arguments)
            return result

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(call())
        finally:
            loop.close()

    def search(self, query: str, max_results: int = 10) -> str:
        """
        Search for emails using Gmail search operators.

        Args:
            query: Gmail search query (supports operators like from:, to:, subject:, etc.)
            max_results: Maximum number of results to return

        Returns:
            Formatted string containing search results with email subjects, senders, and snippets
        """
        _logger.info(f"Searching Gmail for: {query}")

        try:
            result = self._call_tool("search_emails", {"query": query, "maxResults": max_results})

            # Format the result as a readable string
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", str(result))

            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Failed to search Gmail: {e}"
            _logger.error(error_msg)
            return error_msg

    def read(self, message_id: str) -> str:
        """
        Read a specific email by message ID.

        Args:
            message_id: The Gmail message ID

        Returns:
            Formatted string containing the email subject, from, to, date, and body
        """
        _logger.info(f"Reading Gmail message: {message_id}")

        try:
            result = self._call_tool("get_message", {"messageId": message_id})

            # Format the result as a readable string
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", str(result))

            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Failed to read Gmail message '{message_id}': {e}"
            _logger.error(error_msg)
            return error_msg

    def send(self, to: str, subject: str, body: str, cc: str | None = None, bcc: str | None = None) -> str:
        """
        Send an email.

        Args:
            to: Recipient email address
            subject: Email subject line
            body: Email body content
            cc: CC recipients (optional)
            bcc: BCC recipients (optional)

        Returns:
            Status message indicating success or failure
        """
        _logger.info(f"Sending email to: {to}, subject: {subject}")

        try:
            arguments = {
                "to": to,
                "subject": subject,
                "body": body,
            }
            if cc:
                arguments["cc"] = cc
            if bcc:
                arguments["bcc"] = bcc

            result = self._call_tool("send_email", arguments)

            # Format the result as a readable string
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", str(result))

            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Failed to send email: {e}"
            _logger.error(error_msg)
            return error_msg

    def close(self) -> None:
        """Close the MCP session and cleanup resources."""
        if self._session is not None:
            # MCP session cleanup
            _logger.debug("Closed Gmail MCP session")
            self._session = None
