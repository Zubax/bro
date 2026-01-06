"""
Wiki.js API client for accessing Wiki.js instances.

This module provides functionality to fetch content from any Wiki.js instance
using API token authentication. Configure the wiki URL and API token via
environment variables or constructor parameters.
"""

import logging
import os
from typing import Any
import requests

_logger = logging.getLogger(__name__)

_WIKI_INSTRUCTIONS = """Use these functions to access the configured Wiki.js instance which may contain 
documentation, technical guides, and knowledge base articles.

IMPORTANT - Memory Integration:
1. Before searching with wiki_search, ALWAYS use the recall tool first to check if the wiki page path 
   is already known from previous searches. Query memory with sectors ["semantic", "procedural"].
   Example: recall(query="wiki path for configuration guide", sectors=["semantic", "procedural"])

2. After successfully finding a wiki page path (either from search or fetch), ALWAYS use the remember 
   tool to store the mapping between the topic and the page path for future reference.
   Example: remember(text="Wiki page for configuration guide is at public/documentation/getting-started", 
                     tags=["semantic", "wiki", "configuration", "guide"])

This avoids redundant searches and builds up knowledge of the wiki structure over time.

Examples of searching the wiki:

```
{
    "query": "configuration guide"
}
```

Examples of fetching a specific page by path:

```
{
    "path": "public/documentation/getting-started"
}
```

Note: The path structure may vary depending on the Wiki.js instance configuration.
"""

tools = [
    {
        "type": "function",
        "name": "wiki_search",
        "description": _WIKI_INSTRUCTIONS
        + "\n\nBefore using this function, check memory first using recall() to see if the wiki path is already known. "
        + "After finding results, use remember() to store the page paths for future reference."
        + "\n\nNOTE: Wiki.js native search can be limited. If this returns no results or you can't find what you're "
        + "looking for, use wiki_list_pages() to get all pages and search through them yourself.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to find relevant wiki pages."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "wiki_list_pages",
        "description": _WIKI_INSTRUCTIONS
        + "\n\nThis function returns a complete list of all pages in the Wiki with their titles, paths, and descriptions. "
        + "Use this when wiki_search() fails to find what you're looking for, or when you need to browse available pages. "
        + "You can then search through the results yourself by looking for keywords in titles, paths, or descriptions.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "wiki_fetch_page",
        "description": _WIKI_INSTRUCTIONS
        + "\n\nAfter successfully fetching a page, use remember() to store the mapping between the topic "
        + "and page path so you can find it quickly next time.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the wiki page (e.g., 'public/development/python-conventions').",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class WikiClient:
    """
    Client for accessing any Wiki.js instance via its GraphQL API.

    The API token and wiki URL should be provided via environment variables:
    - BRO_WIKI_API_TOKEN: API token for authentication
    - BRO_WIKI_URL: Base URL of the Wiki.js instance (optional, defaults to https://wiki.zubax.com)

    Alternatively, both can be passed as constructor parameters.
    """

    def __init__(self, api_token: str | None = None, base_url: str | None = None) -> None:
        """
        Initialize the wiki client.

        Args:
            api_token: API token for authentication. If not provided, will try to read from
                      BRO_WIKI_API_TOKEN environment variable.
            base_url: Base URL of the Wiki.js instance. If not provided, will try to read from
                      BRO_WIKI_URL environment variable, or fall back to https://wiki.zubax.com

        Raises:
            ValueError: If no API token is provided or found in environment.
        """
        self._api_token = api_token or os.getenv("BRO_WIKI_API_TOKEN")
        if not self._api_token:
            raise ValueError(
                "Wiki API token not provided. Set BRO_WIKI_API_TOKEN environment variable "
                "or pass api_token parameter."
            )

        # Get base URL from parameter, environment variable, or use default
        self._base_url = (base_url or os.getenv("BRO_WIKI_URL") or "https://wiki.zubax.com").rstrip("/")

        self._graphql_endpoint = f"{self._base_url}/graphql"
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._api_token}",
                "Content-Type": "application/json",
            }
        )

        _logger.info(f"Initialized WikiClient for {self._base_url}")

    def list_pages(self) -> str:
        """
        List all pages in the wiki.

        Returns:
            Formatted string containing all pages with titles, paths, and descriptions
        """
        graphql_query = """
        query {
            pages {
                list(orderBy: TITLE) {
                    id
                    path
                    title
                    description
                    locale
                }
            }
        }
        """

        _logger.info("Fetching list of all wiki pages")

        try:
            results = self._execute_graphql(graphql_query, {}, "pages.list")

            if not results:
                return "No pages found in the wiki."

            # Format results as a readable string
            output = [f"Found {len(results)} pages in the wiki:\n"]
            for i, page in enumerate(results, 1):
                title = page.get("title", "Untitled")
                path = page.get("path", "")
                description = page.get("description", "No description")
                output.append(f"{i}. {title}")
                output.append(f"   Path: {path}")
                output.append(f"   Description: {description}\n")

            return "\n".join(output)

        except Exception as e:
            error_msg = f"Failed to list wiki pages: {e}"
            _logger.error(error_msg)
            return error_msg

    def search(self, query: str) -> str:
        """
        Search for wiki pages matching a query.

        Args:
            query: Search query string

        Returns:
            Formatted string containing search results with titles, descriptions, and paths
        """
        graphql_query = """
        query ($query: String!) {
            pages {
                search(query: $query) {
                    results {
                        id
                        path
                        title
                        description
                        locale
                    }
                }
            }
        }
        """

        variables = {"query": query}
        _logger.info(f"Searching wiki for: {query}")

        try:
            result = self._execute_graphql(graphql_query, variables, "pages.search")
            results = result.get("results", [])

            if not results:
                return f"No wiki pages found matching query: {query}"

            # Format results as a readable string
            output = [f"Found {len(results)} wiki pages matching '{query}':\n"]
            for i, page in enumerate(results, 1):
                title = page.get("title", "Untitled")
                path = page.get("path", "")
                description = page.get("description", "No description")
                output.append(f"{i}. {title}")
                output.append(f"   Path: {path}")
                output.append(f"   Description: {description}\n")

            return "\n".join(output)

        except Exception as e:
            error_msg = f"Failed to search wiki: {e}"
            _logger.error(error_msg)
            return error_msg

    def fetch_page(self, path: str, locale: str = "en") -> str:
        """
        Fetch a wiki page by its path.

        Args:
            path: Path to the page (e.g., "public/development/python-conventions")
            locale: Language locale (default: "en")

        Returns:
            Formatted string containing the page title and content
        """
        graphql_query = """
        query ($path: String!, $locale: String!) {
            pages {
                singleByPath(path: $path, locale: $locale) {
                    id
                    path
                    title
                    description
                    content
                    contentType
                    createdAt
                    updatedAt
                }
            }
        }
        """

        # Remove leading slash if present
        path = path.lstrip("/")

        variables = {"path": path, "locale": locale}
        _logger.info(f"Fetching wiki page: {path}")

        try:
            page = self._execute_graphql(graphql_query, variables, "pages.singleByPath")

            if not page:
                return f"Wiki page not found: {path}"

            title = page.get("title", "Untitled")
            content = page.get("content", "No content available")
            description = page.get("description", "")

            # Format the page content
            output = [f"# {title}"]
            if description:
                output.append(f"\n{description}\n")
            output.append(f"\n{content}")

            return "\n".join(output)

        except Exception as e:
            error_msg = f"Failed to fetch wiki page '{path}': {e}"
            _logger.error(error_msg)
            return error_msg

    def _execute_graphql(
        self, query: str, variables: dict[str, Any], result_path: str
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Execute a GraphQL query.

        Args:
            query: The GraphQL query string
            variables: Variables for the query
            result_path: Dot-separated path to extract result from response (e.g., "pages.single")

        Returns:
            The extracted result from the GraphQL response

        Raises:
            requests.HTTPError: If the request fails
            ValueError: If the response contains errors
        """
        payload = {"query": query, "variables": variables}

        try:
            response = self._session.post(self._graphql_endpoint, json=payload)
            response.raise_for_status()

            data = response.json()

            # Check for GraphQL errors
            if "errors" in data:
                error_messages = [err.get("message", str(err)) for err in data["errors"]]
                error_str = "; ".join(error_messages)
                _logger.error(f"GraphQL errors: {error_str}")
                raise ValueError(f"GraphQL errors: {error_str}")

            # Extract result using the result_path
            result = data.get("data", {})
            for key in result_path.split("."):
                result = result.get(key, {})

            return result

        except requests.HTTPError as e:
            _logger.error(f"HTTP error during GraphQL request: {e.response.status_code}")
            raise
        except requests.RequestException as e:
            _logger.error(f"Network error during GraphQL request: {e}")
            raise

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()
        _logger.debug("Closed WikiClient session")
