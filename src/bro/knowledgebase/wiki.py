"""
Wiki.js API client for accessing internal wiki.zubax.com pages.

This module provides functionality to fetch content from the internal Zubax wiki
using API token authentication. The wiki is powered by Wiki.js.
"""

import logging
import os
from typing import Any
import requests

_logger = logging.getLogger(__name__)

_WIKI_INSTRUCTIONS = """Use these functions to access the internal Zubax wiki (wiki.zubax.com) which contains 
company documentation, technical guides, and knowledge base articles.

Examples of searching the wiki:

```
{
    "query": "GNSS calibration procedure"
}
```

Examples of fetching a specific page by path:

```
{
    "path": "public/development/python-conventions"
}
```
"""

tools = [
    {
        "type": "function",
        "name": "wiki_search",
        "description": _WIKI_INSTRUCTIONS,
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
        "name": "wiki_fetch_page",
        "description": _WIKI_INSTRUCTIONS,
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
    Client for accessing wiki.zubax.com internal pages via Wiki.js GraphQL API.

    The API token should be provided via the BRO_WIKI_API_TOKEN environment variable.
    """

    def __init__(self, api_token: str | None = None, base_url: str = "https://wiki.zubax.com") -> None:
        """
        Initialize the wiki client.

        Args:
            api_token: API token for authentication. If not provided, will try to read from
                      BRO_WIKI_API_TOKEN environment variable.
            base_url: Base URL of the wiki (default: https://wiki.zubax.com)

        Raises:
            ValueError: If no API token is provided or found in environment.
        """
        self._api_token = api_token or os.getenv("BRO_WIKI_API_TOKEN")
        if not self._api_token:
            raise ValueError(
                "Wiki API token not provided. Set BRO_WIKI_API_TOKEN environment variable "
                "or pass api_token parameter."
            )

        self._base_url = base_url.rstrip("/")
        self._graphql_endpoint = f"{self._base_url}/graphql"
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._api_token}",
                "Content-Type": "application/json",
            }
        )

        _logger.info(f"Initialized WikiClient for {self._base_url}")

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
