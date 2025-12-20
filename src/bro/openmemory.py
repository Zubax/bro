import logging
import os
from typing import Any

from openmemory import OpenMemory

from bro.brofiles import MEMORY_FILE

_ADDING_MEMORY_PROMPT = """
When adding new information to the memory, you need to also decide the tags of the information. 
The tags are a list containing the information sector and keywords related to the context of the conversation.
There are 5 types of information sectors: EPISODIC, SEMANTIC, PROCEDURAL, EMOTIONAL and REFLECTIVE. 
For example:

```
{
    "text": "My morning routine: coffee, then check emails, then code",
    "tags": ["routine", "procedural"]
},
{
    "text": "I feel really excited about the new AI project",
    "tags": ["emotion", "ai"]
},
{
    "text": "I went to Paris yesterday and loved the Eiffel Tower", 
    "tags": ["episodic", "travel", "paris"]
}
```
"""

tools = [
    {
        "type": "function",
        "name": "remember",
        "description": _ADDING_MEMORY_PROMPT,
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the information to be memorized."},
                "tags": {"type": "string", "description": "the list of information sector and keywords."},
            },
            "required": ["text", "tags"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "recall",
        "description": "Query the long term memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the question to be answered."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

_logger = logging.getLogger(__name__)

_memory = OpenMemory(
    mode="local",
    path=MEMORY_FILE,
    tier="smart",
    embeddings={"provider": "openai", "apiKey": os.getenv("OPENAI_API_KEY"), "model": "text-embedding-3-small"},
)


def memory_handler(name: str, args: str) -> str:
    match name, args:
        case ("remember", {"text": text, "tags": tags}):
            _logger.info("Adding memory...")
            try:
                mem = _memory.add(text, tags=tags)
                _logger.info(f"Memory stored: {mem['id']} with tags {tags}")
                return "Memory is added."
            except Exception as e:
                return f"Memory can't be added. Error: {e}"

        case ("recall", {"query": query}):
            _logger.info("Querying memories...")
            results = _memory.query(query)
            _logger.info(f"Found {len(results)} matching memories:")
            if not results:
                return "Found no matching memories."
            else:
                for i, match in enumerate(results):
                    content_preview = match["content"][:50] + "..." if len(match["content"]) > 50 else match["content"]
                    score = match.get("score", 0)
                    _logger.info(f"     {i + 1}. [score: {score:.3f}] {content_preview}")
                highest_match: dict[str, str] = max(results, key=lambda r: r.get("score", 0))
                return highest_match["content"]

    return "Unrecognized function call."
