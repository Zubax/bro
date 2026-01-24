import logging
from openmemory import OpenMemory
from bro.brofiles import MEMORY_DB

_ADDING_AND_RECALLING_MEMORY = """When adding new information to the memory, you need to also decide the tags of the 
information. The tags are a list containing the information sector and keywords related to the context of the 
conversation. There are 5 types of information sectors: EPISODIC (Events & Experiences), SEMANTIC (Facts & 
Knowledge), PROCEDURAL (Skills & How-to), EMOTIONAL (Feelings & Sentiment) and REFLECTIVE (Meta-cognition & 
Insights). 

IMPORTANT: For large data (documents, reports, transaction logs, etc.), store only the FILE LOCATION in memory, 
not the full content. The memory acts as an index to help you find where information is stored on the filesystem.

Examples of adding information to memory:

```
{
    "text": "Bishop Q4 2024 financial forecast stored in /home/user/documents/bishop_forecast_q4_2024.xlsx",
    "tags": ["semantic", "bishop", "financial", "forecast", "q4-2024"]
},
{
    "text": "Bank transaction exports from Luminor are in /home/user/accounting/luminor_transactions_2024/",
    "tags": ["semantic", "procedural", "luminor", "bank", "transactions", "accounting"]
},
{
    "text": "Company tax filing procedure: use form 1234, submit via e-Tax portal by March 31",
    "tags": ["procedural", "tax", "filing", "deadline", "estonia"]
},
{
    "text": "Customer contract templates are stored in ~/templates/contracts/, use template_standard_2024.docx for new clients",
    "tags": ["procedural", "semantic", "contracts", "templates", "customers"]
},
{
    "text": "GNSS firmware binaries location: /opt/zubax/firmware/gnss/, use flash_tool.py with --verify flag",
    "tags": ["procedural", "gnss", "firmware", "flashing", "zubax"]
},
```

When querying the long term memory you need to decide which sectors to query. Examples:

```
{
    "query": "Where is the Bishop financial forecast?",
    "sectors": ["semantic"]
},
{
    "query": "Where are the bank transaction exports?",
    "sectors": ["semantic", "procedural"]
},
{
    "query": "What is the tax filing procedure?",
    "sectors": ["procedural", "semantic"]
}
```
"""

tools = [
    {
        "type": "function",
        "name": "remember",
        "description": _ADDING_AND_RECALLING_MEMORY,
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
        "description": _ADDING_AND_RECALLING_MEMORY,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the question to be answered."},
                "sectors": {"type": "string", "description": "the list of sectors the information could belong to."},
            },
            "required": ["query", "sectors"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

_logger = logging.getLogger(__name__)


class Memory:
    def __init__(self, api_key: str | None):
        self._memory = OpenMemory(
            mode="local",
            path=MEMORY_DB,
            tier="smart",
            embeddings={"provider": "openai", "apiKey": api_key, "model": "text-embedding-3-small"},
            reflection={"enabled": True, "intervalMinutes": 1440, "minMemories": 10},  # Daily
        )

    def recall(self, query: str, sectors: list[str]) -> str:
        _logger.info(f"Querying memories in sectors {sectors}...")
        results = self._memory.query(query, filters={"sectors": sectors})
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

    def remember(self, text: str, tags: list[str]) -> str:
        _logger.info(f"Adding memory with the following tags {tags}...")
        try:
            mem = self._memory.add(text, tags=tags)
            memory_id = str(mem["id"])  # Explicitly cast to str
            _logger.debug(f"Memory stored. Memory id {memory_id}")
            return memory_id
        except Exception as e:
            return f"Memory can't be added. Error: {e}"

    def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        try:
            self._memory.delete(memory_id)
            _logger.info(f"Deleted memory {memory_id}")
            return True
        except Exception as e:
            _logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
