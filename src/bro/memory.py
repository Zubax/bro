import logging
import os

from openmemory import OpenMemory

from bro.brofiles import MEMORY_DB

_ADDING_AND_RECALLING_MEMORY = """When adding new information to the memory, you need to also decide the tags of the 
information. The tags are a list containing the information sector and keywords related to the context of the 
conversation. There are 5 types of information sectors: EPISODIC (Events & Experiences), SEMANTIC (Facts & 
Knowledge), PROCEDURAL (Skills & How-to), EMOTIONAL (Feelings & Sentiment) and REFLECTIVE (Meta-cognition & 
Insights). For example:

```
{
    "text": "My morning routine: coffee, then check emails, then code",
    "tags": ["routine", "procedural"]
},
{
    "text": "I feel really excited about the new AI project",
    "tags": ["emotion", "ai"],
},
{
    "text": "The company address is at Tallinn, Estonia", 
    "tags": ["episodic", "semantics"],
}
```

When querying the long term memory you need to decide which sectors to query. For example:

```
{
    "text": "What is our company address?",
    "sectors": ["procedural", "reflective", "semantic"]
},
{
    "text": "How to flash GNSS using Dr Watson?",
    "sectors": ["procedural", "reflective"]
},
{
    "text": "What is the user preferred language for communication?",
    "sectors": ["episodic", "reflective", "semantic"],
},

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

_memory = OpenMemory(
    mode="local",
    path=MEMORY_DB,
    tier="smart",
    embeddings={"provider": "openai", "apiKey": os.getenv("OPENAI_API_KEY"), "model": "text-embedding-3-small"},
    reflection={"enabled": True, "intervalMinutes": 1440, "minMemories": 10},  # Daily
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

        case ("recall", {"query": query, "sectors": sectors}):
            _logger.info(f"Querying memories in sectors {sectors}...")
            results = _memory.query(query, filters={"sectors": sectors})
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
