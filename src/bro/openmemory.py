import logging
import os

from openmemory import OpenMemory

from bro.brofiles import MEMORY_FILE

tools = [
    {
        "type": "function",
        "name": "remember",
        "description": "Add memory by text.",  # TODO BETTER DESCRIPTION WITH EXAMPLES; CHECK MCP OR EXISTING TOOLS
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the information to be memorized."},
            },
            "required": ["text"],
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
    result = ""
    match name, args:
        case ("remember", {"text": text}):
            try:
                _memory.add(text)
                result = "memory is added."
            except Exception as e:
                result = f"Sorry memory can't be added. Error: {e}"

        case ("recall", {"query": query}):
            _logger.info("Searching the memory...")
            results = _memory.query(query)
            if not results:
                result = "Sorry the memory is empty."
            else:
                result = results[0]["content"]

    return result
