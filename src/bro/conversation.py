import base64
import json
import logging
import os.path
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import textwrap

import openai
import yaml
from PIL import Image
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from bro import util
from bro.memory import Memory, tools as memory_tools
from bro.knowledgebase.wiki import WikiClient, tools as wiki_tools

from bro.connector import Message, Connector, Channel, ReceivedMessage, User
from bro.reasoner import Context, Reasoner
from bro.util import prune_context_text_only, image_to_base64, detect_file_format

_logger = logging.getLogger(__name__)

_CONTEXT_EMBEDDING_FILE_MAX_BYTES = 10_000_000


_OPENAI_CONVERSATION_PROMPT = """
You are a confident autonomous AI agent named Bro, designed to complete complex tasks using the reasoner tool. 
The reasoner is a computer-use agent that can complete arbitrary tasks on the local computer like a human would.
It can analyze data, search the Web, write and run programs, and do anything else you would expect a human user to do.

An example of what the reasoner can do is searching the web, compiling reports, entering data into bookkeeping 
software, creating and running programs, installing software, creating user accounts, and so on. An example of what 
it cannot do is run periodic activities or actions that involve delays, such as waiting for events.

You should handle all tasks independently, without asking for permission.
Delegate only complex or high-level reasoning tasks to the reasoner when necessary.

KNOWLEDGE RETRIEVAL STRATEGY - WIKI FIRST:
When users ask questions about procedural knowledge, company information, technical guides, processes, or how-to topics,
you MUST check the Wiki FIRST before delegating to the reasoner or providing an answer. The Wiki is your PRIMARY
source of truth for:
- Company procedures and policies (e.g., "shipping instructions", "onboarding process", "travel policy")
- Technical documentation and guides (e.g., "how to configure X", "API documentation", "setup instructions")
- Process workflows (e.g., "how to file paperwork", "procurement process", "approval workflows")
- Product information and specifications
- Any "how do I", "where is", "what is the process for", "how to" type questions

When you receive such questions, follow this workflow:
1. FIRST use `recall()` to check if you already know the wiki path for this topic [sectors: "semantic", "procedural"]
2. If not in memory, use `wiki_search()` with the initial query
3. If the search returns no results or irrelevant results, you have TWO strategies:
   a) Try `wiki_search()` again with different keywords (broader/narrower terms, synonyms, related terms)
   b) Use `wiki_list_pages()` to get ALL pages and manually search through titles, paths, and descriptions yourself
      - This is the MOST RELIABLE method since Wiki.js native search is very limited
      - Use this after 2-3 failed wiki_search attempts, or immediately if you suspect search won't work
4. Once you find relevant results, use `wiki_fetch_page()` to retrieve the full content
5. Use `remember()` to store the topic-to-path mapping for future reference
6. Provide the answer to the user based on the Wiki content
7. Only delegate to the reasoner if the Wiki doesn't contain the information OR if the task requires actual execution

IMPORTANT: Wiki.js native search is LIMITED and often misses pages even when keywords exist in them.
If wiki_search() fails 2-3 times, immediately use wiki_list_pages() to get ALL pages and search through them yourself.
Be PERSISTENT - the information is likely in the Wiki, you just need to find it.

Do NOT delegate simple information lookup to the reasoner. Handle Wiki queries yourself directly.
Be PROACTIVE - do not wait for users to tell you to check the Wiki.

All messages MUST follow the schema defined below. Attachments field is a list of file paths for files included 
with the message. If there are no attachments, this should be [].
```
via: "<channel name>"
user: "<user name>"
attachments: ["path/to/file1", "path/to/file2", ...]
---
<user message verbatim>
```

SENDING MESSAGES:
You can send messages to any channel or person by formatting your response with the message schema above:
- Set "via" to the target channel name (e.g., "sell-or-die") or user ID
- Set "user" to "Bro" (your name)
- Add file paths to "attachments" if needed
- Put your message content after the "---" separator

Example - posting to a channel:
```
via: "general"
user: "Bro"
attachments: []
---
@channel I need help with this task.
```

You can proactively post messages to channels when you need human input or want to share information.

CRITICAL: You MUST only send ONE message block per response. Do NOT send multiple message blocks to different 
channels in the same response (e.g., one to a channel + one confirmation DM). If you need to send messages to 
multiple destinations (like posting to a channel AND sending a confirmation DM), send them in SEPARATE responses 
- first send one message, then in your next response send the other message.

The computer use agent sends messages under the name `Bro Reasoner`. When you receive a message from the reasoner, 
consider notifying the user by sending an appropriately formatted response with the user name and `via` specified as 
necessary.

Important:
- When writing a prompt for the reasoner, provide only the end goal, not step-by-step instructions.
- There is no need to check the reasoner’s status before calling task_reasoner.
- The reasoner may need multiple iterations to complete a task. Keep the conversation going until the task is done.
"""

_EMAIL_MANAGEMENT_WORKFLOW = """
When checking emails, categorize and handle them as follows:

1. Promotional emails, bills, invoices, receipts:
   - Mark as read: modify_gmail_message_labels with {"remove_label_names": ["UNREAD"]}
   - No further action needed

2. Customer inquiry emails:
   - Use get_gmail_message_content to read the full email content
   - If order-related (customer mentions order number), use Shopify tools to lookup order details
   - Post to the appropriate Slack channel using this template:

```
via: "<channel-name>"
user: "Bro"
attachments: []
---
📧 Customer email needs response

*Question:* <paste customer's question verbatim>

*Order Details:* (only if order-related, otherwise omit this section)
- Order: #<order_number>
- Date: <order_date>
- Items: <item1>, <item2>

What should I tell the customer?
```

   - Wait for team response with the answer
   - Send the email using send_gmail_message
   - Mark as read: modify_gmail_message_labels with {"remove_label_names": ["UNREAD"]}

IMPORTANT: 
- Handle email workflows yourself using Gmail MCP tools. Do NOT delegate to reasoner.
- ALWAYS ask team for the answer before responding to customers
- Only include order details if the inquiry is order-related
- Keep order details minimal: order number, date, and items only
"""

_RESPOND_OR_IGNORE_PROMPT = """
You are given a conversation history between an agentic AI named Bro and a number of human users.
Your objective is to determine if the conversation warrants a response or an action on behalf of Bro.
This would be the case if any of the humans are directly or indirectly addressing Bro,
or responding to one of its earlier posts.
This would not be the case if the human users are merely talking to each other.

In case of ambiguity err toward non-engagement.

The response shall contain a brief summary of the observed conversation history,
followed by a detailed elaboration of whether Bro needs to engage, and why exactly.
Finally, the response shall end with a JSON block following the schema below:

```
{
    "response_required": bool
}
```
"""

_TOOLS = [
    {
        "type": "function",
        "name": "task_reasoner",
        # TODO More detailed, provide examples.
        "description": "Activate the Bro reasoner by providing a summary of the user's goal and necessary context",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "describe what the user wants to do with the needed context.",
                },
                "channel": {"type": "string", "description": "the channel id where the task comes from."},
            },
            "required": ["prompt", "channel"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_reasoner_status",
        "description": "Update users on the current task’s progress. If the response is None, it means there is no "
        "active task and the reasoner has finished its work",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
]


@dataclass(frozen=True)
class Task:
    """
    Remember the channel so that the bot could send updates about the task when it's finished.
    """

    channel: Channel
    summary: str


def _parse_message(msg_data: str) -> tuple[str, str, str, str] | None:
    try:
        metadata, text = msg_data.split("\n---", 1)
        metadata = yaml.safe_load(metadata)
        via, user, attachments = metadata.get("via"), metadata.get("user"), metadata.get("attachments")
    except (AttributeError, KeyError, TypeError) as e:
        _logger.error(f"Wrong message format. Error: {e}")
        return None
    except Exception as e:
        _logger.error(f"Unknown error: {e}")
        return None
    return via, user, text, attachments


class ConversationHandler:
    """
    This class handles receiving messages, replying to them, and delegating tasks to the Reasoner.
    """

    def __init__(
        self,
        connector: Connector,
        user_system_prompt: str | None,
        client: OpenAI,
        reasoner: Reasoner,
        memory: Memory,
        wiki: WikiClient | None = None,
        google_workspace: Any = None,
        shopify: Any = None,
    ) -> None:
        self._msgs: list[ReceivedMessage] = []
        self._current_task: Task | None = None
        self._user_system_prompt = user_system_prompt
        self.connector = connector
        self._context = self._build_system_prompt()
        self._client = client
        self._reasoner = reasoner
        self._reasoner.on_task_completed_cb = self._on_task_completed_cb
        self._memory = memory
        self._wiki = wiki
        self._google_workspace = google_workspace
        self._shopify = shopify

        # Build tool name mapping for MCP clients
        self._mcp_tool_map: dict[str, Any] = {}
        if self._google_workspace:
            for tool in self._google_workspace.get_tools():
                self._mcp_tool_map[tool["name"]] = self._google_workspace
        if self._shopify:
            for tool in self._shopify.get_tools():
                self._mcp_tool_map[tool["name"]] = self._shopify

    def _build_system_prompt(self) -> list[dict[str, Any]]:
        ctx: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": _OPENAI_CONVERSATION_PROMPT},
                ],
            },
        ]
        if self._user_system_prompt:
            ctx[0]["content"].append({"type": "input_text", "text": self._user_system_prompt})
        return ctx

    def _process_response_output(self, output: Any) -> None:
        _logger.info("Processing response output...")
        addendum = output.copy()

        for item in addendum:
            if item.get("type") == "reasoning" and "status" in item:
                del item["status"]
                _logger.debug("Ignoring reasoning message...")
                continue

        self._context += addendum

        # Track if we processed any function calls
        had_function_calls = False

        for item in addendum:
            _logger.info(f"Received item from the conversation model: {item}")
            if item.get("type") == "function_call":
                had_function_calls = True
            msg_data = self._process(item)
            _logger.info(f"After processing, got msg_data: {msg_data}")
            if msg_data:
                if parsed_msg := _parse_message(msg_data):
                    via, user, text, fpaths = parsed_msg
                    _logger.info(f"Message from the conversation model after parsing: {text}.")
                    if fpaths != "":
                        attachments = [Path(file_path.strip()) for file_path in fpaths]
                    else:
                        attachments = []
                    self.connector.send(Message(text=text, attachments=attachments), via=Channel(via))
                else:
                    _logger.error(f"Message can't be parsed. Received data: {msg_data}")
                    # TODO rerunning inference using Tenacity

        if had_function_calls:
            _logger.info("Function calls were processed, requesting follow-up inference...")
            conversation_response = self._request_inference(self._context)
            follow_up_output = conversation_response["output"]
            if follow_up_output:
                self._process_response_output(follow_up_output)

    def _on_task_completed_cb(self, message: str) -> None:
        _logger.warning("🏁 " * 40 + "\n" + message)
        input_data = textwrap.dedent(
            f"""\
        via:  
        user: Bro Reasoner
        attachments: []
        ---
        {message}
        """
        )
        self._context += [
            {
                "type": "message",
                "role": "user",
                "content": input_data,
            }
        ]

        self._current_task = None
        _logger.info("Requesting conversation response after receiving reasoner response...")
        conversation_response = self._request_inference(self._context)
        output = conversation_response["output"]
        if not output:
            _logger.warning("No output from conversation model; response: %s", conversation_response)
        self._process_response_output(output)

    def _process(self, item: dict[str, Any]) -> str | None:
        _logger.debug(f"Processing item: {item}")
        match item:
            case {"type": "message", "content": content}:
                return str(content[0]["text"])

            case {"type": "reasoning"}:
                for x in item["summary"]:
                    if x.get("type") == "summary_text":
                        _logger.debug(f"💭 {x['text']}")

            case {"type": "function_call", "name": name, "arguments": arguments}:
                args = json.loads(arguments)
                _logger.debug(f"Received function call arguments: {args}")
                result = None

                if self._current_task and name != "get_reasoner_status":
                    result = f"Rejected. I am currently working on another task: '{self._current_task.summary}'. Inform the user to wait."
                else:
                    match name, args:
                        case ("task_reasoner", {"prompt": prompt, "channel": channel}):
                            _logger.info("Tasking the reasoner...")
                            _logger.debug(f"Prompt for the reasoner: {prompt}")
                            if self._reasoner.task(Context(prompt=prompt, files=[])):
                                self._current_task = Task(summary=prompt, channel=Channel(name=channel))
                                result = "Successfully tasked the reasoner."
                            else:
                                result = "Failed to task the reasoner."
                        case ("get_reasoner_status", {}):
                            _logger.info("Calling legilimens for task progress...")
                            result = self._reasoner.legilimens()
                            if not self._current_task:
                                _logger.error(
                                    f"Missing current task context. Cannot route message to any channel. Message "
                                    f"content: {result}"
                                )
                            else:
                                self._msgs.append(
                                    ReceivedMessage(
                                        via=self._current_task.channel,
                                        user=User(name="Bro"),
                                        text=f"Send message to the user: {result}",
                                        attachments=[],
                                    )
                                )
                        case ("recall", {"query": query, "sectors": sectors}):
                            result = self._memory.recall(query, sectors)

                        case ("remember", {"text": text, "tags": tags}):
                            result = self._memory.remember(text, tags)

                        case ("wiki_search", {"query": query}):
                            if self._wiki:
                                result = self._wiki.search(query)
                            else:
                                result = "Wiki client not available. Set BRO_WIKI_API_TOKEN environment variable."

                        case ("wiki_list_pages", _):
                            if self._wiki:
                                result = self._wiki.list_pages()
                            else:
                                result = "Wiki client not available. Set BRO_WIKI_API_TOKEN environment variable."

                        case ("wiki_fetch_page", {"path": path}):
                            if self._wiki:
                                result = self._wiki.fetch_page(path)
                            else:
                                result = "Wiki client not available. Set BRO_WIKI_API_TOKEN environment variable."

                        case _:
                            # Try MCP tools
                            if name in self._mcp_tool_map:
                                try:
                                    mcp_client = self._mcp_tool_map[name]
                                    _logger.info(f"Attempting to call MCP tool: {name}")
                                    result = mcp_client.call_tool(name, args)
                                    _logger.info(f"MCP tool result: {result}")
                                except Exception as e:
                                    _logger.error(f"MCP tool call failed: {e}")
                                    result = f"Error calling MCP tool '{name}': {str(e)}"
                            else:
                                _logger.error(f"Unrecognized function call: {name!r}({args})")

                if result:
                    _logger.info(f"Function call result: {result}")
                    self._context += [{"type": "function_call_output", "call_id": item["call_id"], "output": result}]

        return None

    def _determine_response_required(self) -> bool:
        ctx = prune_context_text_only(self._context) + [
            {"role": "user", "content": [{"type": "input_text", "text": _RESPOND_OR_IGNORE_PROMPT}]}
        ]
        response = self._request_inference(ctx, reasoning_effort="none")
        output: str = response["output"][-1]["content"][0]["text"]
        response_required_json = util.split_trailing_json(output)[1]
        if not response_required_json:
            _logger.info(f"Can't determine whether response is required. Default to True.")
            return True
        response_required: bool = response_required_json.get("response_required", True)
        _logger.info(f"Response required: {response_required}")
        return response_required

    def spin(self) -> bool:
        self._msgs = self.connector.poll()
        if self._msgs:
            for msg in self._msgs:
                _logger.info(f"Processing user message: {msg}")
                input_data = textwrap.dedent(
                    f"""\
                via: {msg.via.name!r} 
                user: {msg.user.name!r}
                attachments: {list(map(str, msg.attachments))}
                ---
                {msg.text}
                """
                )
                _logger.debug(f"Adding user message to context.")
                self._context += [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": input_data},
                        ],
                    },
                ]

                for file_path in msg.attachments:
                    text_msg = {
                        "type": "input_text",
                        "text": f"User uploaded this file: {file_path}. Content of the file in the next message.",
                    }
                    file_size = os.path.getsize(file_path)
                    file_format = detect_file_format(file_path)
                    match (file_format, file_size):
                        case ("text/plain", size) if size < _CONTEXT_EMBEDDING_FILE_MAX_BYTES:
                            with open(file_path, "rb") as file_content:
                                self._context += [
                                    {
                                        "role": "user",
                                        "content": [
                                            text_msg,
                                            {"type": "input_text", "text": file_content.read().decode()},
                                        ],
                                    }
                                ]
                        case ("application/pdf", size) if size < _CONTEXT_EMBEDDING_FILE_MAX_BYTES:
                            with open(file_path, "rb") as file_content:
                                file_bytes = base64.b64encode(file_content.read())
                                self._context += [
                                    {
                                        "role": "user",
                                        "content": [
                                            text_msg,
                                            {
                                                "type": "input_file",
                                                "filename": file_path.name,
                                                "file_data": f"data:{file_format};base64,{file_bytes.decode()}",
                                            },
                                        ],
                                    },
                                ]
                        case (fmt, size) if fmt and "image" in fmt and size < _CONTEXT_EMBEDDING_FILE_MAX_BYTES:
                            self._context += [
                                {
                                    "role": "user",
                                    "content": [
                                        text_msg,
                                        {
                                            "type": "input_image",
                                            "image_url": f"data:{fmt};base64,{image_to_base64(Image.open(file_path))}",
                                        },
                                    ],
                                },
                            ]
                        case _:
                            self._context += [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_text",
                                            "text": f"User uploaded this file: {file_path}."
                                            f"File can't be processed because it is too big or file format "
                                            f"isn't supported. Please task the reasoner instead.",
                                        },
                                    ],
                                },
                            ]

                if msg.via.name.startswith("D"):  # always answer messages from direct channel
                    should_respond = True
                else:
                    should_respond = self._determine_response_required()

                if should_respond:
                    _logger.info("Generating response from the conversation model...")
                    conversation_response = self._request_inference(self._context)
                    output = conversation_response["output"]

                    if not output:
                        _logger.warning("No output from model; response: %s", conversation_response)

                    self._process_response_output(output)
            return True
        return False

    @retry(
        reraise=True,
        stop=stop_after_attempt(12),
        wait=wait_exponential(),
        retry=(retry_if_exception_type(openai.OpenAIError)),
        before_sleep=before_sleep_log(_logger, logging.ERROR),
    )
    def _request_inference(
        self, ctx: list[dict[str, Any]], /, *, model: str | None = None, reasoning_effort: str | None = None
    ) -> dict[str, Any]:
        _logger.debug(f"Requesting inference with {len(ctx)} context items...")

        # Build tools list
        tools = _TOOLS + memory_tools + wiki_tools

        # Add Google Workspace tools if available
        if self._google_workspace:
            gw_tools = self._google_workspace.get_tools()
            _logger.info(f"Adding {len(gw_tools)} Google Workspace tools to conversation")
            tools = tools + gw_tools

        # Add Shopify tools if available
        if self._shopify:
            shopify_tools = self._shopify.get_tools()
            _logger.info(f"Adding {len(shopify_tools)} Shopify tools to conversation")
            tools = tools + shopify_tools

        # noinspection PyTypeChecker
        return self._client.responses.create(  # type: ignore
            model=model or "gpt-5.1",
            input=ctx,
            tools=tools,
            reasoning={"effort": reasoning_effort or "low", "summary": "detailed"},
            text={"verbosity": "low"},
            service_tier="default",
            truncation="auto",
        ).model_dump()
