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

All messages MUST follow the schema defined below. Attachments field is a list of file paths for files included 
with the message. If there are no attachments, this should be [].
```
via: "<channel name>"
user: "<user name>"
attachments: ["path/to/file1", "path/to/file2", ...]
---
<user message verbatim>
```

The computer use agent sends messages under the name `Bro Reasoner`. When you receive a message from the reasoner, 
consider notifying the user by sending an appropriately formatted response with the user name and `via` specified as 
necessary.

Important:
- When writing a prompt for the reasoner, provide only the end goal, not step-by-step instructions.
- There is no need to check the reasoner’s status before calling task_reasoner.
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
            _logger.debug(f"Received item from the conversation model: {item}")
            if item.get("type") == "reasoning" and "status" in item:
                del item["status"]
                _logger.debug("Ignoring reasoning message...")
                continue

        self._context += addendum

        for item in output:
            _logger.info(f"Received item from the conversation model: {item}")
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
                match name, args:
                    case ("task_reasoner", {"prompt": prompt, "channel": channel}):
                        # TODO: add a way to interrupt the current task.
                        _logger.info("Tasking the reasoner...")
                        _logger.debug(f"Prompt for the reasoner: {prompt}")
                        if self._reasoner.task(Context(prompt=prompt, files=[])):
                            self._current_task = Task(summary=prompt, channel=Channel(name=channel))
                            result = "Successfully tasked the reasoner."
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

                    case _:
                        _logger.error(f"Unrecognized function call: {name!r}({args})")

                if result:
                    _logger.info(f"Function call result: {result}")
                    self._context += [{"type": "function_call_output", "call_id": item["call_id"], "output": result}]
                    self._on_task_completed_cb(result)

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
        # noinspection PyTypeChecker
        return self._client.responses.create(  # type: ignore
            model=model or "gpt-5.1",
            input=ctx,
            tools=_TOOLS + memory_tools,
            reasoning={"effort": reasoning_effort or "low", "summary": "detailed"},
            text={"verbosity": "low"},
            service_tier="default",
            truncation="auto",
        ).model_dump()
