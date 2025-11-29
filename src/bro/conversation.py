import json
import logging
from dataclasses import dataclass
from typing import Any
import textwrap

import openai
import yaml
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from bro import util
from bro.connector import Message, Connector, Channel, ReceivedMessage, User
from bro.reasoner import Context, Reasoner
from bro.util import prune_context_text_only

_logger = logging.getLogger(__name__)

_OPENAI_CONVERSATION_PROMPT = """
You are a confident autonomous AI agent named Bro, designed to complete complex tasks using the reasoner tool. 
The reasoner is a computer-use agent that can complete arbitrary tasks on the local computer like a human would.
It can analyze data, search the Web, write and run programs, and do anything else you would expect a human user to do.

An example of what the reasoner can do is searching the web, compiling reports, entering data into bookkeeping 
software, creating and running programs, installing software, creating user accounts, and so on. An example of what 
it cannot do is run periodic activities or actions that involve delays, such as waiting for events.

You should handle all tasks independently, without asking for permission.
Delegate only complex or high-level reasoning tasks to the reasoner when necessary.

All messages MUST follow the schema defined below:
```
via: "<channel name>"
user: "<user name>"
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

tools = [
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


def _parse_message(msg_data: str) -> tuple[str, str, str] | None:
    try:
        metadata, text = msg_data.split("\n---\n", 1)
        metadata = yaml.safe_load(metadata)
        via, user = metadata.get("via"), metadata.get("user")
    except (AttributeError, KeyError, TypeError) as e:
        _logger.error(f"Wrong message format. Error: {e}")
        return None
    except Exception as e:
        _logger.error(f"Unknown error: {e}")
        return None
    return via, user, text


class ConversationHandler:
    """
    This class handles receiving messages, replying to them, and delegating tasks to the Reasoner.
    """

    def __init__(
        self, connector: Connector, user_system_prompt: str | None, client: OpenAI, reasoner: Reasoner
    ) -> None:
        self._msgs: list[ReceivedMessage] = []
        self._current_task: Task | None = None
        self._user_system_prompt = user_system_prompt
        self.connector = connector
        self._context = self._build_system_prompt()
        self._client = client
        self._reasoner = reasoner
        self._reasoner.on_task_completed_cb = self._on_task_completed_cb

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
        for item in output:
            _logger.debug(f"Received item from the conversation model: {item}")
            msg_data = self._process(item)
            _logger.debug(f"After processing, got msg_data: {msg_data}")
            if msg_data:
                if parsed_msg := _parse_message(msg_data):
                    via, user, text = parsed_msg
                    _logger.debug(f"Message from the conversation model after parsing: {text}.")
                    self.connector.send(Message(text=text, attachments=[]), via=Channel(via))
                else:
                    _logger.error(f"Message can't be parsed. Received data: {msg_data}")
                    # TODO rerunning inference using Tenacity

    def _on_task_completed_cb(self, message: str) -> None:
        _logger.warning("🏁 " * 40 + "\n" + message)
        input_data = textwrap.dedent(
            f"""\
        via:  
        user: Bro Reasoner
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
        msg = None
        match item:
            case {"type": "message", "content": content}:
                msg = content[0]["text"]

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
                        if result:
                            if not self._current_task:
                                _logger.error(
                                    f"Missing current task context. Cannot route message to any channel. Message "
                                    f"content: {result}"
                                )
                                return None
                            self._msgs.append(
                                ReceivedMessage(
                                    via=self._current_task.channel,
                                    user=User(name="Bro"),
                                    text=result,
                                    attachments=[],
                                )
                            )

                    case _:
                        _logger.error(f"Unrecognized function call: {name!r}({args})")
                _logger.info(f"Function call result: {result}")
                self._context += [{"type": "function_call_output", "call_id": item["call_id"], "output": result}]

        return msg

    def _determine_response_required(self) -> bool:
        ctx = prune_context_text_only(self._context) + [
            {"role": "user", "content": [{"type": "input_text", "text": _RESPOND_OR_IGNORE_PROMPT}]}
        ]
        response = self._request_inference(ctx, model="gpt-5-mini")
        output: str = response["output"][-1]["content"][0]["text"]
        response_required_json = util.split_trailing_json(output)[1]
        response_required = response_required_json.get("response_required", True)
        _logger.info(f"Response required: {response_required}")
        return response_required

    def spin(self) -> bool:
        self._msgs = self.connector.poll()
        _logger.info("Polling for user messages...")
        if self._msgs:
            for msg in self._msgs:
                _logger.debug(f"Processing user message: {msg}")
                input_data = textwrap.dedent(
                    f"""\
                via: {msg.via.name!r} 
                user: {msg.user.name!r}
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
                should_respond = self._determine_response_required()
                if should_respond:
                    _logger.debug("Generating response from the conversation model...")
                    conversation_response = self._request_inference(self._context)
                    output = conversation_response["output"]

                    if not output:
                        _logger.warning("No output from model; response: %s", conversation_response)

                    addendum = output.copy()

                    for item in addendum:
                        _logger.debug(f"Received item from the conversation model: {item}")
                        if item.get("type") == "reasoning" and "status" in item:
                            del item["status"]
                            _logger.debug("Ignoring reasoning message...")
                            continue

                    self._context += addendum

                    for item in output:
                        msg_data = self._process(item)
                        _logger.debug(f"After processing, got msg_data: {msg_data}")
                        if msg_data:
                            parsed_msg = _parse_message(msg_data)
                            if parsed_msg:
                                via, user, text = parsed_msg
                                _logger.debug(f"Message from the reasoner after parsing: {text}.")
                                self.connector.send(Message(text=text, attachments=[]), Channel(name=via))
            return True
        return False
        # TODO: attach file to ctx

    @retry(
        reraise=True,
        stop=stop_after_attempt(12),
        wait=wait_exponential(),
        retry=(retry_if_exception_type(openai.OpenAIError)),
        before_sleep=before_sleep_log(_logger, logging.ERROR),
    )
    def _request_inference(self, ctx: list[dict[str, Any]], /, *, model: str | None = None) -> dict[str, Any]:
        _logger.debug(f"Requesting inference with {len(ctx)} context items...")
        # noinspection PyTypeChecker
        return self._client.responses.create(  # type: ignore
            model=model or "gpt-5.1",
            input=ctx,
            tools=tools,
            reasoning={"effort": "low", "summary": "detailed"},
            text={"verbosity": "low"},
            service_tier="default",
            truncation="auto",
        ).model_dump()
