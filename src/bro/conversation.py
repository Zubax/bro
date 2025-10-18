import json
import logging
from dataclasses import dataclass
from typing import Any

import openai
import yaml
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from bro.connector import Message, Connector, Channel
from bro.reasoner import Context, Reasoner, StepResultCompleted, StepResultNothingToDo, StepResultInProgress

_logger = logging.getLogger(__name__)

# Need better examples.
_OPENAI_CONVERSATION_PROMPT = """
You are a confident autonomous AI agent named Bro, designed to complete complex tasks using the reasoner tool. 
The reasoner is a computer-use agent that can complete arbitrary tasks on the local computer like a human would.
It can analyze data, search the Web, write and run programs, and do anything else you would expect a human user to do.

An example of what the reasoner can do is open browser, giving summary of a document or web page.
An example of what it cannot do is run periodic activities or actions that involve delays, such as schedule a task, 
or click then wait.

You should handle all tasks independently, without asking for permission.
Delegate only complex or high-level reasoning tasks to the reasoner when necessary.

All messages MUST follow the schema defined below:
```
via: "<channel name>"
user: "<user name>"
---
<user message verbatim>
```

Important:
- When writing a prompt for the reasoner, provide only the goal, not step-by-step instructions.
- There is no need to check the reasoner’s status before calling task_reasoner.
"""

tools = [
    {
        "type": "function",
        "name": "task_reasoner",
        # More detailed, provide examples.
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
        "description": "Update users on current task progress.",
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


class ConversationHandler:
    """
    This class handles receiving messages, replying to them, and delegating tasks to the Reasoner.
    """

    def __init__(self, connector: Connector, user_system_prompt: str, client: OpenAI, reasoner: Reasoner) -> None:
        self._current_task = None
        self._user_system_prompt = user_system_prompt
        self.connector = connector
        self._context = self._build_system_prompt()
        self._client = client
        self._reasoner = reasoner

    def _build_system_prompt(self) -> list[dict[str, Any]]:
        ctx = [
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

    def _process(self, item: dict[str, Any]) -> tuple[str, None] | tuple[None, str]:
        _logger.debug(f"Processing item: {item}")
        msg, text = None, None
        match item["type"]:
            case "message":
                msg = item["content"][0]["text"]

            case "reasoning":
                for x in item["summary"]:
                    if x.get("type") == "summary_text":
                        _logger.info(f"💭 {x['text']}")

            case "function_call":
                name, args = item["name"], json.loads(item["arguments"])
                match name:
                    case "task_reasoner":
                        # TODO: add a way to interrupt the current task.
                        prompt = json.loads(item["arguments"])["prompt"]
                        channel = json.loads(item["arguments"])["channel"]
                        _logger.info(f"Reasoner prompt:  {prompt}")
                        if self._reasoner.task(Context(prompt=prompt, files=[])):
                            self._current_task = Task(summary=prompt, channel=Channel(name=channel))
                            _logger.info(f"Current task is set. Updates will be sent to channel {channel}.")
                            text = "Successfully tasked the reasoner."
                        else:
                            text = "Sorry the reasoner is busy. Please try again later."
                    case "get_reasoner_status":
                        msg = self._reasoner.legilimens()
                    case _:
                        text = f"ERROR: Unrecognized function call: {name!r}({args})"
                        _logger.error(f"Unrecognized function call: {name!r}({args})")
        return msg, text

    def spin(self) -> bool:
        msgs = self.connector.poll()
        _logger.info("Polling...")
        # TODO (architectural issue): The reasoner should be stepped from a separate thread,
        # TODO such that we can still talk to the user(s) while the reasoner is busy.
        match self._reasoner.step():
            case StepResultCompleted(message):
                _logger.warning("🏁 " * 40 + "\n" + message)
                self.connector.send(Message(text=message, attachments=[]), via=self._current_task.channel)
                self._context += [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": message},
                        ],
                    },
                ]
            case StepResultInProgress():
                pass
            case StepResultNothingToDo():
                pass
        if msgs:
            for msg in msgs:
                _logger.info(f"Processing text from user: {msg}")
                input_data = f"""
                via: {msg.via.name} 
                user: {msg.user.name}
                ---
                {msg.text}
                """
                self._context += [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": input_data},
                        ],
                    },
                ]
                response = self._request_inference(self._context)
                output = response["output"]
                if not output:
                    _logger.warning("No output from model; response: %s", response)

                for out in output:
                    if out.get("type") == "reasoning":
                        del out["status"]

                self._context += output
                for item in output:
                    _logger.info(f"Received output item: {item}")
                    msg_data, text = self._process(item)
                    if msg_data:
                        _logger.info(f"Received message data: {msg_data}.")
                        try:
                            metadata, text = msg_data.split("\n---\n", 1)
                            metadata = yaml.safe_load(metadata)
                            via, user = metadata["via"], metadata["user"]
                        except (AttributeError, KeyError, TypeError) as e:
                            _logger.error(f"Wrong message format. Error: {e}")
                            return
                        except Exception as e:
                            _logger.error(f"Unknown error: {e}")
                            return
                        if item.get("call_id"):
                            self._context += [
                                {
                                    "type": "function_call_output",
                                    "call_id": item["call_id"],
                                    "output": msg,
                                }
                            ]
                        self.connector.send(Message(text=text, attachments=[]), Channel(name=via))
                    elif text:
                        _logger.error(f"Received text: {text}.")
                        self.connector.send(Message(text=text, attachments=[]), self._current_task.channel)
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
    def _request_inference(self, ctx: list[dict[str, Any]]) -> dict[str, Any]:
        _logger.debug(f"Requesting inference with {len(ctx)} context items...")
        # noinspection PyTypeChecker
        return self._client.responses.create(
            model="gpt-5",
            input=ctx,
            tools=tools,
            reasoning={"effort": "low", "summary": "detailed"},
            text={"verbosity": "low"},
            service_tier="default",
            truncation="auto",
        ).model_dump()
