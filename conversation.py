import json
import logging
from typing import Any

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from bro.connector import Message, Connector
from bro.reasoner import Context, Reasoner, StepResultCompleted, StepResultNothingToDo, StepResultInProgress

_logger = logging.getLogger(__name__)

# TODO Define a YAML message schema using plain strings: https://yaml-multiline.info/
_OPENAI_CONVERSATION_PROMPT = """
You are a confident and autonomous AI agent named Bro, designed to complete complex tasks using the reasoner.
The reasoner can perform actions such as searching the web, reading remote files, and controlling the computer.

You should handle all user interactions and simpler tasks independently, without asking for permission.
Delegate only complex or high-level reasoning tasks to the reasoner when necessary.

Important:
- When writing a prompt for the reasoner, provide only the goal, not step-by-step instructions.
- There is no need to check the reasoner’s status before calling task_reasoner.
"""

tools = [
    {
        "type": "function",
        "name": "task_reasoner",
        "description": "Activate the Bro reasoner by providing a summary of the user's goal and necessary context",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "describe what the user wants to do with the needed context."
                },
            },
            "required": ["prompt"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "get_reasoner_status",
        "description": "Update users on current task progress.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        },
        "strict": True
    }

]


class ConversationHandler:
    """
    This class handles receiving messages, replying to them, and delegating tasks to the Reasoner.
    """

    def __init__(self, connector: Connector, user_system_prompt: str, client: OpenAI, reasoner: Reasoner) -> None:
        self._current_channel = None
        self._user_system_prompt = user_system_prompt
        self.connector = connector
        self._context = self._build_system_prompt()
        self._client = client
        self._reasoner = reasoner
        # todo add better logging throughout

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

    def _process(self, item: dict[str, Any]) -> str | None:
        _logger.debug(f"Processing item: {item}")
        msg = None
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
                        _logger.info(f"Reasoner prompt: {prompt}")
                        if self._reasoner.task(Context(prompt=prompt, files=[])):
                            msg = "Successfully tasked the reasoner."
                        else:
                            msg = "Sorry the reasoner is busy. Please try again later."
                    case "get_reasoner_status":
                        msg = self._reasoner.legilimens()
                    case _:
                        msg = f"ERROR: Unrecognized function call: {name!r}({args})"
                        _logger.error(f"Unrecognized function call: {name!r}({args})")
        return msg

    def spin(self) -> None:
        msgs = self.connector.poll()
        _logger.info("Polling...")
        match self._reasoner.step():
            case StepResultCompleted(message):
                _logger.warning("🏁 " * 40 + "\n" + message)
                self.connector.send(Message(text=message, attachments=[]), via=self._current_channel)
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
                _logger.info(msg)
                self._context += [
                    {
                        "role": "user",
                        "content": [
                            # TODO SPECIFY WHERE THE MESSAGE COMES FROM (use JSON?)
                            {"type": "input_text", "text": msg.text},
                        ],
                    },
                ]
                self._current_channel = msg.via
                _logger.info(f"Current channel is set to {self._current_channel}")
                response = self._request_inference(self._context)
                output = response["output"]
                if not output:
                    _logger.warning("No output from model; response: %s", response)

                for out in output:
                    if out.get("type") == "reasoning":
                        del out["status"]

                self._context += output
                for item in output:
                    msg = self._process(item)
                    if msg:
                        self._context += [
                            {
                                "type": "function_call_output",
                                "call_id": item["call_id"],
                                "output": json.dumps(msg),
                            }
                        ]
                        _logger.info(f"Received response: {msg}")
                        self.connector.send(Message(text=msg, attachments=[]), self._current_channel)
        # TODO: attach file to ctx

    @retry(
        reraise=True,
        stop=stop_after_attempt(12),
        wait=wait_exponential(),
        retry=(retry_if_exception_type(openai.OpenAIError)),
        before_sleep=before_sleep_log(_logger, logging.ERROR),
    )
    def _request_inference(
            self,
            ctx: list[dict[str, Any]]
    ) -> dict[str, Any]:
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
