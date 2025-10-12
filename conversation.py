import json
import logging
from typing import Any

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from bro.connector import Message, Connector
from bro.reasoner import Context, Reasoner, StepResultCompleted, StepResultNothingToDo, StepResultInProgress

_logger = logging.getLogger(__name__)

_OPENAI_CONVERSATION_PROMPT = """
You are an assistant working alongside an AI agent named Bro, responsible for handling complex reasoning tasks.
Your role is to engage with users, and delegate more complex tasks (e.g., controlling the computer) to Bro.

You have access to the following tools:
- task_reasoner: a function that activates the Bro reasoner by providing a task summary.
- get_reasoner_status: a function that checks whether Bro is ready to take on new tasks.

You can pass tasks to Bro using the task_reasoner tool and update users about task progress using get_reasoner_status.

Important:
To users, there is no distinction between you and Bro. To them, you are Bro.
"""

tools = [
    {
        "type": "function",
        "name": "task_reasoner",
        "description": "Give a new task to the reasoner with the needed context.",
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
        "description": "Check the status of the task the reasoner is given.",
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
    This class handles receiving messages and replying.
    """

    def __init__(self, connector: Connector, user_system_prompt: str, client: OpenAI, reasoner: Reasoner):
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

    def _process(self, item: dict[str, Any]) -> tuple[
        list[dict[str, Any]],
        str | None,
    ]:
        _logger.debug(f"Processing item: {item}")
        match item["type"]:
            case "message":
                msg = item["content"][0]["text"]
                return [], msg

            case "reasoning":
                for x in item["summary"]:
                    if x.get("type") == "summary_text":
                        _logger.info(f"💭 {x['text']}")
                return [], None

            case "function_call":
                name, args = item["name"], json.loads(item["arguments"])
                result, msg = None, None
                match name:
                    case "task_reasoner":
                        # TODO: add a way to interrupt the current task.
                        prompt = json.loads(item["arguments"])["prompt"]
                        msg = "Sorry the reasoner is busy. Please try again later."
                        _logger.info(f"Reasoner prompt: {prompt}")
                        if self._reasoner.task(Context(prompt=prompt, files=[])):
                            msg = "Successfully tasked the reasoner."
                    case "get_reasoner_status":
                        msg = self._reasoner.legilimens()
                    case _:
                        result = f"ERROR: Unrecognized function call: {name!r}({args})"
                        _logger.error(f"Unrecognized function call: {name!r}({args})")
                return [
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": json.dumps(result),
                    }
                ], msg

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
                    new_ctx, text = self._process(item)
                    self._context += new_ctx
                    if text:
                        _logger.info(f"Received response: {text}")
                        self.connector.send(Message(text=text, attachments=[]), msg.via)
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
