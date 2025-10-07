import json
import os
import logging
from time import sleep
from typing import Any

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from bro.connector import Message, Connecting
from bro.reasoner import Context
from bro.reasoner.openai_generic import OpenAiGenericReasoner

_logger = logging.getLogger(__name__)
slack_bot_token, slack_app_token = os.environ["SLACK_BOT_TOKEN"], os.environ["SLACK_APP_TOKEN"]

OPENAI_CONVERSATION_PROMPT = """
You are a bot talking to multiple people in a workspace. 
When you need to do complex work, for example controlling the computer, use the task reasoner tool.
When user asks for the reasoner status, use the get_reasoner_status tool.
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
    This class handles receiving message from slack and replying
    """

    def __init__(self, connector: Connecting, user_system_prompt: str, client: OpenAI, reasoner: OpenAiGenericReasoner):
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
                    {"type": "input_text", "text": OPENAI_CONVERSATION_PROMPT},
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
                        prompt = json.loads(item["arguments"])["prompt"]
                        self._reasoner.task(Context(prompt=prompt, files=[]))
                        result = self._reasoner.legilimens()
                        msg = result["text"]

                    case "get_reasoner_status":
                        result = self._reasoner.legilimens()
                        msg = result["text"]

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

    def start(self):
        interval, step = 30, 10
        while True:
            msgs = self.connector.poll()
            _logger.info("Polling...")
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
                    response = self._request_inference(self._context, tools=tools, reasoning_effort="minimal")
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

            for remaining in range(interval, 0, -step):
                _logger.info(f"Next poll in {remaining} seconds")
                sleep(step)
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
            ctx: list[dict[str, Any]],
            /,
            *,
            model: str | None = None,
            reasoning_effort: str | None = None,
            tools: list[dict[str, Any]] | None = None,
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
