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
    }
    # {
    #     "type": "function",
    #     "name": "get_reasoner_status",
    #     "description": "Provide a summary of the current internal state of the reasoner.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {},
    #         "additionalProperties": False
    #     },
    #     "strict": True
    # }
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
                    _logger.info(response["output"])
                    for item in response["output"]:
                        if item["type"] == "function_call":
                            if item["name"] == "task_reasoner":
                                _logger.info(json.loads(item["arguments"]))
                                self._reasoner.task(Context(prompt=json.loads(item["arguments"])["prompt"], files=[]))
                                reasoner_status = self._reasoner.legilimens()
                                self._context += [
                                    {
                                        "type": "function_call_output",
                                        "call_id": item["call_id"],
                                        "output": json.dumps({
                                            "status": reasoner_status
                                        })
                                    },
                                ]
                        if response["output"][-1].get("content"):
                            reflection = response["output"][-1]["content"][0]["text"]
                            _logger.info(f"Received response: {reflection}")
                            self.connector.send(Message(text=reflection, attachments=[]), msg.via)

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
