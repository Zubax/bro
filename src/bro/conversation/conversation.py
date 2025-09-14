import os
import logging
from time import sleep
from typing import Any

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from bro.connector import Channel, Message, Messaging

_logger = logging.getLogger(__name__)
slack_bot_token, slack_app_token = os.environ["SLACK_BOT_TOKEN"], os.environ["SLACK_APP_TOKEN"]

_OPENAI_CONVERSATION_PROMPT = """
You are a Slack bot talking to multiple people in Slack. You get your response from another reasoning model. 
"""


def _generate_response(msg: str) -> Message:
    response = ""
    return Message(text=response, attachments=[])


class ConversationHandler:
    """
    This class handles receiving message from slack and replying
    """

    def __init__(self, connector: Messaging, user_system_prompt: str, client: OpenAI):
        self._user_system_prompt = user_system_prompt
        self.connector = connector
        self._context = self._build_system_prompt()
        self._client = client

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

    def start(self):
        while True:
            msgs = self.connector.poll()
            if msgs:
                for msg in msgs:
                    self._context += [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": msg},
                            ],
                        },
                    ]
            sleep(60)
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
            tools=[],  # TODO: add tools: interpreter and web search, plus agent?
            reasoning={"effort": "low", "summary": "detailed"},
            text={"verbosity": "low"},
            service_tier="default",
            truncation="auto",
        ).model_dump()


def demo():
    from bro.connector.slack import SlackConnector

    OpenAI_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    connector = SlackConnector(bot_token=slack_bot_token, app_token=slack_app_token)
    conversation = ConversationHandler(connector, "User-defined prompt in prompt.txt", OpenAI_client)
    conversation.start()


if __name__ == "__main__":
    demo()
