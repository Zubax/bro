import os
import logging
from time import sleep
from typing import Any

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from bro.connector import Message, Connecting

_logger = logging.getLogger(__name__)
slack_bot_token, slack_app_token = os.environ["SLACK_BOT_TOKEN"], os.environ["SLACK_APP_TOKEN"]

_OPENAI_CONVERSATION_PROMPT = """
You are a Slack bot talking to multiple people in Slack. You get your response from another reasoning model. 
"""


class ConversationHandler:
    """
    This class handles receiving message from slack and replying
    """

    def __init__(self, connector: Connecting, user_system_prompt: str, client: OpenAI):
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
        interval = 60
        while True:
            msgs = self.connector.poll()
            _logger.info("Polling...")
            for remaining in range(interval, 0, -10):
                _logger.info(f"Next poll in {remaining} seconds")
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
                        response = self._request_inference(self._context, tools=[], reasoning_effort="minimal")
                        reflection = response["output"][-1]["content"][0]["text"]
                        _logger.info(f"Received response: {reflection}")
                        self.connector.send(Message(text=reflection, attachments=[]), msg.via)
                sleep(10)
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
    from bro.connector.slack import SlackConnecting
    from bro import logs
    from bro.brofiles import LOG_FILE, DB_FILE

    logs.setup(log_file=LOG_FILE, db_file=DB_FILE)
    _logger.setLevel(logging.DEBUG)

    OpenAI_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    connector = SlackConnecting(bot_token=slack_bot_token, app_token=slack_app_token)
    conversation = ConversationHandler(connector, _OPENAI_CONVERSATION_PROMPT, OpenAI_client)
    conversation.start()


if __name__ == "__main__":
    demo()
