import os
import logging
from threading import Lock
import tempfile
from pathlib import Path
from typing import Any

import requests
from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web import SlackResponse

from bro.connector import Connector, Channel, ReceivedMessage, Message, User

_logger = logging.getLogger(__name__)

ATTACHMENT_FOLDER = tempfile.mkdtemp()
BRO_USER_ID = os.getenv("BRO_USER_ID")


def _download_attachment(url: str) -> Path | None:
    response = requests.get(url)
    file_location = Path(ATTACHMENT_FOLDER) / url.split("/")[-1]
    file_location.write_bytes(response.content)
    _logger.info(f"File is saved at {file_location}")
    return file_location


class SlackConnector(Connector):
    """
    SlackConnector is the logic layer that does the polling, sending, downloading attachments using the Slack Socket Mode API.
    This class needs BOT_TOKEN and APP_TOKEN variables.
    Both can be obtained from Slack app's settings → Basic Information → App-Level Tokens → Generate Token
    """

    def __init__(self, bot_token: str, app_token: str) -> None:
        self._web_client = WebClient(token=bot_token)
        self._client = SocketModeClient(app_token=app_token, web_client=self._web_client)
        self._unread_msgs: list[ReceivedMessage] = []

        self._client.socket_mode_request_listeners.append(self._process_message)
        self._client.connect()
        self._mutex = Lock()

        self._seen_events: set[Any] = set()

    def _process_message(self, client: SocketModeClient, req: SocketModeRequest) -> None:
        with self._mutex:
            if req.type == "events_api":
                response = SocketModeResponse(envelope_id=req.envelope_id)
                client.send_socket_mode_response(response)
                event_id = req.payload["event_id"]
                user_id = req.payload["event"]["user"]
                _logger.debug(f"Received event payload: {req.payload['event']}")
                if event_id in self._seen_events:
                    return
                self._seen_events.add(event_id)
                if req.payload["event"]["type"] == "message":
                    if req.payload["event"]["user"] == BRO_USER_ID:
                        _logger.info("Bro sent a text message.")
                        return None
                    attachments = []
                    text = ""
                    channel_id = req.payload["event"]["channel"]
                    if req.payload["event"].get("text"):
                        text = req.payload["event"]["text"]
                        _logger.info("Received a text message: %s", text)
                    if req.payload["event"].get("subtype") == "file_share":
                        _logger.info("Received one attachment or more.")
                        files = req.payload["event"]["files"]
                        for file in files:
                            try:
                                file_info = self._web_client.files_info(file=file["id"])
                                file_download_url = file_info["file"]["url_private_download"]
                                file_download_path = _download_attachment(url=file_download_url)
                                if file_download_path:
                                    attachments.append(file_download_path)
                            except Exception as ex:
                                _logger.exception(f"Failed to save attachment from {file_download_url!r}: {ex}")
                                return None
                    _logger.info("Received a total of %d attachments." % len(attachments))
                    user_info = self._web_client.users_info(user=user_id)["user"]
                    user_name = user_info["name"]
                    _logger.debug(f"User info: id={user_id}, name={user_name}")
                    self._unread_msgs.append(
                        ReceivedMessage(
                            via=Channel(name=channel_id), user=User(name=user_name), text=text, attachments=attachments
                        )
                    )
                    return None
                return None
            return None

    def list_channels(self) -> list[Channel] | SlackResponse:
        with self._mutex:
            response: SlackResponse = self._web_client.conversations_list(
                types="public_channel, private_channel, mpim", exclude_archived=True
            )
            if response["ok"]:
                return list(map(lambda c: Channel(c["id"]), response["channels"]))
            return response

    def list_users(self) -> list[User] | SlackResponse:
        with self._mutex:
            response: SlackResponse = self._web_client.conversations_list(types="im")
            if response["ok"]:
                return list(map(lambda c: User(c["user"]), response["channels"]))
            return response

    def poll(self) -> list[ReceivedMessage]:
        with self._mutex:
            last_unread_msgs = self._unread_msgs
            self._unread_msgs = []
        return last_unread_msgs

    def send(self, message: Message, via: Channel) -> None:
        with self._mutex:
            self._web_client.chat_postMessage(
                channel=via.name,
                text=message.text,
            )


if __name__ == "__main__":
    from bro import logs
    from bro.brofiles import LOG_FILE, DB_FILE

    logs.setup(log_file=LOG_FILE, db_file=DB_FILE)
    _logger.setLevel(logging.INFO)

    connector = SlackConnector(os.environ["SLACK_BOT_TOKEN"], os.environ["SLACK_APP_TOKEN"])
    while True:
        connector.poll()
