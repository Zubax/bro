import os
import logging
from threading import Lock
import tempfile
from pathlib import Path
from typing import Any

import requests
from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.client import BaseSocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web import SlackResponse

from bro.connector import Connector, Channel, ReceivedMessage, Message, User

_logger = logging.getLogger(__name__)

ATTACHMENT_FOLDER = tempfile.mkdtemp(f".conversation.pid{os.getpid()}.bro")


class SlackConnector(Connector):
    """
    SlackConnector is the logic layer that does the polling, sending, downloading attachments using the Slack Socket Mode API.
    The tokens can be obtained from Slack app's settings → Basic Information → App-Level Tokens → Generate Token.
    """

    def __init__(self, *, bot_token: str, app_token: str, bro_user_id: str) -> None:
        self._web_client = WebClient(token=bot_token)
        self._bot_token = bot_token
        self._client = SocketModeClient(app_token=app_token, web_client=self._web_client)
        self._unread_msgs: list[ReceivedMessage] = []
        self._bro_user_id: str = bro_user_id

        self._client.socket_mode_request_listeners.append(self._process_message)
        self._client.connect()
        self._mutex = Lock()

        # TODO this is a hack to filter out duplicate events from the Slack API
        self._seen_events: set[Any] = set()

    def _download_attachment(self, url: str) -> Path:
        headers = {"Authorization": f"Bearer {self._bot_token}"}
        file_location = Path(ATTACHMENT_FOLDER) / url.split("/")[-1]
        response = requests.get(url, headers=headers)
        file_location.write_bytes(response.content)
        _logger.info(f"File is saved at {file_location}")
        return file_location

    def _process_message(self, client: BaseSocketModeClient, req: SocketModeRequest) -> None:
        with self._mutex:
            event_id = req.payload["event_id"]
            if event_id in self._seen_events:
                return None
            self._seen_events.add(event_id)

            event = req.payload["event"]
            user_id = event.get("user")
            text = event.get("text") or ""
            attachments = []
            if user_id == self._bro_user_id:
                _logger.debug("Bro sent a text message: %s", text)
                return None

            subtype = event.get("subtype")
            channel_id = event.get("channel")

            match (req.type, event.get("type")):
                case ("events_api", "message"):
                    response = SocketModeResponse(envelope_id=req.envelope_id)
                    client.send_socket_mode_response(response)
                    _logger.debug(f"Received event payload: {req.payload['event']}")
                    match subtype:
                        case "file_share":
                            _logger.info("Received one attachment or more.")
                            files = event.get("files")
                            for file in files:
                                file_id = file["id"]
                                try:
                                    file_info = self._web_client.files_info(file=file_id)
                                    file_download_url = file_info["file"]["url_private_download"]
                                    file_download_path = self._download_attachment(url=file_download_url)
                                    attachments.append(file_download_path)
                                except Exception as ex:
                                    _logger.exception("Failed to save attachment for file %r: %s", file_id, ex)
                                    return None
                            _logger.info("Received a total of %d attachments." % len(attachments))
                        case None:
                            pass
                        case _:
                            _logger.debug("Ignoring message with subtype %r", subtype)
                            return None

                    user_info = self._web_client.users_info(user=user_id)["user"]
                    user_name = user_info["name"]
                    _logger.debug(f"User info: id={user_id}, name={user_name}")
                    self._unread_msgs.append(
                        ReceivedMessage(
                            via=Channel(name=channel_id),
                            user=User(name=user_name),
                            text=text,
                            attachments=attachments,
                        )
                    )
                    return None
            return None

    def list_channels(self) -> list[Channel]:
        with self._mutex:
            response: SlackResponse = self._web_client.conversations_list(
                types="public_channel, private_channel, mpim", exclude_archived=True
            )
            if response["ok"]:
                return list(map(lambda c: Channel(c["id"]), response["channels"]))
            _logger.error(response)
            return []

    def list_users(self) -> list[User]:
        with self._mutex:
            response: SlackResponse = self._web_client.conversations_list(types="im")
            if response["ok"]:
                return list(map(lambda c: User(c["user"]), response["channels"]))
            _logger.error(response)
            return []

    def poll(self) -> list[ReceivedMessage]:
        with self._mutex:
            last_unread_msgs = self._unread_msgs
            self._unread_msgs = []
        return last_unread_msgs

    def send(self, message: Message, via: Channel) -> None:
        with self._mutex:
            self._web_client.chat_postMessage(channel=via.name, text=message.text)
            _logger.info("Message is posted to the channel.")
            for file_path in message.attachments:
                try:
                    self._web_client.files_upload_v2(
                        file=file_path,
                        channel=via.name,
                    )
                    _logger.info("File is uploaded to the channel.")
                except Exception as e:
                    _logger.error(f"Can't upload file {file_path}. Exception: {e}")
                    self._web_client.chat_postMessage(channel=via.name, text=f"File upload error: {e}")

            return None


if __name__ == "__main__":
    from bro import logs
    from bro.brofiles import LOG_FILE, DB_FILE

    logs.setup(log_file=LOG_FILE, db_file=DB_FILE)
    _logger.setLevel(logging.INFO)

    connector = SlackConnector(
        bot_token=os.environ["SLACK_BOT_TOKEN"],
        app_token=os.environ["SLACK_APP_TOKEN"],
        bro_user_id=os.environ["BRO_USER_ID"],
    )
    while True:
        connector.poll()
