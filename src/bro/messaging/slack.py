import logging
import os
import sys
from asyncio import Lock
import tempfile
from pathlib import Path

import requests
from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web import SlackResponse

from bro.messaging import Messaging, Channel, ReceivedMessage, Message

bot_token = os.environ["SLACK_BOT_TOKEN"]
app_token = os.environ["SLACK_APP_TOKEN"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _download_attachment(url: str) -> Path | None:
    attachment_folder = tempfile.mkdtemp()
    file_location: Path | None = None
    try:
        response = requests.get(url)
        file_location = Path(attachment_folder) / url.split("/")[-1]
        file_location.write_bytes(response.content)
        logging.info(f"File is saved at {file_location}")
    except:
        logging.error("Unexpected error: %s", sys.exc_info()[0])
    return file_location


class SlackMessaging(Messaging):
    def __init__(self) -> None:
        self.web_client = WebClient(token=bot_token)
        self.client = SocketModeClient(app_token=app_token, web_client=self.web_client)
        self.unread_msgs: list[ReceivedMessage] = []

        self.client.socket_mode_request_listeners.append(self._process_message)
        self.client.connect()
        self.mutex = Lock()

    def _process_message(self, client: SocketModeClient, req: SocketModeRequest) -> None:
        with self.mutex:
            if req.type == "events_api":
                response = SocketModeResponse(envelope_id=req.envelope_id)
                client.send_socket_mode_response(response)
                if req.payload["event"]["type"] == "message":
                    attachments = []
                    text = ""
                    channel_id = req.payload["event"]["channel"]
                    if req.payload["event"].get("text"):
                        text = req.payload["event"]["text"]
                        logging.info("Received a text message.")
                    if req.payload["event"].get("subtype") == "file_share":
                        logging.info("Received one attachment or more.")
                        files = req.payload["event"]["files"]
                        for file in files:
                            file_info = self.web_client.files_info(file=file["id"])
                            file_download_url = file_info["file"]["url_private_download"]
                            file_download_path = _download_attachment(url=file_download_url)
                            if file_download_path:
                                attachments.append(file_download_path)
                    logging.info("Received a total of %d attachments." % len(attachments))
                    self.unread_msgs.append(
                        ReceivedMessage(via=Channel(name=channel_id), text=text, attachments=attachments)
                    )

    def list_channels(self) -> list[Channel] | SlackResponse:
        with self.mutex:
            response: SlackResponse = self.web_client.conversations_list(
                types="public_channel, private_channel", exclude_archived=True
            )
            if response["ok"]:
                return list(map(lambda c: Channel(c["name"]), response["channels"]))
            return response

    def poll(self) -> list[ReceivedMessage]:
        with self.mutex:
            last_unread_msgs = self.unread_msgs
            self.unread_msgs = []
            return last_unread_msgs

    def send(self, message: Message, via: Channel) -> None:
        with self.mutex:
            self.web_client.chat_postMessage(
                channel=via.name,
                text=message.text,
            )
