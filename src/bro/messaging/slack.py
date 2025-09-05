import logging
import os
from time import sleep

from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web import SlackResponse

from bro.messaging import Messaging, Channel, ReceivedMessage, Message

bot_token = os.environ["SLACK_BOT_TOKEN"]
app_token = os.environ["SLACK_APP_TOKEN"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class SlackMessaging(Messaging):
    def __init__(self) -> None:
        self.web_client=WebClient(token=bot_token)
        self.client = SocketModeClient(
            app_token=app_token,
            web_client=self.web_client
        )
        self.unread_msgs: list[ReceivedMessage] = []

        self.client.socket_mode_request_listeners.append(self._process_message)
        self.client.connect()

    def _process_message(self, client: SocketModeClient, req: SocketModeRequest) -> None:
        if req.type == "events_api":
            response = SocketModeResponse(envelope_id=req.envelope_id)
            client.send_socket_mode_response(response)
            if req.payload["event"]["type"] == "message":
                attachments = []
                text = ''
                channel_id = req.payload["event"]["channel"]
                if req.payload["event"].get("text"):
                    text = req.payload["event"]["text"]
                    logging.info("Received a text message.")
                if req.payload["event"].get("subtype") == "file_share":
                    logging.info("Received one file or more.")
                    files = req.payload["event"]["files"]
                    for file in files:
                        file_info = self.web_client.files_info(file=file["id"])
                        file_download_url = file_info["file"]["url_private_download"]
                        attachments.append(file_download_url)
                logging.info("Received a total of %d files." % len(attachments))
                self.unread_msgs.append(ReceivedMessage(via=Channel(name=channel_id),
                                                        text=text,
                                                        attachments=attachments))
                # TODO fetch attachments, not just download links

    def list_channels(self) -> list[Channel] | SlackResponse:
        response: SlackResponse = self.web_client.conversations_list(
            types="public_channel, private_channel",
            exclude_archived=True
        )
        if response["ok"]:
            return list(map(lambda c: Channel(c["name"]), response["channels"]))
        return response

    def poll(self) -> list[ReceivedMessage]:
        last_unread_msgs = self.unread_msgs
        self.unread_msgs = []
        # add mutex everywhere
        return last_unread_msgs

    def send(self, message: Message, via: Channel) -> None:
        self.web_client.chat_postMessage(
            channel=via.name,
            text=message.text,
        )
