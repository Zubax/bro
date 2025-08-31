import os
from time import sleep
from typing import Type

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web import SlackResponse

from bro.messaging import Messaging, Channel, ReceivedMessage, Message

bot_token = os.environ["SLACK_BOT_TOKEN"]
app_token = os.environ["SLACK_APP_TOKEN"]

class SlackMessaging(Messaging):
    def __init__(self) -> None:
        self.web_client=WebClient(token=bot_token)
        self.client = SocketModeClient(
            app_token=app_token,
            web_client=self.web_client
        )
        self.unread_msgs: list[ReceivedMessage] = []

        #self.send(Message("hello", []), Channel("general"))
        #self.list_channels()
        self.client.socket_mode_request_listeners.append(self._process_message)
        self.client.connect()

    def _process_message(self, client: SocketModeClient, req: SocketModeRequest) -> None:
        if req.type == "events_api":
            response = SocketModeResponse(envelope_id=req.envelope_id)
            client.send_socket_mode_response(response)
            if req.payload["event"]["type"] == "message" \
                    and req.payload["event"].get("subtype") is None:
                print(f"Received message: {req.payload['event']['text']}")
                try:
                    self.unread_msgs.append(ReceivedMessage(via=req.payload["event"]["channel"],
                                                            text=req.payload["event"]["text"],
                                                            attachments=req.payload ["event"]["attachments"]))

                    # self.client.web_client.reactions_add(
                    #     name="eyes",
                    #     channel=req.payload["event"]["channel"],
                    #     timestamp=req.payload["event"]["ts"],
                    # )

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
        return last_unread_msgs

    def send(self, message: Message, via: Channel) -> None:
        self.web_client.chat_postMessage(
            channel=via.name,
            text=message.text,
        )

