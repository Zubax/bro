import os
from time import sleep

from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

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
        #self.send(Message("hello", []), Channel("general"))
        self.client.socket_mode_request_listeners.append(self._process_message)
        self.client.connect()

    def _process_message(self, client: SocketModeClient, req: SocketModeRequest) -> None:
        if req.type == "events_api":
            response = SocketModeResponse(envelope_id=req.envelope_id)
            client.send_socket_mode_response(response)
            print(req)

    def list_channels(self) -> list[Channel]:
        pass

    def poll(self) -> list[ReceivedMessage]:
        raise NotImplementedError

    def send(self, message: Message, via: Channel) -> None:
        self.web_client.chat_postMessage(
            channel=via.name,
            text=message.text,
        )
