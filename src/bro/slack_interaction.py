import os
from dataclasses import dataclass
from typing import Any

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.web import SlackResponse

app = App(token=os.environ["SLACK_BOT_TOKEN"])

@dataclass
class SlackFile:
    id: str
    name: str
    filetype: str
    user: str
    size: int

    def __init__(self, file: SlackResponse) -> None:
        self.id = file["id"]
        self.name = file["name"]
        self.filetype = file["filetype"]
        self.user = file["user"]

@dataclass
class SlackMessage:
    text: str = None
    sender: str = None
    channel: str = None

    def __init__(self, message: dict[str, Any]) -> None:
        self.text = message["text"]
        self.sender = message["user"]
        self.channel = message["channel"]

@app.event("message")
def handle_text_message(message, say):
    msg = SlackMessage(message)
    say(f"Hey there <@{msg.sender}>!")

@app.event("file_shared")
def handle_file_attachment(body, say):
    say(f"Thanks for the file!")
    file_id = body["event"]["file_id"]
    file = app.client.files_info(file=file_id)
    return SlackFile(file=file)

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()