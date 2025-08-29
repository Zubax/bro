import os
import requests
from dataclasses import dataclass
from typing import Any

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

token=os.environ["SLACK_BOT_TOKEN"]=os.environ["SLACK_BOT_TOKEN"]
app = App(token=token)

@dataclass
class SlackFile:
    id: str
    name: str
    filetype: str
    user: str
    size: int

    def __init__(self, file: dict[str, Any]) -> None:
        self.id = file["id"]
        self.name = file["name"]
        self.filetype = file["filetype"]
        self.user = file["user"]
        self.size = file["size"]
        self.slack_link = file["url_private"] or file["url_private_download"]

    def get_file(self):
        data = requests.get(url=self.slack_link, headers={"Authorization": f"Bearer {token}"})
        return data.content

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
    file_id = body["event"]["file_id"]
    file_obj = app.client.files_info(file=file_id)["file"]
    file_data = SlackFile(file_obj).get_file()
    say(f"Fetched file content: {file_data}")

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()