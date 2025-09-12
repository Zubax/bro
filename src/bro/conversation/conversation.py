import os
import logging
from bro.connector.slack import SlackConnector

_logger = logging.getLogger(__name__)
slack_bot_token, slack_app_token = os.environ["SLACK_BOT_TOKEN"], os.environ["SLACK_APP_TOKEN"]

class ConversationHandler:
    """
    This class handles receiving message from slack and replying
    """
    def __init__(self, connector: SlackConnector):
        self.connector = connector

    def start(self):
        while True:
            self.connector.poll()

    def send_message(self, message):

    def receive_message(self, message):


if __name__ == "__main__":
    connector = SlackConnector(bot_token=slack_bot_token, app_token=slack_app_token)
    conversation = ConversationHandler(connector)
    conversation.start()
