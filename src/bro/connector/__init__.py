from __future__ import annotations
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Channel:
    """
    Represents either a group channel or a direct message channel. For example, `#general`.
    """

    name: str


@dataclass(frozen=True)
class User:
    """
    Represents either a user or a direct message channel. For example, `@pavel.kirenko`.
    """

    name: str


@dataclass(frozen=True)
class Message:
    """
    Attachments point to local files, possibly temporary.
    When receiving a message, its attachments need to be downloaded to the local filesystem with the file names
    preserved.
    """

    text: str
    attachments: list[Path]


@dataclass(frozen=True)
class Task:
    """
    Remember the channel so that the bot could send updates about the task when it's finished.
    """
    channel: Channel
    summary: str


@dataclass(frozen=True)
class ReceivedMessage(Message):
    via: Channel
    user: User


class MessagingError(Exception):
    """For now, there are no specialized exceptions, but we may add them later; e.g., recipient unknown, etc."""

    pass


class Connector(ABC):
    @abstractmethod
    def list_channels(self) -> list[Channel]:
        """Both group chats and channels."""
        raise NotImplementedError

    def list_dms(self) -> list[User]:
        """Only direct messages"""
        raise NotImplementedError

    @abstractmethod
    def poll(self) -> list[ReceivedMessage]:
        """Non-blockingly check for new messages; return all at once."""
        raise NotImplementedError

    @abstractmethod
    def send(self, message: Message, via: Channel) -> None:
        """Submit message for transmission."""
        raise NotImplementedError
