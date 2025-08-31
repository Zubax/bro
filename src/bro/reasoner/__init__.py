from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Context:
    prompt: str
    files: list[Path]


class Reasoner(ABC):
    """
    The Reasoner is responsible for planning and decision-making based on the given context,
    and controlling the Executive to perform actions in the real world.
    """

    @abstractmethod
    def task(self, ctx: Context, /) -> None:
        """
        Commence a new task with the given context.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self) -> str | None:
        """
        Perform a single reasoning-action step.
        Returns a string result if the task is complete, or None if more steps are needed.
        """
        raise NotImplementedError

    @abstractmethod
    def legilimens(self) -> str:
        """
        Provide a summary of the current internal state.
        This action does not affect the context or state.
        """
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> Any:
        """
        Capture the current state of the Reasoner for later restoration.
        The value is opaque but JSON-serializable.
        """
        raise NotImplementedError

    @abstractmethod
    def restore(self, state: Any, /) -> None:
        """
        Restore the Reasoner to a previously captured state as returned by `snapshot()`.
        """
        raise NotImplementedError
