from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Context:
    prompt: str
    files: list[Path]


OnTaskCompleted = Callable[[str], None]


class Reasoner(ABC):
    """
    The Reasoner is responsible for planning and decision-making based on the given context,
    and controlling the Executive to perform actions in the real world.
    """

    @abstractmethod
    def task(self, ctx: Context, on_task_completed_cb: OnTaskCompleted, /) -> bool:
        """
        Commence a new task with the given context.
        The callable is invoked from a worker thread with the final response once the task is finished.
        TODO: allow the reasoner to return files and images.
        Returns True if the task is accepted,
        False if another task is still running.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Close the background thread.
        """
        raise NotImplementedError

    @abstractmethod
    def legilimens(self) -> str | None:
        """
        Provide a summary of the current internal state.
        This action does not affect the context or state.
        Returns None if there is no task at the moment.
        """
        raise NotImplementedError
