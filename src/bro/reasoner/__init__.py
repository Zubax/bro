from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Context:
    prompt: str
    files: list[Path]


OnTaskCompleted = Callable[[str, bool], None]  # (message, scheduled)


class Reasoner(ABC):
    """
    The Reasoner is responsible for planning and decision-making based on the given context,
    and controlling the Executive to perform actions in the real world.
    """

    @abstractmethod
    def task(self, ctx: Context, /, *, scheduled: bool = False) -> bool:
        """
        Commence a new task with the given context. The callback set via on_task_completed_cb is invoked from a
        worker thread with the final response once the task is finished.
        TODO: allow the reasoner to return files and images.
        Args:
            ctx: The task context
            scheduled: If True, this is a scheduled task (won't trigger user notification callback)
        Returns True if the task is accepted, False if another task is still running.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def on_task_completed_cb(self) -> OnTaskCompleted:
        raise NotImplementedError

    @on_task_completed_cb.setter
    @abstractmethod
    def on_task_completed_cb(self, value: OnTaskCompleted) -> None:
        raise NotImplementedError

    @abstractmethod
    def abort(self) -> None:
        """
        Abort the currently running task immediately.
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

    @abstractmethod
    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return the list of tools available to this reasoner.
        Used by the conversation handler to inform its decisions about task delegation.
        """
        raise NotImplementedError
