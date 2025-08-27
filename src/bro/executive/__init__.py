from __future__ import annotations
from abc import ABC, abstractmethod
import enum


class Mode(enum.Enum):
    FAST = "fast"
    THOROUGH = "thorough"


class Executive(ABC):
    """
    The Executive is responsible for grounding the agent's actions in the real world.
    One would send requests to the agent via this interface and wait for responses describing the outcome.

    The instructions should be concise and straightforward, as the Executive may have limited reasoning capabilities
    and may be easily confused by complex instructions. It is recommended to split complex tasks into basic steps
    and check the results after each step.
    """

    @abstractmethod
    def act(self, goal: str, mode: Mode) -> str:
        """
        If switching the mode affects the reasoning level or other generation parameters of the Executive,
        then the Executive should drop the context of the current conversation to avoid context contamination.
        Earlier low-effort tokens anchor style and search, and once the model sees a short token budget,
        it rarely expands reasoning depth reliably.
        """
        raise NotImplementedError
