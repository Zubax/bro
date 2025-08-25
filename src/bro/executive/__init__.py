from __future__ import annotations
from abc import ABC, abstractmethod


class Executive(ABC):
    """
    The Executive is responsible for grounding the agent's actions in the real world.
    One would send requests to the agent via this interface and wait for responses describing the outcome.

    The instructions should be concise and straightforward, as the Executive may have limited reasoning capabilities
    and may be easily confused by complex instructions. It is recommended to split complex tasks into basic steps
    and check the results after each step.
    """

    @abstractmethod
    def act(self, goal: str) -> str:
        raise NotImplementedError
