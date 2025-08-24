from __future__ import annotations
from abc import ABC, abstractmethod


class Executive(ABC):
    @abstractmethod
    def act(self, goal: str) -> str:
        raise NotImplementedError
