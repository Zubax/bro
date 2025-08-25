from __future__ import annotations
from abc import ABC, abstractmethod
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
    def run(self, ctx: Context, /) -> str:
        pass
