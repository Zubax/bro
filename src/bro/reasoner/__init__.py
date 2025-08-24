from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Context:
    prompt: str
    files: list[Path]


class Reasoner(ABC):
    @abstractmethod
    def run(self, ctx: Context, /) -> str:
        pass
