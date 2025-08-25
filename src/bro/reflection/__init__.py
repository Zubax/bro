from __future__ import annotations
from abc import ABC, abstractmethod


class Reflection(ABC):
    """
    The Reflection module is responsible for evaluating the performance of the Reasoner
    and providing feedback for improvement.
    It analyzes past actions and outcomes to identify areas for enhancement.
    It may suggest adjustments to strategies, highlight mistakes, and recommend learning resources.
    This module helps the Reasoner to learn from experience and refine its decision-making process over time.
    """

    @abstractmethod
    def reflect(self, message: str, /) -> str:
        pass
