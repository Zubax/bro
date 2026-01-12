from __future__ import annotations
from abc import ABC, abstractmethod


class KnowledgeBase(ABC):
    """
    Abstract base class for knowledge base providers.
    Knowledge bases provide access to various information sources like wikis, email, documents, etc.
    """

    @abstractmethod
    def search(self, query: str) -> str:
        """
        Search the knowledge base for information matching the query.
        Returns a string representation of the search results.
        """
        raise NotImplementedError
