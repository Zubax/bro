"""
Knowledge base module for accessing various information sources.

This module provides access to different knowledge sources like wikis, email, documents, etc.
"""

from bro.knowledgebase.wiki import WikiClient, tools as wiki_tools

__all__ = [
    "WikiClient",
    "wiki_tools",
]
